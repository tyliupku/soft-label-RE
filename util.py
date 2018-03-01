import re
import numpy as np
import time, os, sys, shutil

# =================== word vector =================== #
def get_lower_sentence(sentence):
    sentence = sentence.strip()
    sentence = sentence.lower()
    return sentence

# load the same pre-trained vector from https://github.com/thunlp/NRE/tree/master/data
def word2vec():
    fvec=open("data/vector1.txt", "r")
    vec={}
    for line in fvec:
        line=line.strip()
        line=line.split('\t')
        vec[line[0]]=line[1:51]
    fvec.close()
    return vec


def get_word_vec(vectors, one_or_att="att"):
    """
    :param vectors:
    :return: word_vocab: word -> id
             word_vectors: numpy
    """
    sorted_vectors = sorted(vectors.items(), key=lambda d:d[0])
    word_vocab = {}
    word_vectors = []
    for it in sorted_vectors:
        word_vocab[it[0]] = len(word_vocab)
        word_vectors.append(it[1])
    assert len(word_vocab)==len(word_vectors)
    word_vocab['UNK'] = len(word_vocab)
    # word_vocab['BLANK'] = len(word_vocab)
    word_vectors.append(np.random.normal(size=50, loc=0, scale=0.05))
    if one_or_att == "att":
        word_vocab['BLANK'] = len(word_vocab)
        word_vectors.append(np.random.normal(size=50, loc=0, scale=0.05))
    # word_vectors.append(np.random.normal(size=50, loc=0, scale=0.05))
    word_vectors = np.array(word_vectors, dtype=float)
    return word_vocab, word_vectors


def get_sentence_seq(sentence, word_vocab):
    vec = []
    words = sentence.split()
    for word in words:
        try:
            id = word_vocab[word]
        except:
            id = word_vocab["UNK"]
        vec.append(id)
    return vec


# =================== load data =================== #
def get_data(istrain=True, word_vocab=None):
    frel = open("data/RE/relation2id.txt")
    rel2id = {}

    for line in frel:
        line = line.strip()
        it = line.split()
        rel2id[it[0]] = it[1]

    if istrain:
        file = open("data/RE/train.txt")
    else:
        file = open("data/RE/test.txt")

    sen_id = []
    real_sen = []
    lpos = []
    rpos = []
    namepos = []
    bag_id = {}
    midrel = {}
    cnt = 0
    for line in file:
        line = line.strip()
        line = line[:-10].strip()
        it = line.split('\t')
        mid1, mid2, rel, sen = it[0], it[1], it[-2], get_lower_sentence(it[-1])

        if rel not in rel2id:
            continue
        rel = rel2id[rel]

        if len(sen.split()) > 200:
            continue

        name1, name2 = it[2], it[3]
        en1, en2, wps_left, wps_right = get_position(sen, name1, name2)
        if en1 == 0 or en2 == 0:
            cnt += 1
            continue
        name_set = key_position(sen, en1, en2)
        if istrain:
            key = mid1 + '\t' + mid2 + '\t' + rel
        else:
            key = mid1 + '\t' + mid2
            key1 = mid1 + '\t' + mid2 + '\t' + rel
            if rel != "0":
                midrel[key1] = 1

        if key not in bag_id:
            bag_id[key] = []
        bag_id[key].append(len(sen_id))
        lpos.append(wps_left.tolist())
        rpos.append(wps_right.tolist())
        sen_id.append(get_sentence_seq(sen, word_vocab))
        real_sen.append(sen)
        namepos.append(name_set)
    if istrain:
        return bag_id, sen_id, lpos, rpos, real_sen, namepos
    else:
        return bag_id, sen_id, midrel, lpos, rpos, real_sen, namepos


def key_position(sen, en1, en2):
    sen_len = len(sen.split())
    en = [en1, en2]
    ean = []
    for eni in en:
        eans = 0
        if eni == 1:
            eans = 0
        elif eni == sen_len:
            eans = eni - 3
        else:
            eans = eni - 2
        ean.append(eans)
    return ean


def get_position(sen, name1, name2):
    sentence = sen.split()
    llen = len(sentence)
    wps_left = np.array(range(1, llen + 1))
    wps_right = np.array(range(1, llen + 1))
    en1, en2 = 0, 0
    for i, it in enumerate(sentence):
        if it == name1:
            en1 = i + 1
        if it == name2:
            en2 = i + 1
    wps_left -= int(en1)
    wps_right -= int(en2)
    wps_left = np.minimum(np.maximum(wps_left, -30), 30) + 30
    wps_right = np.minimum(np.maximum(wps_right, -30), 30) + 30
    return en1, en2, wps_left, wps_right


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]



# =================== evaluation =================== #
def evaluate(path, triple_list, midrel, epoch):
    frel = open("data/RE/relation2id.txt")
    fmid = open("data/RE/entity2name.txt")

    rel2id = {}
    id2rel = {}
    mid2name = {}
    triple = []
    for line in frel:
        line = line.strip()
        it = line.split()
        rel2id[it[0]] = it[1]
        id2rel[int(it[1])] = it[0]
    for line in fmid:
        line = line.strip()
        it = line.split('\t')
        mid2name[it[0]] = it[1]

    for item in triple_list:
        mid = item["mid"].split('\t')
        rel = item["rel"]
        score = item["score"]
        ent1, ent2 = mid2name[mid[0]], mid2name[mid[1]]
        rname = id2rel[rel]
        key = ent1 + '\t' + ent2 + '\t' + rname
        mid_key = mid[0] + '\t' + mid[1] + '\t' + str(rel)
        crt = "0"
        if mid_key in midrel:
            crt = "1"
        triple.append({"triple":key, "val":score, "crt":crt})
    sorted_triple = sorted(triple, key=lambda x: x["val"])
    prfile = open(path, "w")
    correct = 0
    tot_recall = len(midrel.keys())
    for i, item in enumerate(sorted_triple[::-1]):
        if str(item["crt"]) == "1":
            correct += 1
        prfile.write("{0:.5f}\t{1:.5f}\t{2:.5f}\t".format(float(correct)/(i+1), float(correct)/tot_recall, float(item["val"])))
        prfile.write(str(item["triple"])+'\n')
        if i+1 > 2000:
            break
    prfile.close()


# Progress bar

TOTAL_BAR_LENGTH = 100.
last_time = time.time()
begin_time = last_time
# print os.popen('stty size', 'r').read()
_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

