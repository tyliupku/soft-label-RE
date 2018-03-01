from util import *
import attmodel as attmodel
import onemodel as onemodel
import tensorflow as tf
import os, time



tf.app.flags.DEFINE_integer("pad_len", 200, "Pad sentences to this length for convolution.")
tf.app.flags.DEFINE_integer("embedding_size", 50, "Size of word embedding.")
tf.app.flags.DEFINE_integer("pos_embedding", 5, "Size of position embedding.")
tf.app.flags.DEFINE_integer("batch_num", 160, "Batch size for sentence encoding.")
tf.app.flags.DEFINE_integer("num_rels", 53, "Number of pre-defined relations.")
tf.app.flags.DEFINE_integer("window_size", 3, "Size of sliding window.")
tf.app.flags.DEFINE_integer("num_filters", 230, "Number of filters for convolution.")
tf.app.flags.DEFINE_float("dropout", 0.7,'dropout')

tf.app.flags.DEFINE_string("one_or_att",'one','at-least-one or selective attention model')
tf.app.flags.DEFINE_boolean("use_pre_train_model", False,'use pre-trained model or label') 
tf.app.flags.DEFINE_string("load_model_name", 'pretrain/model.ckpt-3300','the path of pre-trained model without soft-label')
tf.app.flags.DEFINE_boolean("save_model", False,'save models or not')

tf.app.flags.DEFINE_boolean("use_soft_label", True,'use soft label or not')
tf.app.flags.DEFINE_float("confidence", 0.9,'confidence of distant-supervised label')

tf.app.flags.DEFINE_string("dir",'res','dir to store results')
tf.app.flags.DEFINE_integer("report", 100, "report loss & save models after every *** batches.")
FLAGS = tf.app.flags.FLAGS



# =================== make new dirs =================== #
prefix = str(int(time.time() * 1000))
top_dir = os.path.join(FLAGS.dir, prefix) # dir to save all the results in this run
if not os.path.exists(FLAGS.dir):
    os.mkdir(FLAGS.dir)
if not os.path.exists(top_dir):
    os.mkdir(top_dir)
checkpoint_dir = os.path.join(top_dir, "checkpoint") # dir to save models
log_file = os.path.join(top_dir, 'log.txt')
def write_log(s):
    print(s)
    with open(log_file, 'a') as f:
        f.write(s+'\n')

# =================== load data =================== #
print("load training and testing data ...")
start_time = time.time()
vect = word2vec() # load pre-trained word vector 
word_vocab, word_vector = get_word_vec(vect, one_or_att=FLAGS.one_or_att) # load vocabulary and pre-defined word vectors
'''
bag_train: a dict , key is triple (h, r, t), related value is the list of sentence ids which contain the triple.
sen_id: idlized sentences in the training data. 
real_sen: original sentences in the training set.
lpos/ rpos: the distance of each token to the head/tail entities, for position embedding.
keypos: the position of two key (head and tail) entities in the sentences.
'''
bag_train, sen_id, lpos, rpos, real_sen, keypos = get_data(istrain=True, word_vocab=word_vocab)
bag_test, sen_id1, midrel, ltpos, rtpos, real_sen1, keypos1 = get_data(istrain=False, word_vocab=word_vocab)
bag_keys = bag_train.keys()

span = time.time() - start_time
print("training and testing data loaded, using %.3f seconds" % span)
write_log("training size: %d   testing size: %d" % (len(bag_train.keys()), len(bag_test.keys())) )


# =================== model initialization =================== #
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
if FLAGS.one_or_att == 'att':
    load_model = attmodel
else:
    load_model = onemodel
model = load_model.model(pad_len=FLAGS.pad_len,
                         num_rels=FLAGS.num_rels,
                         word_vectors=word_vector,
                         window_size=FLAGS.window_size,
                         num_filters=FLAGS.num_filters,
                         embedding_size=FLAGS.embedding_size,
                         dropout=FLAGS.dropout,
                         pos_embedding=FLAGS.pos_embedding,
                         batch_num=FLAGS.batch_num,
                         joint_p=FLAGS.confidence)

saver = tf.train.Saver(max_to_keep=70)
if FLAGS.use_pre_train_model:
    saver.restore(sess, FLAGS.load_model_name)
    write_log("load pre-trained model from " + FLAGS.load_model_name)
# ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
else:
    sess.run(tf.global_variables_initializer())
    write_log("create new model")
print "Initial model complete"


# =================== training stage =================== #
batches = batch_iter(bag_train.keys(), FLAGS.batch_num, 20)
loss, start_time = 0.0, time.time()
for batch in batches:
    if len(batch) < FLAGS.batch_num:
        continue
    loss += model.train(sess, batch, bag_train, sen_id, lpos, rpos, real_sen, keypos, FLAGS.use_soft_label)
    step = tf.train.global_step(sess, model.global_step)
    progress_bar(step % FLAGS.report, FLAGS.report)
    if step % FLAGS.report == 0: # report PR-curve results on the testing set
        cost_time = time.time() - start_time
        epoch = step // FLAGS.report
        write_log("%d : loss = %.3f, time = %.3f " % (step // FLAGS.report, loss, cost_time))
        print "evaluating after epoch " + str(epoch)
        pair_score = model.test(sess, bag_test.keys(), bag_test, sen_id1, ltpos, rtpos, real_sen1, keypos1)
        evaluate(top_dir + "/pr"+str(epoch)+".txt", pair_score, midrel, epoch)
        loss, start_time = 0.0, time.time()
        if FLAGS.save_model:
            checkpoint_path = os.path.join(checkpoint_dir, "model.ckpt")
            saver.save(sess, checkpoint_path, global_step=model.global_step)
            write_log("save model in " + str(sess.run(model.global_step)))
