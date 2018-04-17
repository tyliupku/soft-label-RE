#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-3-21 下午1:50
# @Author  : Tianyu Liu

import tensorflow as tf
import numpy as np
import util, sys



class model(object):
    def __init__(self, pad_len, num_rels, word_vectors, window_size, num_filters, embedding_size, pos_embedding, dropout, batch_num, joint_p, l2_reg=0.0):
        self.input = tf.placeholder(tf.int32, [None, pad_len], name="input")
        self.preds = tf.placeholder(tf.int32, [None, num_rels], name="preds")
        self.num_filters = num_filters
        self.mask1 = tf.placeholder(tf.float32, [None, pad_len - window_size + 1, self.num_filters], name="mask_before")
        self.mask2 = tf.placeholder(tf.float32, [None, pad_len - window_size + 1, self.num_filters], name="mask_between")
        self.mask3 = tf.placeholder(tf.float32, [None, pad_len - window_size + 1, self.num_filters], name="mask_after")
        self.wps1 = tf.placeholder(tf.int32, [None, pad_len], name="wps1")
        self.wps2 = tf.placeholder(tf.int32, [None, pad_len], name="wps2")
        self.pad_len = pad_len
        self.window_size = window_size
        self.num_rels = num_rels
        self.PAD = len(word_vectors) - 1
        self.bag_num = tf.placeholder(tf.int32, [batch_num + 1], name="bag_num")
        self.soft_label_flag = tf.placeholder(tf.float32, [batch_num], name="soft_label_flag")
        self.joint_p = joint_p
        total_num = self.bag_num[-1]
        self.batch_num = batch_num
        l2_loss = tf.constant(0.0)

        with tf.device('/cpu:0'):
            self.embedding = tf.Variable(word_vectors, dtype=tf.float32)
            self.inputs = tf.nn.embedding_lookup(self.embedding, self.input)
        with tf.name_scope('joint'):
            wpe1 = tf.get_variable("wpe1", shape=[62, pos_embedding], initializer=tf.contrib.layers.xavier_initializer())
            wpe2 = tf.get_variable("wpe2", shape=[62, pos_embedding], initializer=tf.contrib.layers.xavier_initializer())
            pos_left = tf.nn.embedding_lookup(wpe1, self.wps1)
            pos_right = tf.nn.embedding_lookup(wpe2, self.wps2)
            self.pos_embed = tf.concat([pos_left, pos_right], 2)
        with tf.name_scope('conv'):
            self._input = tf.concat([self.inputs, self.pos_embed], 2)
            filter_shape = [window_size, embedding_size + 2*pos_embedding, 1, num_filters]
            W = tf.get_variable("conv-W", shape=filter_shape, initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable("conv-b", shape=[num_filters], initializer=tf.contrib.layers.xavier_initializer())
            self.conv = tf.nn.conv2d(tf.expand_dims(self._input, -1), W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
            h = tf.nn.tanh(tf.nn.bias_add(self.conv, b), name="tanh")
            self.h1 = tf.add(h, tf.expand_dims(self.mask1, 2))
            self.h2 = tf.add(h, tf.expand_dims(self.mask2, 2))
            self.h3 = tf.add(h, tf.expand_dims(self.mask3, 2))
            pooled1 = tf.nn.max_pool(self.h1, ksize=[1, self.pad_len - self.window_size + 1, 1, 1], strides=[1, 1, 1, 1], padding="VALID",name="pool")
            poolre1 = tf.reshape(pooled1, [-1, self.num_filters])
            pooled2 = tf.nn.max_pool(self.h2, ksize=[1, self.pad_len - self.window_size + 1, 1, 1], strides=[1, 1, 1, 1], padding="VALID",name="pool")
            poolre2 = tf.reshape(pooled2, [-1, self.num_filters])
            pooled3 = tf.nn.max_pool(self.h3, ksize=[1, self.pad_len - self.window_size + 1, 1, 1], strides=[1, 1, 1, 1], padding="VALID",name="pool")
            poolre3 = tf.reshape(pooled3, [-1, self.num_filters])
            poolre = tf.concat([poolre1, poolre2, poolre3], 1)
            pooled = tf.nn.dropout(poolre, dropout)
        with tf.name_scope("map"):
            W = tf.get_variable(
                "W",
                shape=[3*self.num_filters, self.num_rels],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable("b", shape=[self.num_rels], initializer=tf.contrib.layers.xavier_initializer())
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)

            # the implementation of Lin et al 2016 comes from https://github.com/thunlp/TensorFlow-NRE/blob/master/network.py
            sen_a = tf.get_variable("attention_A", [3*self.num_filters], initializer=tf.contrib.layers.xavier_initializer())
            sen_q = tf.get_variable("query", [3*self.num_filters, 1], initializer=tf.contrib.layers.xavier_initializer())
            sen_r = []
            sen_s = []
            sen_out = []
            sen_alpha = []
            self.bag_score = []
            self.predictions = []
            self.losses = []
            self.accuracy = []
            self.total_loss = 0.0
            # selective attention model, use the weighted sum of all related the sentence vectors as bag representation
            for i in range(batch_num):
                sen_r.append(pooled[self.bag_num[i]:self.bag_num[i+1]])
                bag_size = self.bag_num[i+1] - self.bag_num[i]
                sen_alpha.append(tf.reshape(tf.nn.softmax(tf.reshape(tf.matmul(tf.multiply(sen_r[i], sen_a), sen_q), [bag_size])), [1, bag_size]))
                sen_s.append(tf.reshape(tf.matmul(sen_alpha[i], sen_r[i]), [1, 3*self.num_filters]))
                sen_out.append(tf.reshape(tf.nn.xw_plus_b(sen_s[i], W, b), [self.num_rels]))
                self.bag_score.append(tf.nn.softmax(sen_out[i]))

                with tf.name_scope("output"):
                    self.predictions.append(tf.argmax(self.bag_score[i], 0, name="predictions"))

                with tf.name_scope("loss"):

                    nscor = self.soft_label_flag[i] * self.bag_score[i] + joint_p * tf.reduce_max(self.bag_score[i])* tf.cast(self.preds[i], tf.float32)
                    self.nlabel = tf.reshape(tf.one_hot(indices=[tf.argmax(nscor, 0)], depth=self.num_rels, dtype=tf.int32), [self.num_rels])
                    self.ccc = self.preds[i]
                    self.losses.append(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=sen_out[i], labels=self.nlabel)))

                    if i == 0:
                        self.total_loss = self.losses[i]
                    else:
                        self.total_loss += self.losses[i]

                with tf.name_scope("accuracy"):
                    self.accuracy.append(tf.reduce_mean(tf.cast(tf.equal(self.predictions[i], tf.argmax(self.preds[i], 0)), "float"), name="accuracy"))


        with tf.name_scope("update"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            self.train_op = optimizer.minimize(self.total_loss, global_step=self.global_step)

    # pad sentences for piecewise max-pooling operation described in 
    # "Distant Supervision for Relation Extraction via Piecewise Convolutional Neural Networks"
    def sen_padding(self, sen_id, instance, lpos, rpos, real_sen, namepos):
        # instance :[5, 233, 3232, ...] sentences in bag
        instances = []
        mask_before = []
        mask_between= []
        mask_after = []
        lwps = []
        rwps = []
        for id in instance:
            seq = sen_id[id]
            wps_left = lpos[id]
            wps_right = rpos[id]
            en1, en2 = namepos[id]
            t1, t2 = self.get_split(en1, en2)
            seq_len = len(real_sen[id].split())
            assert seq_len <= self.pad_len
            if seq_len <= self.pad_len:
                mask1 = np.zeros([self.pad_len-self.window_size+1, self.num_filters], dtype=float)
                mask1[t1+1:, :] = -100.0
                mask2 = np.zeros([self.pad_len-self.window_size+1, self.num_filters], dtype=float)
                mask2[:t1, :] = -100.0
                mask2[t2+1:, :] = -100.0
                mask3 = np.zeros([self.pad_len - self.window_size + 1, self.num_filters], dtype=float)
                mask3[:t2, :] = -100.0
                mask3[seq_len-self.window_size+1:, :] = -100.0
                # mask = [1] * (seq_len-self.window_size+1) + [0] * (self.pad_len-seq_len)
            if len(seq) < self.pad_len:
                llen = self.pad_len - len(seq)
                seq.extend([self.PAD] * (self.pad_len - len(seq)))
                wps_left.extend([61] * llen)
                wps_right.extend([61] * llen)
            mask_before.append(mask1)
            mask_between.append(mask2)
            mask_after.append(mask3)
            instances.append(seq)
            lwps.append(wps_left)
            rwps.append(wps_right)
        return instances, mask_before, mask_between, mask_after, lwps, rwps

    def get_split(self, en1, en2):
        t1, t2 = en1, en2
        if en1 > en2:
            t1 = en2
            t2 = en1
        assert t1 <= t2
        return t1,t2


    def train(self, sess, bag_key, train_bag, sen_id, lpos, rpos, real_sen, namepos, use_soft_label = False):
        # bag_key: mid1 mid2 rel
        batch = []
        pred = []
        mask_before = []
        mask_between = []
        mask_after = []
        wps_left = []
        wps_right = []
        batch_sen_num = []
        soft_label_flag = []
        cnt_sen = 0
        for key in bag_key:
            rel = int(key.split('\t')[-1])
            if use_soft_label:soft_label_flag.append(1)
            else:soft_label_flag.append(0)
            sentences = train_bag[key]
            sen_vec, mask_bef, mask_bet, mask_aft, llpos, rrpos = self.sen_padding(sen_id, sentences, lpos, rpos, real_sen, namepos)
            batch.extend(sen_vec)
            mask_before.extend(mask_bef)
            mask_between.extend(mask_bet)
            mask_after.extend(mask_aft)
            pred.append(rel)
            wps_left.extend(llpos)
            wps_right.extend(rrpos)
            batch_sen_num.append(cnt_sen)
            cnt_sen += len(sentences)
        batch_sen_num.append(cnt_sen)
        preds = np.zeros([len(bag_key), self.num_rels])
        preds[np.arange(len(bag_key)), pred] = 1
        _, hh, loss, acc, step = sess.run([self.train_op, self.h1, self.total_loss, self.accuracy, self.global_step], feed_dict={
            self.input: batch,
            self.mask1: mask_before,
            self.mask2: mask_between,
            self.mask3: mask_after,
            self.preds: preds,
            self.wps1: wps_left,
            self.wps2: wps_right,
            self.bag_num: batch_sen_num,
            self.soft_label_flag: soft_label_flag
        })

        assert np.min(np.max(hh, axis=1)) > -50.0
        acc = np.reshape(np.array(acc), (self.batch_num))
        acc = np.mean(acc)
        return loss

    def test(self, sess, bag_key, test_bag, sen_id, lpos, rpos, real_sen, namepos):
        # bag_key: mid1 mid2
        pair_score = []
        cnt_i = 1
        batches = util.batch_iter(bag_key, self.batch_num, 1, shuffle=True)
        for bat in batches:
            if len(bat) < self.batch_num:
                continue
            batch = []
            mask_before = []
            mask_between = []
            mask_after = []
            wps_left = []
            wps_right = []
            batch_sen_num = []
            cnt_sen = 0
            for key in bat:
                # sys.stdout.write('testing %d cases...\r' % cnt_i
                # sys.stdout.flush()
                cnt_i += 1
                sentences = test_bag[key]
                sen_vec, mask_bef, mask_bet, mask_aft, llpos, rrpos = self.sen_padding(sen_id, sentences, lpos, rpos, real_sen, namepos)
                batch.extend(sen_vec)
                mask_before.extend(mask_bef)
                mask_between.extend(mask_bet)
                mask_after.extend(mask_aft)
                wps_left.extend(llpos)
                wps_right.extend(rrpos)
                batch_sen_num.append(cnt_sen)
                cnt_sen += len(sentences)
            batch_sen_num.append(cnt_sen)
            soft_label_flag = [0] * len(bat)

            scores = sess.run(self.bag_score, feed_dict={
                self.input: batch,
                self.mask1: mask_before,
                self.mask2: mask_between,
                self.mask3: mask_after,
                self.wps1: wps_left,
                self.wps2: wps_right,
                self.bag_num: batch_sen_num,
                self.soft_label_flag: soft_label_flag})
            # score = np.max(scores, axis=0)
            for k, key in enumerate(bat):
                for i, sc in enumerate(scores[k]):
                    if i==0: continue
                    pair_score.append({"mid": key, "rel": i, "score": sc})
        return pair_score


