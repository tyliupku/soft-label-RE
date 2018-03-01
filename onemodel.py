#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-3-27 上午10:39
# @Author  : Tianyu Liu
import tensorflow as tf
import numpy as np
import sys


class model(object):
    def __init__(self, pad_len, num_rels, word_vectors, window_size, num_filters, embedding_size, pos_embedding, dropout, joint_p, batch_num=None, l2_reg=0.0):
        self.input = tf.placeholder(tf.int32, [None, pad_len], name="input")
        self.preds = tf.placeholder(tf.int32, [None, num_rels], name="preds")
        self.num_filters = num_filters
        self.mask1 = tf.placeholder(tf.float32, [None, pad_len - window_size + 1, self.num_filters], name="mask_before")
        self.mask2 = tf.placeholder(tf.float32, [None, pad_len - window_size + 1, self.num_filters], name="mask_between")
        self.mask3 = tf.placeholder(tf.float32, [None, pad_len - window_size + 1, self.num_filters], name="mask_after")
        self.wps1 = tf.placeholder(tf.float32, [None, pad_len, 61], name="wps1")
        self.wps2 = tf.placeholder(tf.float32, [None, pad_len, 61], name="wps2")
        self.pad_len = pad_len
        self.window_size = window_size
        self.num_rels = num_rels
        self.PAD = len(word_vectors) - 1
        self.joint_p = joint_p
        self.soft_label_flag = tf.placeholder(tf.float32, [None], name="soft_label_flag")
        l2_loss = tf.constant(0.0)

        with tf.device('/cpu:0'):
            self.embedding = tf.Variable(word_vectors, dtype=tf.float32)
            self.inputs = tf.nn.embedding_lookup(self.embedding, self.input)
        with tf.name_scope('joint'):
            wpe1 = tf.Variable(tf.truncated_normal([61, pos_embedding], stddev=0.01), name="wpe1")
            wpe2 = tf.Variable(tf.truncated_normal([61, pos_embedding], stddev=0.01), name="wpe2")
            pos_left = tf.reshape(tf.matmul(tf.reshape(self.wps1, [-1, 61]), wpe1), [-1, pad_len, pos_embedding])
            pos_right = tf.reshape(tf.matmul(tf.reshape(self.wps2, [-1, 61]), wpe2), [-1, pad_len, pos_embedding])
            self.pos_embed = tf.concat([pos_left, pos_right], 2)
        with tf.name_scope('conv'):
            self._input = tf.concat([self.inputs, self.pos_embed], 2)
            filter_shape = [window_size, embedding_size + 2*pos_embedding, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.01, shape=[num_filters]), name="b")
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
            self.pooled = tf.nn.dropout(poolre, dropout)
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[3*self.num_filters, self.num_rels],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.num_rels]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.linear_scores = tf.nn.xw_plus_b(self.pooled, W, b, name="scores")
            self.scores = tf.nn.softmax(self.linear_scores)
            nscore = tf.expand_dims(self.soft_label_flag, -1) * self.scores + self.joint_p * tf.reshape(tf.reduce_max(self.scores, 1), [-1, 1]) * tf.cast(self.preds, tf.float32)
            self.nlabel = tf.one_hot(indices=tf.reshape(tf.argmax(nscore, axis=1), [-1]), depth=self.num_rels, dtype=tf.int32)
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.linear_scores, labels=self.nlabel)
            self.loss = tf.reduce_mean(losses) + l2_reg * l2_loss
        with tf.name_scope("update"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

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
            llwps, rrwps = self.pos_padding(wps_left, wps_right, seq_len)
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
                seq.extend([self.PAD] * (self.pad_len - len(seq)))
            mask_before.append(mask1)
            mask_between.append(mask2)
            mask_after.append(mask3)
            instances.append(seq)
            lwps.append(llwps)
            rwps.append(rrwps)
        return instances, mask_before, mask_between, mask_after, lwps, rwps

    def get_split(self, en1, en2):
        t1, t2 = en1, en2
        if en1 > en2:
            t1 = en2
            t2 = en1
        assert t1 <= t2
        return t1,t2

    def pos_padding(self, wps_left, wps_right, llen):
        pos_left = np.zeros([llen, 61], dtype=int)
        pos_right = np.zeros([llen, 61], dtype=int)
        pos_left[np.arange(llen), wps_left] = 1
        pos_right[np.arange(llen), wps_right] = 1
        if llen < self.pad_len:
            pad = np.zeros([self.pad_len-llen, 61], dtype=int)
            pad[np.arange(self.pad_len-llen), [60]*(self.pad_len-llen)] = 1
            pos_left = np.concatenate((pos_left, pad), axis=0)
            pos_right = np.concatenate((pos_right, pad), axis=0)
        return pos_left, pos_right

    def train(self, sess, bag_key, train_bag, sen_id, lpos, rpos, real_sen, namepos, use_soft_label=False):
        # bag_key: mid1 mid2 rel
        batch = []
        pred = []
        mask_before = []
        mask_between = []
        mask_after = []
        wps_left = []
        wps_right = []
        soft_label_flag = []
        for key in bag_key:
            rel = int(key.split('\t')[-1])
            if use_soft_label:soft_label_flag.append(1)
            else:soft_label_flag.append(0)
            sentences = train_bag[key]
            sen_vec, mask_bef, mask_bet, mask_aft, llpos, rrpos = self.sen_padding(sen_id, sentences, lpos, rpos, real_sen, namepos)
            scores = sess.run(self.linear_scores, feed_dict={self.input: sen_vec, self.mask1: mask_bef, self.mask2: mask_bet, self.mask3: mask_aft, self.wps1: llpos, self.wps2: rrpos})
            id_max = np.argmax(scores[:, rel])
            batch.append(sen_vec[id_max])
            mask_before.append(mask_bef[id_max])
            mask_between.append(mask_bet[id_max])
            mask_after.append(mask_aft[id_max])
            pred.append(rel)
            wps_left.append(llpos[id_max])
            wps_right.append(rrpos[id_max])
        preds = np.zeros([len(bag_key), self.num_rels])
        preds[np.arange(len(bag_key)), pred] = 1
        loss, step, inpp, _, ip, conv, hh, pp = sess.run([self.loss, self.global_step, self.inputs, self.train_op, self._input, self.conv, self.h1, self.pooled], feed_dict={
            self.input: batch,
            self.mask1: mask_before,
            self.mask2: mask_between,
            self.mask3: mask_after,
            self.preds: preds,
            self.wps1: wps_left,
            self.wps2: wps_right,
            self.soft_label_flag: soft_label_flag
        })
        assert np.min(np.max(hh, axis=1)) > -50.0
        return loss

    def test(self, sess, bag_key, test_bag, sen_id, lpos, rpos, real_sen, namepos):
        # bag_key: mid1 mid2
        pair_score = []
        cnt_i = 1
        for key in bag_key:
            # sys.stdout.write('testing %d cases...\r' % cnt_i
            # sys.stdout.flush()
            cnt_i += 1
            sentences = test_bag[key]
            sen_vec, mask_bef, mask_bet, mask_aft, llpos, rrpos = self.sen_padding(sen_id, sentences, lpos, rpos, real_sen, namepos)
            scores = sess.run(self.scores, feed_dict={self.input: sen_vec, self.mask1: mask_bef, self.mask2: mask_bet, self.mask3: mask_aft, self.wps1: llpos, self.wps2: rrpos, self.soft_label_flag: [0]})
            score = np.max(scores, axis=0)
            for i, sc in enumerate(score):
                if i==0: continue
                pair_score.append({"mid": key, "rel": i, "score": sc})
        return pair_score





