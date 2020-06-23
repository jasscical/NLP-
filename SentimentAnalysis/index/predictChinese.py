# -*- coding: utf-8 -*-

import argparse
import logging
import time
import os
import jieba
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.python.saved_model import tag_constants
import yaml
from tensorflow.contrib import learn

path_yaml_config = 'E:/TensorFlowProgram/SentimentAnalysis/index/config.yml'


def load_yaml_config(yaml_file):
    with open(yaml_file) as f:
        config = yaml.load(f)
    return config


def texts_to_sequences(text, vocab_path, stopword_path):
    vocab = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    stopword = [line.strip() for line in open(stopword_path, 'r', encoding='utf-8').readlines()]

    # 输入是一个可迭代对象，输出是generator[array]
    text_seq = vocab.transform(
        (" ".join([x for x in jieba.cut(text) if x not in stopword]),))

    return text_seq


config = load_yaml_config(path_yaml_config)
vocab_path = config["data"]["vocab_path"]
stopword_path = config["data"]["stopword_path"]
ckpt_path = config["model"]["ckpt_path"]
pb_path = config["model"]["pb_path"]


def predict_ckpt(text):
    """从检查点导入模型"""
    with tf.Session() as sess:
        checkpoint_file = tf.compat.v1.train.import_meta_graph(ckpt_path)
        print(checkpoint_file)
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        graph = tf.get_default_graph()
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        probs = graph.get_tensor_by_name("softmaxLayer/probs:0")
        start_at = time.time()
        text_seq = texts_to_sequences(text, vocab_path, stopword_path)
        pred = sess.run(probs, feed_dict={input_x: list(text_seq), keep_prob: 1.0})
        label = "正向" if pred[0][0] > 0.65 else "负向" if pred[0][0] < 0.35 else "中性"
        print(label)
        return {"label":        label, "score": float(pred[0][0]),
                "elapsed_time": time.time() - start_at}


def predict_pb(text):
    """从冻结图导入模型"""
    with tf.Session() as sess:
        print("hello")
        tf.saved_model.loader.load(sess, [tag_constants.SERVING],
                                   os.path.join(os.getcwd(), 'index', 'tfserving', '1568521090', ))
        graph = tf.get_default_graph()
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        probs = graph.get_tensor_by_name("softmaxLayer/probs:0")

        start_at = time.time()
        print(stopword_path)
        text_seq = texts_to_sequences(text, os.path.join(os.getcwd(), 'index', 'text_data', 'vocab.pkl'),
                                      os.path.join(os.getcwd(), 'index', 'text_data', 'stopwords.txt'))

        pred = sess.run(probs, feed_dict={input_x: list(text_seq), keep_prob: 1.0})
        print("predict values: {}".format(pred[0]))
        print("{}".format("正向" if pred[0][0] > 0.7 else "负向" if pred[0][0] < 0.3 else "中性"))
        label = "正向" if pred[0][0] > 0.65 else "负向" if pred[0][0] < 0.35 else "中性"
        print(label)
        return {"label":        label, "score": float(pred[0][0]),
                "elapsed_time": time.time() - start_at}
