# coding=utf-8

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import threading
import os


def LSTM(inputs, hidden_size, layer_num, batch_size, keep_prob):
    lstm_cell = rnn.LSTMCell(num_units=hidden_size,
                             forget_bias=1.0,
                             state_is_tuple=True)
    #                         time_major=False)

    lstm_cell = rnn.DropoutWrapper(cell=lstm_cell,
                                   input_keep_prob=1.0,
                                   output_keep_prob=keep_prob)

    mlstm_cell = rnn.MultiRNNCell([lstm_cell] * layer_num, state_is_tuple=True)

    init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)

    outputs, state = tf.nn.dynamic_rnn(mlstm_cell,
                                       inputs=inputs,
                                       initial_state=init_state)
    h_state = outputs[:, -1, :]

    return h_state


def Affine(input, input_dim, output_dim, trainable=True):
    m_W = tf.Variable(tf.truncated_normal([input_dim,
                                           output_dim],
                                          stddev=0.1),
                      dtype=tf.float32, trainable=trainable)
    m_bias = tf.Variable(tf.constant(0.1, shape=[output_dim]),
                         dtype=tf.float32, trainable=trainable)

    return tf.matmul(input, m_W) + m_bias
