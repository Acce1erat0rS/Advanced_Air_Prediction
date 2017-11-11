#coding=utf-8

# SPLIT stand for Strategic Poly Logic Initiative Technology
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import threading
import os


seqLength = 71
hop = 24
timestep_size = 71                # Hours of looking ahead
output_parameters = 3   # Number of predicting parameters
num_stations = 3        # Number of monitoring stations

training_epochs = 2000
_batch_size = 384

data_dir = "../tf_learn/dev_data/"


def process(x):
    if x == '?':
        return 0.0
    else:
        return float(x)

dataset = []
split = 200
leap = 6
length = 16

lr = 0.0001
hidden_size = 256
layer_num = 3


# from UNIX time 1395691200
# to UNIX time 1448564400
# Read from the file of the training set
data = []
target_set = []
raw_data = []

print "Reading data from disk"
list = os.listdir(data_dir)
for file in list:
    path = os.path.join(data_dir, file)
    if os.path.isfile(path):
        f1 = open(data_dir + file, 'rb')
        step = []
        for line in f1.readlines():
            ls = line.split('#')
            step.append(map(float, ls[4:16]))
        raw_data.append(step)
#print raw_data
print np.shape(raw_data)

# defines how many hours is used to predict

print("Processing target set")
start = 1395691200
end = 1448564400
cur_start = start
cur_end = start + seqLength * 3600

for i in range(len(raw_data)-seqLength-hop):
    buff = []
    for j in range(seqLength):
        hour = []
        for line in raw_data[i+j]:
            hour = hour+line[:]
        buff.append(hour)
    data.append(buff)
    ans = raw_data[i+seqLength+hop]
    #print(ans[0][3:5])
    target_set.append(ans[0][2:5])

'''
while(cur_end<end-(120+288)*3600):
    buff = []
    for i in range(hop):
        hour = []
        f1 = open(data_dir+(str)(cur_start+i*3600),'rb')
        for line in f1.readlines():
            ls = line.split('#')
            hour = hour+(map(float,ls[4:16]))
        f1.close()
        buff.append(hour);
    data.append(buff)
    f1 = open(data_dir+(str)(cur_start+120*3600),'rb')
    for line in f1.readlines():
        ls = line.split("#")
        target_set.append(map(float,ls[7:10]))
        break
    cur_start = cur_start+3600
    cur_end = cur_end+3600
'''
# s_target = random.shuffle(target_set)
print(len(target_set))
np_data = np.asarray(data)
np_target = np.asarray(target_set)
print("Target shape :",np_target.shape)
print("Data shape : :",np_data.shape)
# training_data = np.hstack((np_data,np_target))
# np.random.shuffle(training_data)
# X = training_data[:, :-1]
#y  = training_data[:, -1]

X = np_data
y = np_target

set = np.array(X[1920:])
target = np.array(y[1920:])
val_set = np.array(X[:1920])
val_target = np.array(y[:1920])


sess = tf.InteractiveSession()
batch_size = tf.placeholder(tf.int32)
_X = tf.placeholder(tf.float32, [None, timestep_size, 36])     # TODO change this to the divided ver
y = tf.placeholder(tf.float32, [None, 3])
keep_prob = tf.placeholder(tf.float32)

# --------------------------------------------
#             Construct LSTM cells
# --------------------------------------------

lstm_cell = rnn.LSTMCell(num_units=hidden_size,
                              forget_bias=1.0,
                              state_is_tuple=True)
#                              time_major=False)

lstm_cell = rnn.DropoutWrapper(cell=lstm_cell,
                               input_keep_prob=1.0,
                               output_keep_prob=keep_prob)

mlstm_cell = rnn.MultiRNNCell([lstm_cell] * layer_num, state_is_tuple=True)

init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)

# ** 当 time_major==False 时， outputs.shape = [batch_size, timestep_size, hidden_size]
# ** 所以，可以取 h_state = outputs[:, -1, :] 作为最后输出
# ** state.shape = [layer_num, 2, batch_size, hidden_size],
# ** 或者，可以取 h_state = state[-1][1] 作为最后输出
# ** 最后输出维度是 [batch_size, hidden_size]
outputs, state = tf.nn.dynamic_rnn(mlstm_cell,
                                   inputs=_X,
                                   initial_state=init_state)
h_state = outputs[:, -1, :]  # 或者 h_state = state[-1][1]

# *************** 为了更好的理解 LSTM 工作原理，我们把上面 步骤6 中的函数自己来实现 ***************
# 通过查看文档你会发现， RNNCell 都提供了一个 __call__()函数（见最后附），我们可以用它来展开实现LSTM按时间步迭代。
# **步骤6：方法二，按时间步展开计算
# outputs = list()
# state = init_state
# with tf.variable_scope('RNN'):
#     for timestep in range(timestep_size):
#         if timestep > 0:
#             tf.get_variable_scope().reuse_variables()
#         # 这里的state保存了每一层 LSTM 的状态
#         (cell_output, state) = mlstm_cell(X[:, timestep, :], state)
#         outputs.append(cell_output)
# h_state = outputs[-1]


# --------------------------------------------
#    Convert LSTM output to tensor of three
# --------------------------------------------
W = tf.Variable(tf.truncated_normal([hidden_size, output_parameters],
                                    stddev=0.1),
                dtype=tf.float32)
bias = tf.Variable(tf.constant(0.1,shape=[output_parameters]),
                   dtype=tf.float32)
y_pre = tf.matmul(h_state, W) + bias

cross_entropy = -tf.reduce_mean(y * tf.log(y_pre))
loss = tf.reduce_mean(tf.abs(y_pre-y), 0)

train_op = tf.train.AdamOptimizer(lr).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


sess.run(tf.global_variables_initializer())
count = 0
for i in range(training_epochs):
    for batch in range(5, 36):
        start = batch*_batch_size
        end = (batch+1)*_batch_size
        sess.run(train_op,
                 feed_dict={_X:data[start:end],
                            y: target_set[start:end],
                            keep_prob: 0.5,
                            batch_size: 384})
    #    print("========Iter:"+str(i)+",Accuracy:========",(acc))

    acc = sess.run(loss, feed_dict={_X:data[1152:1536],y:target_set[1152:1536],batch_size:384,keep_prob:1})
    print("Epoch:"+str(i)+str(acc))
