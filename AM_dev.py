# coding=utf-8

# SPLIT stand for Strategic Poly Logic Initiative Technology
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import threading
import os
import gc

from Archetectures import LSTM
from Archetectures import Affine


seqLength = 71
hop = 24
timestep_size = 71                # Hours of looking ahead
output_parameters = 3   # Number of predicting parameters
num_stations = 3        # Number of monitoring stations

phase_1 = 20
phase_2 = 20


do_phase_1 = [1,2,3,4,5,7,9,11,13,15,17,19,21,24,27,30,33,36,40,45,50,55,60]
do_phase_2 = [5,10,14,17,20,23,26,29,32,35,37,39,41,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60]

training_epochs = 2000
_batch_size = 384

atm_dim = 6
aqi_dim = 7

data_dir = "../tf_learn/dev_data/"
# data_dir = "/home/spica/mnt_device/aqi/dev_data/timesub/"
log_dir = "/home/spica/mnt_device/aqi/result"


def process(x):
    if x == '?':
        return 0.0
    else:
        return float(x)


# Hyperparameters

dataset = []
split = 200
leap = 6
length = 16

lr = 0.0001
hidden_size = 256
layer_num = 3

# defines how many hours is used to predict

print("Processing target set")
start = 1395691200
end = 1448564400
cur_start = start
cur_end = start+hop*3600

# --------------------------------------------
#             Data Dictionary
# --------------------------------------------

# 0  RecordID
# 1  LocatID     	Location ID generated by unique latitude-longitude pairs for each monitor location
# 2  StationName	Station Name as defined by the monitor owner  (might change, might not be unique)
# 3  ChName  	    Chinese language station name (UTF8 encoding)
# 4  Latitude    	Latitude North in decimal degrees WGS84
# 5  Longitude   	Longitude East in decimal degrees WGS84
# 6  PM2.5       	Particulate Matter 2.5 micron diameter   µg/m3 (micrograms per cubic metre)
# 7  PM10        	Particulate Matter 10 micron diameter  µg/m3 (micrograms per cubic metre)
# 8  O3          	Ozone   pphm (parts per hundred million)
# 9  NO2         	NOX  NitrogenDioxide   pphm (parts per hundred million)
# 10 SO2           	SOX  SodiumDioxide  pphm (parts per hundred million)
# 11 CO         	CarbonMonoxide  ppm (parts per million)
# 12 Temperature	degrees Celsius
# 13 DewPoint   	degrees Celsius
# 14 Pressure   	millibars
# 15 Humidity   	absolute humidity in grams/meter3
# 16 Wind	        km / hour
# 17 UMT_time   	data collection time Greenwich Meat Time

# --------------------------------------------
#             Data Preparation
# --------------------------------------------

# from UNIX time 1395691200
# to UNIX time 1448564400
# Read from the file of the training set
raw_data = []
atm_data = []
aqi_data = []
target_set = []
seq_target = []
wth_pre = []

# TODO:Change latitude and longitude to an uni encoding

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
print np.shape(raw_data)

print("Preparing training set")
i = 0
while i < len(raw_data)-seqLength-hop:
    aqi_buff = []
    wth_buff = []
    atm_buff = []
    for j in range(hop):
        wth_hor = []
        for line in raw_data[i+seqLength+hop]: 
            wth_hor.append(raw_data[j + seqLength + hop][0][0:2]+raw_data[j + seqLength + hop][0][8:])
        wth_buff.append(wth_hor)
    for j in range(seqLength):
        aqi_hour = []
        atm_hour = []
        seq_ans = []
        for line in raw_data[i+j]:
            aqi_hour = aqi_hour + line[:7]
            atm_hour = atm_hour + line[0:2]+line[8:]
        aqi_buff.append(aqi_hour)
        atm_buff.append(atm_hour)
        seq_ans.append(raw_data[j + seqLength + hop][0][2:5])
    aqi_data.append(aqi_buff)
    atm_data.append(atm_buff)
    seq_target.append(seq_ans)
    wth_pre.append(wth_buff)
    ans = raw_data[i+seqLength+hop]
    target_set.append(ans[0][2:5])
    i += seqLength/2

# s_target = random.shuffle(target_set)

del raw_data
gc.collect()

print("Converting AQI into nmpy format...")
np_aqi_data = np.asarray(aqi_data)

print("Converting ATM into nmpy format...")
np_atm_data = np.asarray(atm_data)

print("Converting Target into nmpy format...")
np_target = np.asarray(target_set)

print("Converting Seq Target into nmpy format...")
np_seq_target = np.asarray(seq_target)

print("Seq Target shape :  " + str(np.shape(np_seq_target)))
print("Target shape     :  " + str(np.shape(np_target)))
print("AQI Data shape   :  " + str(np.shape(np_aqi_data)))
print("ATM Data shape   :  " + str(np.shape(np_atm_data)))
print("WTH Data shape   :  " + str(np.shape(wth_pre)))


# training_data = np.hstack((np_data,np_target))
# np.random.shuffle(training_data)
# X = training_data[:, :-1]
# y  = training_data[:, -1]


sess = tf.InteractiveSession()
batch_size = tf.placeholder(tf.int32)
_X = tf.placeholder(tf.float32, [None, timestep_size, 36])     # TODO change this to the divided ver
y = tf.placeholder(tf.float32, [None,3])
atm_x = tf.placeholder(tf.float32, [None, timestep_size, atm_dim*num_stations])
aqi_x = tf.placeholder(tf.float32, [None, timestep_size, aqi_dim*num_stations])
weather_pre = tf.placeholder(tf.float32, [None, hop,num_stations, atm_dim])
train = tf.placeholder(tf.float32)
# reshape

# atm_x = tf.placeholder(tf.float32, [None, num_stations*atm_dim])
# aqi_x = tf.placeholder(tf.float32, [None, num_stations*aqi_dim])

flatten_wth = tf.reshape(weather_pre,[-1,hop*num_stations*atm_dim])
atm_out = LSTM(inputs=atm_x,
               hidden_size=128,
               layer_num=3,
               batch_size=16,
               keep_prob=train,
               scope='ATM')

aqi_out = LSTM(inputs=aqi_x,
               hidden_size=128,
               layer_num=3,
               batch_size=16,
               keep_prob=train,
               scope='AQI')

con = tf.concat([atm_out, aqi_out], 1)

to_weather = Affine(con, 256, atm_dim*num_stations*hop)

keep_prob = tf.placeholder(tf.float32)


cross_entropy = -tf.reduce_mean(flatten_wth * tf.log(to_weather))
train_weather = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

wth_loss = tf.reduce_mean(tf.abs(flatten_wth-to_weather), 0)


W = tf.Variable(tf.truncated_normal([256, output_parameters],
                                    stddev=0.1),
                dtype=tf.float32)
bias = tf.Variable(tf.constant(0.1, shape=[output_parameters]),
                   dtype=tf.float32)
y_pre = tf.matmul(con, W) + bias

cross_entropy = -tf.reduce_mean(y * tf.log(y_pre))
train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
loss = tf.reduce_mean(tf.abs(y_pre-y), 0)

sess.run(tf.global_variables_initializer())

print("Start training")
for j in range(200):
    print ("epoch"+ (str)(j))
    count = 0
    phase_1_acc = 0
    phase_1_count = 0
    phase_2_acc = 0
    phase_2_count = 0
    for i in range(phase_1):
    #for batch in range(100, 400):
        #print("Iter "+str(i) + "/" + str(phase_1))
        batch = random.randint(100,400)
        sess.run(train_weather,
                 feed_dict={atm_x: atm_data[batch:batch+16],
                            aqi_x: aqi_data[batch:batch+16],
                            weather_pre: wth_pre[batch:batch+16],
                            train: 0.5})
    #    print("========Iter:"+str(i)+",Accuracy:========",(acc))
        if(i%21 != 0):
            for i in range(6):
                phase_1_count += 1
                phase_1_acc += sess.run(wth_loss, feed_dict={atm_x: atm_data[i*16:(i+1)*16],
                                                aqi_x: aqi_data[i*16:(i+1)*16],
                                                weather_pre: wth_pre[i*16:(i+1)*16],
                                                train: 1})
    phase_1_acc /= phase_1_count
    phase_1_res = float(sum(phase_1_acc))/len(phase_1_acc)
    print("     Phase1: Epoch" + str(j)+" :" + str(phase_1_res))
    for i in range(phase_2):
        batch = random.randint(100, 400)
        sess.run(train_op,
                 feed_dict={atm_x: atm_data[batch:batch+16],
                            aqi_x: aqi_data[batch:batch+16],
                            y: target_set[batch:batch+16],
                            train: 0.5})
        #    print("========Iter:"+str(i)+",Accuracy:========",(acc))
        for i in range(6):
            phase_2_count += 1
            phase_2_acc += sess.run(loss, feed_dict={atm_x: atm_data[i * 16:(i + 1) * 16],
                                                         aqi_x: aqi_data[i * 16:(i + 1) * 16],
                                                         y: target_set[i * 16:(i + 1) * 16],
                                                         train: 1})
    phase_2_acc /= phase_2_count
    print("     Phase 2: Epoch" + str(j)+ " :" + str(phase_2_acc))
    # acc = sess.run(loss, feed_dict={atm_x: atm_data[99],
    #                                aqi_x: aqi_data[99],
    #                                y: target_set[99],
    #                                train: 1})
    # print("Overall : " + str(count_all) + str(acc))
