import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
import random

data_dir = "/home/spica/mnt_device/aqi/dev_data/timesub/"
def process(x):
    if x == '?':
        return 0.0
    else:
        return float(x)

dataset = []
split = 200
leap = 6
length = 16


# from UNIX time 1395691200
# to UNIX time 1448564400
# Read from the file of the training set
data = []
target_set = []

# defines how many hours is used to predict
hop = 71

print("Processing target set")
start = 1395691200;
end = 1448564400;
cur_start = start;
cur_end = start+hop*3600;

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
    cur_start = cur_start+3600;
    cur_end = cur_end+3600;
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
# np.append(target, 32)
# ----------------------remain to be changed---------------------

# target.size = (set.size()[0], 1)
print "the dataset:\n"
print set
print "the target\n"
print target


# DENSE VER
'''
# build the model
model = Sequential()
model.add(Dense(output_dim=128, input_dim=322))
model.add(Activation("relu"))
model.add(Dense(output_dim=64))
model.add(Activation("relu"))
model.add(Dense(output_dim=1))

# compile the model
keras.optimizers.SGD(lr=0.0001, momentum=0.1, decay=0.05, nesterov=False)
model.compile(loss='mae', optimizer='sgd', metrics=['accuracy'])

# train the model
model.fit(set, target, nb_epoch=100, batch_size=32)
'''


# LSTM VER
model = Sequential()
# model.add(Embedding(1000, 256, input_length=322))
model.add(LSTM(batch_input_shape=(384,71, 36),
	return_sequences=True,
	#stateful=True,
	output_dim=256, 
	activation='sigmoid',
	inner_activation='hard_sigmoid'))
model.add(Activation("relu"))
model.add(LSTM( output_dim = 256, activation='sigmoid', inner_activation='hard_sigmoid', name = "LSTM-2"))
model.add(Activation("relu"))
# model.add(LSTM((384,256),output_dim=128, activation='sigmoid', inner_activation='hard_sigmoid'))
# model.add(Activation("relu"))
model.add(Dense(3))

model.compile(loss='mae',
              optimizer='Adagrad',
              metrics=['accuracy'])
keras.optimizers.Adagrad(lr=0.01, epsilon=1e-08)

model.fit(set, target, batch_size=384, nb_epoch=1000, validation_data=(val_set, val_target))
# score = model.evaluate(testset, testtarget, batch_size=16)


