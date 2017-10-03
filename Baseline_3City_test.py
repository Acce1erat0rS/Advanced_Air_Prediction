import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM

def process(x):
    if x == '?':
        return 0.0
    else:
        return float(x)

dataset = []
split = 200
leap = 6
length = 16

# Read from the file of the training set
data = []

print("Processing target set")
f = open("Beijing", 'r')
l = len(f.readlines())
ls = []
f.close()
buff = []
target_set = []

for line in range(96,14688):
    f1 = open("Beijing", "r")
#    f2 = open("Tianjin","r")
#    f3 = open("Huludao","r")
    ls1 = f1.readlines()[line].split("#")
#    ls2 = f2.readlines()[line].split("#")
#    ls3 = f3.readlines()[line].split("#")
#    buff = []
#    buff.append(map(float,ls1[7:10]))
#    buff.append(map(float,ls2[7:10]))
#    buff.append(map(float,ls3[7:10]))
#    target_set.append(buff)
    f1.close()
    target_set.append(map(float,ls1[7:10]))
#    f2.close()
#    f3.close()

print("Processing trainingset")
for i in range(14592):
    f1 = open("Beijing", "r")
    f2 = open("Tianjin","r")
    f3 = open("Huludao","r")
    buff = []
    for i in range(71):
	buff.append([])
    count = 0
    for line in f1.readlines()[i:i+71]:
        ls = line.split('#')
        buff[count].append(map(float,ls[4:16]))
        count = count+1
    f1.close()
    count = 0
    for line in f2.readlines()[i:i+71]:
        ls = line.split('#')
        buff.append(map(float,ls[4:16]))
        count = count+1
    f2.close()
    count = 0
    for line in f3.readlines()[i:i+71]:
        ls = line.split('#')
        buff.append(map(float,ls[4:16]))
        count = count+1
    f3.close()
    data.append(buff)

set = np.array(data)
target = np.array(target_set)
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
model.add(LSTM(batch_input_shape=(384, 71, 36), output_dim=256, activation='sigmoid', inner_activation='hard_sigmoid'))
model.add(Activation("relu"))
#model.add(LSTM((384,256),output_dim=256, activation='sigmoid', inner_activation='hard_sigmoid'))
#model.add(Activation("relu"))
#model.add(LSTM((384,256),output_dim=128, activation='sigmoid', inner_activation='hard_sigmoid'))
#model.add(Activation("relu"))
model.add(Dense(3))

model.compile(loss='mae',
              optimizer='Adagrad',
              metrics=['accuracy'])
keras.optimizers.Adagrad(lr=0.01, epsilon=1e-08)

model.fit(set, target, batch_size=384, nb_epoch=1000)
# score = model.evaluate(testset, testtarget, batch_size=16)


