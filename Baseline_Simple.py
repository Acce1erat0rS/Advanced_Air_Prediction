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

f = open("Beijing", 'r')
l = len(f.readlines())
ls = []
f.close()
buff = []
target_set = []
l = l-146

f = open("Beijing", "r")
for line in f.readlines()[96: l-4]:
    ls = line.split('#')
    print(ls[7:10])
    target_set.append(map(float,ls[7:10]))
f.close()

for i in range(l-72-24-4):
    f = open("Beijing", 'r')
    buff = []
    for line in f.readlines()[i:i+71]:
        ls = line.split('#')
        buff.append(map(float,ls[4:16]))
    f.close()
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
model.add(LSTM(batch_input_shape=(384, 71, 12), output_dim=256, activation='sigmoid', inner_activation='hard_sigmoid'))
model.add(Activation("relu"))
model.add(Dense(3))

model.compile(loss='mae',
              optimizer='Adagrad',
              metrics=['accuracy'])
keras.optimizers.Adagrad(lr=0.01, epsilon=1e-08)

model.fit(set, target, batch_size=384, nb_epoch=1000)
# score = model.evaluate(testset, testtarget, batch_size=16)


