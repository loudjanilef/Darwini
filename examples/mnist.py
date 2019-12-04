import keras
import keras.backend as K
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.datasets import mnist

from darwini import constants
from darwini.breeder import Breeder

num_classes = 10
scores = []
summaries = []

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_val, y_val) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_train /= 255
x_val /= 255

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)

# split validation data to obtain test data
x_test = x_val[8000:]
y_test = y_val[8000:]
x_val = x_val[:8000]
y_val = y_val[:8000]

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_val.shape[0], 'validation samples')
print(x_test.shape[0], 'test samples')

breeder = Breeder(x_train, y_train, x_val, y_val)
K.clear_session()
for i in range(3):
    _, network, _ = breeder.generation()
    model = network.compile()
    early_stopper = EarlyStopping(patience=3)
    model.fit(x_train, y_train, batch_size=constants.BATCH_SIZE, epochs=constants.EPOCH_NBR, verbose=1,
              validation_data=(x_val, y_val), callbacks=[early_stopper])
    _, score = model.evaluate(x_test, y_test, verbose=1)
    scores.append(score)
    summary = pd.DataFrame([e[0] for e in breeder.population]).describe()
    print(summary)
    summaries.append(summary)
    print("Best model score of generation {} is {}".format(i, score))
    K.clear_session()
