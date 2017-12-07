import cv2
import numpy as np
import os
from tqdm import tqdm
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import Callback
from keras.models import Model
from keras.optimizers import RMSprop, Adam
import random
import pandas as pd
from pandas import Series, DataFrame
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation, Convolution2D
from keras.layers.core import Lambda
from keras.applications.inception_v3 import preprocess_input
from keras.layers import GlobalAveragePooling2D 
from keras.applications.xception import Xception
from keras.callbacks import EarlyStopping

prefix = '../../course/input/dogsvscats/'
train_path = '../../course/input/train/'
test_path = '../../course/input/test/'
n_class = 1
width = 299
lr = 1e-4
loss_function = 'binary_crossentropy'
last_activate = 'sigmoid'
input_shape = (width, width, 3)
np.random.seed(12)

images_path = os.listdir(train_path)
n = len(images_path)
#images_path = images_path[:500] + images_path[n - 500:n]
n = len(images_path)
X = np.zeros((n, width, width, 3), dtype=np.uint8)
y = np.zeros((n, n_class), dtype=np.uint8)
shapes = np.zeros((n, 2), dtype=np.uint16)
for i in range(n):
    label_name = images_path[i].split('.')[0]
    images_path[i] = train_path + images_path[i]
    if label_name == 'dog':
        y[i] = 1#(0, 1)
    else:
        y[i] = 0#(1, 0)
for i in range(100):
    images_path, y = shuffle(images_path, y)
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
    def on_epoch_end(self, batch, logs={}):
        print(logs)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        
def plotPredictions(X, y_pred):
    n = len(y_pred)
    n = 12 if n > 12 else n
    for i in range(n):
        random_index = random.randint(0, len(X) - 1)
        if y_pred[random_index, 0] >= 0.5: 
            print('I am {:.2%} sure this is a Dog'.format(y_pred[random_index][0]))
        else: 
            print('I am {:.2%} sure this is a Cat'.format(1-y_pred[random_index][0]))

        plt.imshow(X[random_index][:,:,::-1])
        plt.show()
def plotLossAndAccuracy(history):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')

    plt.subplot(1, 2, 2)
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.legend(['acc', 'val_acc'])
    plt.ylabel('acc')
    plt.xlabel('epoch')
def predictOnTestSet(model):
    df2 = pd.read_csv(prefix + 'sample_submission.csv')
    n_test = len(df2)
    X_test = np.zeros((n_test, width, width, 3), dtype=np.uint8)
    for i in tqdm(range(n_test)):
        img_path = test_path + '%s.jpg' % df2['id'][i]
        X_test[i] = cv2.resize(cv2.imread(img_path), (width, width))
    y_pred = model.predict(X_test, batch_size=32)
    return X_test, y_pred
def getDistribution(data):
    distribute = []
    min_data = np.min(data)
    max_data = np.max(data)
    bar_sum = 5
    bar_width = int(max_data / bar_sum)
    count_sum = 0
    names = []
    for i in range(bar_sum + 1):
        ceil = bar_width * (i + 1)
        count = np.sum(data < ceil) - count_sum
        count_sum = count_sum + count
        distribute.append(count)
        names.append('< ' + str(ceil))
    print(distribute, names, min_data, max_data)
    return distribute, names, min_data, max_data
def ColorShifting(image_path):
    img = cv2.imread(image_path)
    factor = random.randint(10, 50)
    f_add = np.vectorize(lambda x: 255 if (x + factor) > 255 else (x + factor))
    f_minus = np.vectorize(lambda x: 0 if (x - factor) < 0 else (x - factor))
    img[:,:,[0, 2]] = f_add(img[:,:,[0, 2]])
    img[:,:,[1]] = f_minus(img[:,:,[1]])
    return img
def RandomCrop(image_path):
    img = cv2.imread(image_path)
    shape = img.shape
    w = shape[0]
    h = shape[1]
    x_range_max = h - width
    y_range_max = w - width
    if x_range_max <= 0 or y_range_max <= 0:
        return img
    x = random.randint(0, x_range_max - 1)
    y = random.randint(0, y_range_max - 1)
    crop_img = img[y:(y + width), x:(x + width)] # [y:h, x:w]
    return crop_img
