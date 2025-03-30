import tensorflow as tf
from tensorflow import keras
import numpy as np
import glob
from keras import backend as K
from keras.models import Model
from keras.utils import load_img, img_to_array
from sklearn.model_selection import train_test_split
from settings import setting
import cv2
import re
import os

set = setting()

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

################################
IMAGE_SIZE =  set.image_size# 読み込む画像のサイズ。「28」を指定した場合、縦28横28ピクセルの画像に変換します。
Z_DIM = set.z_dim # オートエンコーダのエンコーダ部分から出力される特徴ベクトルの次元数
TRAIN_DATA_PATH = set.train_data_path # ここを変更。データセットのフォルダ名を入力
FOLDER = [f for f in sorted(os.listdir(TRAIN_DATA_PATH),key=natural_keys) if os.path.isdir(os.path.join(TRAIN_DATA_PATH, f))]
LEARNING_RATE = set.learning_rate # オートエンコーダの学習率
BATCH_SIZE = set.batch_size # オートエンコーダのバッチ数
EPOCHS = set.epochs # オートエンコーダのエポック数
LEARNING_RATE_ML = set.learning_rate_ml # 距離学習モデルの学習率
NUM_BATCHS = set.num_batchs # 距離学習モデルのバッチサイズ
EPOCHS_ML = set.epocks_ml # 距離学習モデルのエポック数
################################

def preprocess(array):
    """
    Normalizes the supplied array and reshapes it into the appropriate format.
    """

    array = array.astype("float32") / 255.0
    array = np.reshape(array, (len(array), IMAGE_SIZE, IMAGE_SIZE, 3))
    return array

def one_depreprocess(array):
    """
    Normalizes the supplied array and reshapes it into the appropriate format.
    """

    array = array.astype("float32") * 255.0
    array = np.reshape(array, (IMAGE_SIZE, IMAGE_SIZE, 3))
    return array

def depreprocess(array):
    """
    Normalizes the supplied array and reshapes it into the appropriate format.
    """

    #array = array.astype("float32") * 255.0
    array = np.reshape(array, (len(array), IMAGE_SIZE, IMAGE_SIZE, 3))
    return array

def r_loss(y_true, y_pred):
  return K.mean(K.square(y_true - y_pred), axis=[1,2,3])

image_size = IMAGE_SIZE
color_setting = 3  #ここを変更。データセット画像のカラー：「1」はモノクロ・グレースケール。「3」はカラー。
class_number = len(FOLDER)
print('今回のデータで分類するクラス数は「', str(class_number), '」です。')

#3 データセットの読み込みとデータ形式の設定・正規化・分割 

X_image = []  
Y_label = []

for index, name in enumerate(FOLDER):
  read_data = TRAIN_DATA_PATH + '/' + name
  files = sorted(glob.glob(read_data + '/*.png'),key=natural_keys)  #ここを変更。png形式のファイルを利用する場合のサンプルです。
  print('--- 読み込んだデータセットは', read_data, 'です。')
  num=0
  for i, file in enumerate(files):
    if color_setting == 1:
      img = load_img(file, color_mode = 'grayscale' ,target_size=(image_size, image_size))  
    elif color_setting == 3:
      img = load_img(file, color_mode = 'rgb' ,target_size=(image_size, image_size))
    array = img_to_array(img)
    X_image.append(array)
    num +=1
    Y_label.append(index)
  print('index: ',index,' num:',num)

X_image = np.array(X_image)
Y_label = np.array(Y_label)

X_image = X_image.astype('float32') / 255
print(len(X_image))

X_test = X_image
Y_test = Y_label

Encoder_model = keras.models.load_model("./model/ae_model_dim{}".format(Z_DIM), custom_objects={"r_loss": r_loss })
Encoder_model.summary()

Metric_model = keras.models.load_model("./model/ml_model_dim{}".format(Z_DIM))
Metric_model.summary()

layer_name = 'encoder_output'
intermediate_layer_model = Model(inputs=Encoder_model.input,
                                 outputs=Encoder_model.get_layer(layer_name).output)

_feature_vector = intermediate_layer_model.predict(X_test)
feature_vector = Metric_model.predict(_feature_vector)

print(feature_vector.shape)

OUTPUT_DIR = "./result"

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

n_clusters = class_number 
kmeans_model = KMeans(n_clusters)
predict_clus = kmeans_model.fit_predict(feature_vector)
print(predict_clus.shape)
print(predict_clus[0])

n=5

for i in range(20):
  ax = plt.subplot(4, n, i + 1)
  plt.imshow(X_test[[i]].reshape(image_size, image_size, 3))
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
  ax.set_title(predict_clus[i])

plt.show()

"""
for i in range(n_clusters):
    cluster_dir = OUTPUT_DIR + "/cluster{}".format(i)
    if os.path.exists(cluster_dir):
        shutil.rmtree(cluster_dir)
    os.makedirs(cluster_dir)
# 結果をクラスタごとにディレクトリに保存

i=0
count=[0,0,0,0]
acc0,acc1,acc2,acc3=[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]
for label in predict_clus:
   im=one_depreprocess(X_test[[i]])
   if label == 0:
      cv2.imwrite(OUTPUT_DIR + '/cluster{}/{}.png'.format(label, i),im)
      count[0]+=1
      real_label = Y_label[i]
      acc0[real_label]+=1
   elif label == 1:
      cv2.imwrite(OUTPUT_DIR + '/cluster{}/{}.png'.format(label, i),im)
      count[1]+=1
      real_label = Y_label[i]
      acc1[real_label]+=1
   elif label == 2:
      cv2.imwrite(OUTPUT_DIR + '/cluster{}/{}.png'.format(label, i),im)
      count[2]+=1
      real_label = Y_label[i]
      acc2[real_label]+=1
   elif label == 3:
      cv2.imwrite(OUTPUT_DIR + '/cluster{}/{}.png'.format(label, i),im)
      count[3]+=1
      real_label = Y_label[i]
      acc3[real_label]+=1
   i+=1

print(count)
print(acc0)
print('============')
count2=0
for i in acc0:
   if i == 0:
      print('none')
   else:
      print('label',count2,'=',acc0[count2]/count[0])
      print('label',count2,'=',acc0[count2]/100)
   count2+=1
print('============')
count2=0
for i in acc1:
   if i == 0:
      print('none')
   else:
      print('label',count2,'=',acc1[count2]/count[1])
      print('label',count2,'=',acc1[count2]/100)
   count2+=1
print('============')
count2=0
for i in acc2:
   if i == 0:
      print('none')
   else:
      print('label',count2,'=',acc2[count2]/count[2])
      print('label',count2,'=',acc2[count2]/100)
   count2+=1
print('============')
count2=0
for i in acc3:
   if i == 0:
      print('none')
   else:
      print('label',count2,'=',acc3[count2]/count[3])
      print('label',count2,'=',acc3[count2]/100)
   count2+=1
print('============')
"""