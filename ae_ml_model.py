import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from collections import defaultdict
from tensorflow import keras
from keras import layers
import glob
from sklearn.model_selection import train_test_split
from keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt
from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Activation, LeakyReLU, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from settings import setting
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


image_size = IMAGE_SIZE
color_setting = 3  #ここを変更。データセット画像のカラー：「1」はモノクロ・グレースケール。「3」はカラー。
class_number = len(FOLDER)
print('今回のデータで分類するクラス数は「', str(class_number), '」です。')

def preprocess(array):
    """
    Normalizes the supplied array and reshapes it into the appropriate format.
    """
    #array = array.astype("float32") / 255.0
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

X_image = []  
Y_label = []

for index, name in enumerate(FOLDER):
  read_data = TRAIN_DATA_PATH + '/' + name
  files = sorted(glob.glob(read_data + '/*.png'),key=natural_keys) #ここを変更。png形式のファイルを利用する場合のサンプルです。
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

train_images, valid_images ,train_labels ,valid_labels = train_test_split(X_image,Y_label,test_size=0.20,shuffle = True)
x_train = train_images
y_train = train_labels
x_test = valid_images
y_test = valid_labels

def r_loss(y_true, y_pred):
  return K.mean(K.square(y_true - y_pred), axis=[1,2,3])

encoder_input = Input(shape=(IMAGE_SIZE,IMAGE_SIZE,3), name='encoder_input')
x = encoder_input
x = Conv2D(filters=3, kernel_size=3, strides=1, padding='same', name='encoder_conv_0')(x)
x = LeakyReLU()(x)
x = Conv2D(filters=16, kernel_size=3, strides=2, padding='same', name='encoder_conv_0_1')(x)
x = LeakyReLU()(x)
x = Conv2D(filters=32, kernel_size=3, strides=2, padding='same', name='encoder_conv_1')(x)
x = LeakyReLU()(x)
#encoder_output2=Dense(Z_DIM, name='encoder_output')(x)
shape_before_flattening = K.int_shape(x)[1:]
x = Flatten()(x)
encoder_output = Dense(Z_DIM, name='encoder_output')(x)
encoder = Model(encoder_input, encoder_output)

# デコーダ
decoder_input = Input(shape=(Z_DIM,), name='decoder_input')
x = Dense(np.prod(shape_before_flattening))(decoder_input)
x = Reshape(shape_before_flattening)(x)
x = Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', name='decoder_conv_t_2')(x)
x = LeakyReLU()(x)
x = Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding='same', name='decoder_conv_t_2_6')(x)
x = LeakyReLU()(x)
x = Conv2DTranspose(filters=3, kernel_size=3, strides=1, padding='same', name='decoder_conv_t_3')(x)
x = Activation('sigmoid')(x)
decoder_output = x
decoder = Model(decoder_input, decoder_output)

# エンコーダ/デコーダ連結
model_input = encoder_input
model_output = decoder(encoder_output)
model = Model(model_input, model_output)
model.summary()
# 学習用設定設定（最適化関数、損失関数）
optimizer = Adam(learning_rate=LEARNING_RATE)

def r_loss(y_true, y_pred):
  return K.mean(K.square(K.log(y_true + 1) - K.log(y_pred + 1)), axis=-1)

model.compile(optimizer=optimizer, loss=r_loss, metrics=['accuracy'])

# 学習実行
history=model.fit(
    x_train,
    x_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_test, x_test),
)

model.summary()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.show()

model.save("./model/ae_model_dim{}".format(Z_DIM))

Encoder_model = keras.models.load_model("./model/ae_model_dim{}".format(Z_DIM), custom_objects={"r_loss": r_loss })
Encoder_model.summary()

layer_name = 'encoder_output'
intermediate_layer_model = Model(inputs=Encoder_model.input,
                                 outputs=Encoder_model.get_layer(layer_name).output)

feature_vector = intermediate_layer_model.predict(x_train)
print(feature_vector.shape)
examples = feature_vector

class_idx_to_train_idxs = defaultdict(list)
for y_train_idx, y in enumerate(y_train):
    class_idx_to_train_idxs[y].append(y_train_idx)

class_idx_to_test_idxs = defaultdict(list)
for y_test_idx, y in enumerate(y_test):
    class_idx_to_test_idxs[y].append(y_test_idx)


num_classes = class_number

class AnchorPositivePairs(keras.utils.Sequence):
    def __init__(self, num_batchs):
        self.num_batchs = num_batchs

    def __len__(self):
        return self.num_batchs

    def __getitem__(self, _idx):
        #x = np.empty((2, num_classes, 64,64,50), dtype=np.float32)
        x = np.empty((2, num_classes, Z_DIM), dtype=np.float32)
        for class_idx in range(num_classes):
            examples_for_class = class_idx_to_train_idxs[class_idx]
            anchor_idx = random.choice(examples_for_class)
            positive_idx = random.choice(examples_for_class)
            while positive_idx == anchor_idx:
                positive_idx = random.choice(examples_for_class)
            x[0, class_idx] = feature_vector[anchor_idx]
            x[1, class_idx] = feature_vector[positive_idx]
        return x

class EmbeddingModel(keras.Model):
    #train_step(self, data) メソッドだけをオーバーライドします。
    def train_step(self, data):
        # Note: Workaround for open issue, to be removed.未解決の問題の回避策。削除される予定です。
        if isinstance(data, tuple):
            data = data[0]
        anchors, positives = data[0], data[1]

        with tf.GradientTape() as tape:
            # Run both anchors and positives through model.
            # モデルを通してアンカーとポジティブの両方を実行します。
            anchor_embeddings = self(anchors, training=True)
            positive_embeddings = self(positives, training=True)

            # Calculate cosine similarity between anchors and positives. As they have
            # been normalised this is just the pair wise dot products.
            # アンカーとポジティブの間のコサイン類似度を計算します。彼らがそうしているように
            # 正規化されているため、これは単なるペアごとの内積です。
            #　ランダムに選択されたものとその対応する同ラベルとの内積
            similarities = tf.einsum(
                "ae,pe->ap", anchor_embeddings, positive_embeddings
            )

            # Since we intend to use these as logits we scale them by a temperature.
            # This value would normally be chosen as a hyper parameter.
            # これらをロジットとして使用するつもりなので、温度によってスケールします。
            # この値は通常、ハイパーパラメータとして選択されます。
            temperature = 0.2
            similarities /= temperature

            # We use these similarities as logits for a softmax. The labels for
            # this call are just the sequence [0, 1, 2, ..., num_classes] since we
            # want the main diagonal values, which correspond to the anchor/positive
            # pairs, to be high. This loss will move embeddings for the
            # anchor/positive pairs together and move all other pairs apart.
            # これらの類似性をソフトマックスのロジットとして使用します。のラベル
            # この呼び出しはシーケンス [0, 1, 2, ..., num_classes] です。
            # アンカー/ポジティブに対応する主な対角値が必要です
            # ペア、高くなります。この損失により、エンベディングが移動します。
            # アンカー/ポジティブペアを一緒に固定し、他のすべてのペアを離します。
            sparse_labels = tf.range(num_classes)
            loss = self.compiled_loss(sparse_labels, similarities)

        # Calculate gradients and apply via optimizer.
        #勾配を計算し、オプティマイザーを介して適用します。
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update and return metrics (specifically the one for the loss value).
        #メトリクス (特に損失値のメトリクス) を更新して返します。
        self.compiled_metrics.update_state(sparse_labels, similarities)
        return {m.name: m.result() for m in self.metrics}



inputs2 = layers.Input(shape=(Z_DIM))
x2 = layers.Dense(units=100, activation='relu')(inputs2)
x2 = layers.Dense(units=200, activation='relu')(x2)
embeddings = layers.Dense(units=100, activation=None)(x2)
embeddings = tf.nn.l2_normalize(embeddings, axis=-1)

model2 = EmbeddingModel(inputs2, embeddings)


model2.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE_ML),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)

history = model2.fit(AnchorPositivePairs(num_batchs=64), epochs=EPOCHS_ML)
model2.summary()
model2.save("./model/ml_model_dim{}".format(Z_DIM))

