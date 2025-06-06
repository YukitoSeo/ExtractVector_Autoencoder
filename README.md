# ExtractVector_Autoencoder
## プログラムの概要
このプログラムは画像からベクトルデータを抽出して、クラスタリングを適用することで、画像の特徴から分類を行うプログラムです。  
画像から抽出したベクトルデータを抽出して、分類を事前にしておくことで検索時の計算量を削減することを目的としています。（本プログラムでは、検索精度については考慮していないことに注意）  

以下に、本プログラムの特徴について記載します。  
・画像データを情報量の小さいベクトルデータに変換することが可能です。  
・settings.pyから機械学習モデルに関するパラメータを、容易に変更することが可能です。  
・クラスタリングをすることで、類似するベクトルデータを分類しておくことができます。  

## Quick Start
### STEP-1:環境設定
・以下のようなフォルダ構造を作ってください。
```
.
└── Folder
    ├── model (Folder)
    ├── dataset
    │   ├── class1
    │   │   ├── image1.png
    │   │   └── ・・・
    │   ├── class2
    │   │   ├── image102.png
    │   │   └── ・・・
    │   └── ・・・
    ├── ae_ml_model.py
    └── clastering.py
```
  
・以下のライブラリをPython環境にインストール、またはrequirements.txtに記載されているライブラリをインストールしてください。
```
pip install numpy==1.24.3
pip install matplotlib==3.7.2
pip install scikit-learn==1.3.0
pip install tensorflow==2.14.0
pip install keras==2.9.0
pip install scikit-learn==1.3.0
pip install opencv-python==4.8.0.76
```
### STEP-2:settings.pyの編集
settings.pyを編集し、各種パラメータとデータセットのパスを設定します。
```
class setting():
    def __init__(self):
        self.image_size = 28 # 読み込む画像のサイズ。「28」を指定した場合、縦28横28ピクセルの画像に変換します。
        self.z_dim = 500 # オートエンコーダのエンコーダ部分から出力される特徴ベクトルの次元数
        self.train_data_path = './dataset' # ここを変更。データセットのフォルダ名を入力
        self.learning_rate = 0.0004 # オートエンコーダの学習率
        self.batch_size = 64 # オートエンコーダのバッチ数
        self.epochs = 50 # オートエンコーダのエポック数
        self.learning_rate_ml = 0.0004 # 距離学習モデルの学習率
        self.num_batchs=64 # 距離学習モデルのバッチサイズ
        self.epocks_ml = 50 # 距離学習モデルのエポック数
```
### STEP-3:オートエンコーダと距離学習モデルの学習
cdコマンドを用いて、プログラムのあるフォルダのパスまで移動します。
以下のコマンドで、ae_ml_model.pyを起動することで、学習を開始します。
```
python ae_ml_model.py
```

### STEP-4:オートエンコーダと距離学習モデルによる推論
以下のコマンドで、ae_ml_model.pyを起動することで、推論を開始します。
```
python clastering.py
```
