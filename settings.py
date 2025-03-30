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

