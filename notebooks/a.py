#!/usr/bin/env python
# coding: utf-8

# ## 1 モジュールの準備

# In[1]:


import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random as rn


# ## 2 データの準備

# ### 2.1 訓練データの取得、設定

# In[2]:


# データを格納するリスト
X = []
Z = []

# 画像のサイズ
IMG_SIZE = 150

# 各花のディレクトリのパス
FLOWER_DAISY_DIR = '../input/flowers/daisy'
FLOWER_DANDI_DIR = '../input/flowers/dandelion'
FLOWER_ROSE_DIR = '../input/flowers/rose'
FLOWER_SUNFLOWER_DIR = '../input/flowers/sunflower'
FLOWER_TULIP_DIR = '../input/flowers/tulip'

# フラグを初期化
added_flowers = set()


# 次のmake_tarin_data関数は、データセットから花の画像を読み込んでデータを準備するためのものです。
# 
# - `flower_type`: 花の種類を指定します。
# - `DIR`: 花の画像が格納されているディレクトリのパスです。
# 
# この関数は、各花のデータを一度だけデータリストに追加し、重複を防ぐために役立ちます。関数内で、`added_flowers`というセットを使用して、すでにデータが追加された花の種類を追跡します。同じ花のデータが複数回追加されるのを防ぐため、各花のデータを追加する前に、その花の種類が`added_flowers`セットに存在しないかを確認します。花の種類がまだ追加されていない場合にのみデータを追加します。
# 

# In[3]:


# データの格納とコメントを追加した関数
def make_train_data(flower_type, DIR):
    if flower_type not in added_flowers:
        added_flowers.add(flower_type)
        # 各花のデータを格納するためのループ
        for img in tqdm(os.listdir(DIR)):
            # ラベルを花の名前に設定
            label = str(flower_type)
            # 画像ファイルのパスを生成
            path = os.path.join(DIR, img)
            # 画像をカラーで読み込み
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            # 画像サイズをリサイズ
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            # データをリストに追加
            X.append(np.array(img))
            Z.append(str(label))


# In[6]:


# 各花に対してデータを作成
make_train_data('Daisy', FLOWER_DAISY_DIR)
make_train_data('Dandelion', FLOWER_DANDI_DIR)
make_train_data('Rose', FLOWER_ROSE_DIR)
make_train_data('Sunflower', FLOWER_SUNFLOWER_DIR)
make_train_data('Tulip', FLOWER_TULIP_DIR)

# データの長さを出力
print(len(X))


# ### 2.2 ランダムなデータの表示

# このコードは、データセットからランダムに選択された9つの花の画像を3x3のグリッドに表示します。各画像には花の名前がタイトルとして表示され、データセットの内容を視覚化します。ランダムなサンプルを表示することで、データの内容を素早く把握できます。
# 

# In[5]:


# 3x3のサブプロットを作成
fig, ax = plt.subplots(3, 3)
fig.set_size_inches(15, 15)

# 9つのランダムな画像を選択して表示
for i in range(3):
    for j in range(3):
        # ランダムにインデックスを選択
        random_index = rn.randint(0, len(Z) - 1)
        
        # 画像をRGB形式に変換
        img_rgb = cv2.cvtColor(X[random_index], cv2.COLOR_BGR2RGB)
        
        # サブプロットに画像を表示
        ax[i, j].imshow(img_rgb)
        
        # タイトルに花の種類を表示
        ax[i, j].set_title('Flower: ' + Z[random_index])

# グラフを整えて表示
plt.tight_layout()
plt.show()


# 
