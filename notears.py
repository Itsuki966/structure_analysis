import pandas as pd
import causalnex as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from causalnex.structure.notears import from_pandas
from causalnex.plots import plot_structure

# データの読み込
print("data loading...")
data_0005 = pd.read_excel("/Users/itsukikuwahara/Desktop/codes/research/data/data1.xlsx", sheet_name="00_05")
data_0510 = pd.read_excel("/Users/itsukikuwahara/Desktop/codes/research/data/data1.xlsx", sheet_name="05_10")
data_1015 = pd.read_excel("/Users/itsukikuwahara/Desktop/codes/research/data/data1.xlsx", sheet_name="10_15")

# データの結合
all_data = pd.concat([data_0005, data_0510, data_1015])
all_data.reset_index(drop=True, inplace=True)

# 説明変数と目的変数を定義
target_data = all_data["若年層人口"]
input_data = all_data.drop(["Unnamed: 0","year", "area", "code", "総人口"], axis=1)
# print(input_data.head())

# データの正規化
scaler = StandardScaler()
input_tr = scaler.fit_transform(input_data)

# データフレーム化
input_tr = pd.DataFrame(input_tr, columns=input_data.columns)
print(input_tr.head())

# データの構造学習
print("structure learning...")
sm = from_pandas(input_tr)
sm.remove_edges_below_threshold(0.5) # しきい値以下のエッジを削除

# エッジの可視化
print("plotting...")
dag = plot_structure(sm)
dag.show("structure.html")
