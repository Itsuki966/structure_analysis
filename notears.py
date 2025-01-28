import math
import pandas as pd
import causalnex as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from causalnex.structure.notears import from_pandas
from causalnex.plots import plot_structure

def processing_df(df):
    df.drop(["Unnamed: 0","year", "area", "code", "総人口"], axis=1, inplace=True)
    df.replace("-", 0, inplace=True)
        
def calc_changing_rate(df_prev, df):
    df_cr = (df - df_prev + 0.01) / (df_prev + 0.01)
    return df_cr

def scaler(x):
    return (2 / (1 + math.exp(-2*x))) - 1
    
def structure_learning(df, object_year, threshold):
    print(("Scaling..."))
    # scaler = StandardScaler()
    # input_tr = scaler.fit_transform(df)
    
    # input_tr = pd.DataFrame(input_tr, columns=df.columns)
    input_tr = df.applymap(scaler)
    print(input_tr.info())

    print("structure learning...")
    sm = from_pandas(input_tr)
    sm.remove_edges_below_threshold(threshold)    # しきい値以下のエッジを削除

    path_name = str(object_year) + ".html"
    dag = plot_structure(sm)
    dag.show(path_name)

# データの読み込み
print("data loading...")
data_0005 = pd.read_excel("/Users/itsukikuwahara/Desktop/codes/research/data/data1.xlsx", sheet_name="00_05")
data_0510 = pd.read_excel("/Users/itsukikuwahara/Desktop/codes/research/data/data1.xlsx", sheet_name="05_10")
data_1015 = pd.read_excel("/Users/itsukikuwahara/Desktop/codes/research/data/data1.xlsx", sheet_name="10_15")

# 年度ごとのデータ
df00 = pd.read_excel("/Users/itsukikuwahara/Desktop/codes/research/data/data1.xlsx", sheet_name="2000")
df05 = pd.read_excel("/Users/itsukikuwahara/Desktop/codes/research/data/data1.xlsx", sheet_name="2005")
df10 = pd.read_excel("/Users/itsukikuwahara/Desktop/codes/research/data/data1.xlsx", sheet_name="2010")
df15 = pd.read_excel("/Users/itsukikuwahara/Desktop/codes/research/data/data1.xlsx", sheet_name="2015")

# 不要な列を削除
processing_df(df00)
processing_df(df05)
processing_df(df10)
processing_df(df15)

# 必要な増減率を計算
df00_05 = calc_changing_rate(df00, df05)
df05_10 = calc_changing_rate(df05, df10)
df10_15 = calc_changing_rate(df10, df15)

# データの結合
# all_data = pd.concat([data_0005, data_0510, data_1015])
# all_data.reset_index(drop=True, inplace=True)
all_df = pd.concat([df00_05, df05_10, df10_15])
all_df.dropna(inplace=True)

# 分析するデータを用意
# input_data = all_data[["若年層人口","一般診療所数/可住地面積","一般診療所数/10万人","自市区町村で従業・通学している人口","第三次産業就業者","児童福祉費","小学校教員数"]]
features = ["若年層人口","一般診療所数/可住地面積","一般診療所数/10万人","自市区町村で従業・通学している人口","第三次産業就業者","児童福祉費","小学校教員数"]
all_features = list(df00.columns)

# 説明変数：00→05、目的変数：00→05
df1 = df00_05[all_features]
# 説明変数：00→05、目的変数：05→10
temp = df1.drop(["若年層人口"], axis=1)
df2 = pd.concat([temp, df05_10[["若年層人口"]]], axis=1) 
# 説明変数：00→05、目的変数：05→10
df3 = pd.concat([temp, df10_15[["若年層人口"]]], axis=1)

structure_learning(all_df, "all", 0.5)
# structure_learning(df2, "05_10_", 0.5)
# structure_learning(df3, "10_15_", 0.5)
