import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier

from sklearn.externals import joblib

#从csv 读取数据
def pre_data():
    #41维表头
    column_names = []
    for i in range(0, 40):
        column_names.append("features_" + str(i+1))
    column_names.append("output")

    #read csv
    rd_csv =pd.read_csv("../data/data_csvs/data.csv", names=column_names)

    X_train, X_test, y_train, y_test = train_test_split(
        rd_csv[column_names[0:40]],
        rd_csv[column_names[40]],
        test_size = 0.25,
        random_state=33

    )

    return X_train, X_test, y_train, y_test


path_models = "../data/data_models"


# LR模型
def model_LR():
   X_train_LR, X_test_LR, y_train_LR, y_test_LR = pre_data()

   # 数据预加工
   # 标准化数据，保证每个维度的特征数据方差为1，均值为0。使得预测结果不会被某些维度过大的特征值而主导
   ss_LR = StandardScaler()
   X_train_LR = ss_LR.fit_transform(X_train_LR)
   X_test_LR = ss_LR.fit_transform(X_test_LR)

   #
   LR = LogisticRegression()
   LR.fit(X_train_LR, y_train_LR)
   # save LR model
   joblib.dump(LR, path_models+"model_LR.m")

   score_LR = LR.score(X_test_LR, y_test_LR)
   return (ss_LR)

# MLPC model
def model_MLPC():
    # get data
    X_train_MLPC, X_test_MLPC, y_train_MLPC, y_test_MLPC = pre_data()

    # 数据预加工
    ss_MLPC = StandardScaler()
    X_train_MLPC = ss_MLPC.fit_transform(X_train_MLPC)
    X_test_MLPC = ss_MLPC.transform(X_test_MLPC)

    # 初始化 MLPC
    MLPC = MLPClassifier(hidden_layer_sizes=(13, 13, 13), max_iter=500)
    # 调用 MLPC 中的 fit() 来训练模型参数
    MLPC.fit(X_train_MLPC, y_train_MLPC)

    # save MLPC model
    joblib.dump(MLPC, path_models + "model_MLPC.m")

    # 评分函数
    score_MLPC = MLPC.score(X_test_MLPC, y_test_MLPC)
    # print("The accurary of MLPC:", score_MLPC)

    return (ss_MLPC)

# Linear SVC model
def model_LSVC():
    # get data
    X_train_LSVC, X_test_LSVC, y_train_LSVC, y_test_LSVC = pre_data()

    # 数据预加工
    ss_LSVC = StandardScaler()
    X_train_LSVC = ss_LSVC.fit_transform(X_train_LSVC)
    X_test_LSVC = ss_LSVC.transform(X_test_LSVC)

    # 初始化 LSVC
    LSVC = LinearSVC()

    # 调用 LSVC 中的 fit() 来训练模型参数
    LSVC.fit(X_train_LSVC, y_train_LSVC)

    # save LSVC model
    joblib.dump(LSVC, path_models + "model_LSVC.m")

    # 评分函数
    score_LSVC = LSVC.score(X_test_LSVC, y_test_LSVC)
    # print("The accurary of LSVC:", score_LSVC)

    return ss_LSVC
# SGDC, Stochastic Gradient Decent Classifier, 随机梯度下降法求解(线性模型)
def model_SGDC():
    # get data
    X_train_SGDC, X_test_SGDC, y_train_SGDC, y_test_SGDC = pre_data()

    # 数据预加工
    ss_SGDC = StandardScaler()
    X_train_SGDC = ss_SGDC.fit_transform(X_train_SGDC)
    X_test_SGDC = ss_SGDC.transform(X_test_SGDC)

    # 初始化 SGDC
    SGDC = SGDClassifier(max_iter=5)
    # 调用 SGDC 中的 fit() 来训练模型参数
    SGDC.fit(X_train_SGDC, y_train_SGDC)

    # save SGDC model
    joblib.dump(SGDC, path_models + "model_SGDC.m")

    # 评分函数
    score_SGDC = SGDC.score(X_test_SGDC, y_test_SGDC)
    # print("The accurary of SGDC:", score_SGDC)

    return ss_SGDC
