#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 14/12/2020 下午 6:58
# @Author: xiaoni
# @File  : res2.py

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline, Pipeline


def polynomial_model(degree=1):
    polynomial_features = PolynomialFeatures(degree=degree,
                                             include_bias=False)
    linear_regression = LinearRegression(normalize=True)  # normalize=True对数据归一化处理
    pipeline = Pipeline([("polynomial_features", polynomial_features),  # 添加多项式特征
                         ("linear_regression", linear_regression)])
    return pipeline


f = open(r"data.json", encoding='UTF-8')
new_dict = json.load(f)
day = []
tem = []
b = 0
for year in range(8):
    for month in range(12):
        ac = new_dict[year][month]
        for i in range(len(ac)):
            tem.append(ac[i]['avetem'])
            day.append(b)
            b = b + 1
results = []
origin_Day = day[:]
day.extend(list(range(b, b + 7)))
datasets_X_1 = np.array(day).reshape(-1, 1)
datasets_X = datasets_X_1[:b]
datasets_X_test = datasets_X_1[b:b + 7]
datasets_Y = np.array(tem).reshape(-1, 1)

model = polynomial_model(degree=26)
X_ploy = model.fit(datasets_X, datasets_Y)
train_score = model.score(datasets_X, datasets_Y)  # 训练集上拟合的怎么样
mse = mean_squared_error(datasets_Y, model.predict(datasets_X))  # 均方误差 cost
results.append({"model": model, "degree": 29, "score": train_score, "mse": mse})
for r in results:
    print("degree: {}; train score: {}; mean squared error: {}".format(r["degree"], r["score"], r["mse"]))

plt.plot(origin_Day, tem, color='cornflowerblue', linewidth=1, label="ground truth")
plt.plot(datasets_X, model.predict(datasets_X), color='red', marker='o', linestyle=':', label='fitted curve')
plt.plot(datasets_X_test, model.predict(datasets_X_test), color='green', marker='o', linestyle=':', label='predict')
plt.legend()
plt.show()
