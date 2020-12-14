#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 14/12/2020 下午 8:21
# @Author: xiaoni
# @File  : res1.py


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
for year in range(1):
    for month in range(3):
        ac = new_dict[year][month]
        for i in range(len(ac)):
            tem.append(ac[i]['avetem'])
            day.append(b)
            b = b + 1
results = []
origin_Day = day[:]
day.extend(list(range(b, b + 30)))
datasets_X_1 = np.array(day).reshape(-1, 1)
datasets_X = datasets_X_1[:b]
datasets_X_test = datasets_X_1[b + 1:b + 30]
datasets_Y = np.array(tem).reshape(-1, 1)

model1 = polynomial_model(degree=2)
X_ploy = model1.fit(datasets_X, datasets_Y)
train_score = model1.score(datasets_X, datasets_Y)  # 训练集上拟合的怎么样
mse = mean_squared_error(datasets_Y, model1.predict(datasets_X))  # 均方误差 cost
results.append({"model": model1, "degree": 2, "score": train_score, "mse": mse})

model2 = polynomial_model(degree=3)
X_ploy = model2.fit(datasets_X, datasets_Y)
train_score = model2.score(datasets_X, datasets_Y)  # 训练集上拟合的怎么样
mse = mean_squared_error(datasets_Y, model2.predict(datasets_X))  # 均方误差 cost
results.append({"model": model2, "degree": 3, "score": train_score, "mse": mse})

model3 = polynomial_model(degree=4)
X_ploy = model3.fit(datasets_X, datasets_Y)
train_score = model3.score(datasets_X, datasets_Y)  # 训练集上拟合的怎么样
mse = mean_squared_error(datasets_Y, model3.predict(datasets_X))  # 均方误差 cost
results.append({"model": model3, "degree": 4, "score": train_score, "mse": mse})

model4 = polynomial_model(degree=5)
X_ploy = model4.fit(datasets_X, datasets_Y)
train_score = model4.score(datasets_X, datasets_Y)  # 训练集上拟合的怎么样
mse = mean_squared_error(datasets_Y, model4.predict(datasets_X))  # 均方误差 cost
results.append({"model": model4, "degree": 5, "score": train_score, "mse": mse})

for r in results:
    print("degree: {}; train score: {}; mean squared error: {}".format(r["degree"], r["score"], r["mse"]))

plt.axvline(x=90, linestyle='--', c="green")
plt.axvline(x=97, linestyle='--', c="lightgreen")

plt.plot(origin_Day, tem, color='cornflowerblue', linewidth=2, label="origin_data")
plt.plot(datasets_X, model1.predict(datasets_X), color='cyan', marker='*', linestyle=':', linewidth=1,
         label='Quadratic polynomial')
plt.plot(datasets_X_test, model1.predict(datasets_X_test), color='cyan', marker='*', linewidth=1, linestyle=':')

plt.plot(datasets_X, model2.predict(datasets_X), color='red', marker='*', linestyle=':', linewidth=1,
         label='Cubic polynomial')
plt.plot(datasets_X_test, model2.predict(datasets_X_test), color='red', marker='*', linewidth=1, linestyle=':')

plt.plot(datasets_X, model3.predict(datasets_X), color='blue', marker='*', linestyle=':', linewidth=1,
         label='Quartic polynomial')
plt.plot(datasets_X_test, model3.predict(datasets_X_test), color='blue', marker='*', linewidth=1, linestyle=':')

plt.plot(datasets_X, model4.predict(datasets_X), color='yellow', marker='*', linestyle=':', linewidth=1,
         label='Fifth degree polynomial')
plt.plot(datasets_X_test, model4.predict(datasets_X_test), color='yellow', marker='*', linewidth=1, linestyle=':')

plt.legend()
plt.show()
