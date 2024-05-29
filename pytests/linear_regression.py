import numpy as np
import rust_dl

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

from sklearn.preprocessing import StandardScaler
s = StandardScaler()
data = s.fit_transform(data)

lr = rust_dl.LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=80718)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
x_train = np.array(x_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32).reshape(-1, 1)
x_test = np.array(x_test, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32).reshape(-1, 1)

print(x_train.shape, y_train.shape)
lr.fit(x_train, y_train, 1000, 23, 0.001, True, 180708)
preds = lr.predict(X=x_test)
print(preds.shape)
print("RMSE", root_mean_squared_error(preds, y_test))
print("r^2", r2_score(preds, y_test))

#print(list(zip(preds.squeeze().tolist(), y_test.squeeze().tolist())))

#plt.plot(X)
