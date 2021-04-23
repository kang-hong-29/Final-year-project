from denmune import fit
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from time import time
from sklearn.neighbors import KDTree 
from scipy.io import arff
import pandas as pd
from sklearn.datasets import load_breast_cancer
# from sklearn.mixture import GaussianMixture
import metrics
import umap
from utils import draw_result 
from mpl_toolkits.mplot3d import Axes3D

# data = arff.loadarff('datasets/artificial/chainlink.arff')
# df = pd.DataFrame(data[0])
# features = df.values[:, 0:len(df.values[0])-1]
# features= np.array(features).astype('float')
# Y = df.values[:, -1]
# Y[Y == b'noise'] = 6
# Y[Y == b'1'] = 0
# Y[Y == b'2'] = 1
# # Y[Y == b'2'] = 2
# # Y[Y == b'3'] = 3
# # Y[Y == b'4'] = 4
# # Y[Y == b'5'] = 5
# # Y[Y == b'6'] = 6
# # Y[Y == b'7'] = 7
# # Y[Y == b'8'] = 8
# Y= np.array(Y).astype('int')
# print(features,Y)

loadpath1 = 'embeddings/mnist-test/embedding.txt'
loadpath2 = "embeddings/mnist-test/labels.txt"
save_dir = "result"

# data = load_breast_cancer()
# features = data.data
# Y = data.target
# print(features,Y)

features = np.loadtxt(loadpath1)
y_true = np.loadtxt(loadpath2)
result1 = []
result2 = []
time1 = []
lens1 = []
T1 = time()
features = umap.UMAP(random_state=0,n_neighbors=50,
                    min_dist=0,
                    n_components=10).fit_transform(features)
T2=time()
print("umap time",T2-T1)
# fig = plt.figure()
# ax1 = plt.axes(projection='3d')
# ax1.scatter3D(features[:, 0], features[:, 1], features[:, 2], c=y_pred2)
# plt.show()
for K in range (350,450):
    y_pred1,metrics1, y_pred2, metrics2,t,l = fit(K,features,10,y_true,0)
    result1.append(metrics1)
    result2.append(metrics2)
    time1.append(t)
    lens1.append(l)

draw_result(result1,result2,time1,lens1,save_dir)

# the test for GMM
# features = np.loadtxt(loadpath1)
# X = umap.UMAP(n_neighbors=20,
#                       min_dist=0,
#                       n_components=10).fit_transform(features)
# y_true = np.loadtxt(loadpath2,dtype=np.int)
# X = features
# gmm = GaussianMixture(n_components=10).fit(X)
# y_pred = gmm.predict(X)
# print(y_pred)
# acc = np.round(metrics.acc_1(y_true, y_pred), 5)
# nmi = np.round(metrics.nmi(y_true, y_pred), 5)
# ari = np.round(metrics.ari(y_true, y_pred), 5)
# print(acc,nmi,ari)
# plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=1, cmap='viridis')
# plt.show()