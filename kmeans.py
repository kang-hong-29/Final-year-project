from pyexpat import features
import numpy as np
import matplotlib.pyplot as plt
from time import time
from scipy.io import arff
import pandas as pd
from sklearn.cluster import KMeans
import metrics
import umap
from DenMune_merge import fit
from utils import draw_result 
# loadpath1 = "embeddings/stl-10/feas_moco_512_l2.npy"
# loadpath2 = "embeddings/stl-10/y_true.txt"
loadpath1 = "embeddings/mnist-test/embedding.txt"
loadpath2 = "embeddings/mnist-test/labels.txt"
save_dir = "result"
result1 = []
result2 = []
# features = np.load(loadpath1)
features = np.loadtxt(loadpath1)
Y = np.loadtxt(loadpath2,dtype=np.int)

T1=time()
features = umap.UMAP(random_state=0,n_neighbors=50,
                    min_dist=0,
                    n_components=10).fit_transform(features)
T2=time()
print("umap time",T2-T1)


cluster = KMeans(n_clusters=10,random_state=0).fit_predict(features)

acc = np.round(metrics.acc(Y, cluster), 5)
nmi = np.round(metrics.nmi(Y, cluster), 5)
ari = np.round(metrics.ari(Y, cluster), 5)

print(acc,nmi,ari)
time1 = []
lens1 = []
for K in range (350,450):
    y_pred1,metrics1, y_pred2, metrics2,t,l = fit(K,features,10,Y,0)
    result1.append(metrics1)
    result2.append(metrics2)
    time1.append(t)
    lens1.append(l)

draw_result(result1,result2,time1,lens1,save_dir)


save_dir = "result"
result1 = []
result2 = []