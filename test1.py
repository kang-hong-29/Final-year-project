from denmune import fit
import numpy as np
import matplotlib.pyplot as plt
from time import time
from scipy.io import arff
import pandas as pd
# import umap

loadpath1 = "datasets/embeddings/stl-10/feas_moco_512_l2.npy"
loadpath2 = "datasets/fyp/simclr/stl10/test_y.txt"
loadpath3 = 'datasets/fyp/simclr/stl10/train_y.txt'
features = np.load(loadpath1)
Y1 = np.loadtxt(loadpath2,dtype=np.int)
Y2 = np.loadtxt(loadpath3,dtype=np.int)
Y = np.append(Y2,Y1)
np.savetxt('datasets/embeddings/stl-10/y_true.txt',Y)
print(np.shape(Y),np.shape(features))
# umap.plot.points(features)
# for i in range (10):
#     for j in range(10):
#         plt.figure()
#         plt.cla()
#         plt.scatter(features[:, i], features[:, j], s=1, alpha=1,marker = 'o')
#         print(i,j)
#         plt.show()
save_dir = "result"
result1 = []
result2 = []
# features = umap.UMAP(random_state=0,n_neighbors=50,
#                       min_dist=0,
#                       n_components=512).fit_transform(features)
for K in range (235,450):
    # features = umap.UMAP(random_state=0,n_neighbors=20,
    #                   min_dist=0,
    #                   n_components=10).fit_transform(features)

    y_pred1,metrics1, y_pred2, metrics2,t,l = fit(K,features,10,Y,0)
    result1.append(metrics1)
    result2.append(metrics2)

m1 = [y[0] for y in result1]
r1 = [y[0] for y in m1]
k1 = [x[1] for x in m1]
m2 = [m[0] for m in result2]
r2 = [n[0] for n in m2]
k2 = [n[1] for n in m2]
fig = plt.figure(1)
ax1=plt.subplot(2,1,1)   
plt.plot(k1,r1)
ax2=plt.subplot(2,1,2)
plt.plot(k2,r2)
plt.savefig(save_dir+ '/' +'/2D.png')
plt.show()



plt.figure()
plt.cla()
plt.scatter(features[:, 0], features[:, 1], c=Y, s=1, alpha=1,marker = 'o')
plt.show()

