from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from time import time
from sklearn.neighbors import KDTree
import metrics

def construct_tree(X,K,leaf_size):
    # kdt = KDTree(X, leaf_size=1, metric='euclidean')
    # indices = kdt.query(X, K, return_distance=False)
    # print(array)
    nbrs = NearestNeighbors(n_neighbors=K, algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(X)
    # print (np.shape(indices),indices)
    return indices
    
def find_mutual_neighbor(k_neighbors):
    MNN_initial = []
    MNN = []
    for i in range(np.size(k_neighbors,0)):
        for j in k_neighbors[i,:]:
            if i in k_neighbors[j,:] and i !=j:
                MNN_initial.append([i,j])
    MNN_initial=np.array(MNN_initial)
    MNN.append(MNN_initial[0,...])
    # print(MNN_initial)
    m=0
    n=0

    while (n<np.shape(MNN_initial)[0]):
        if MNN_initial[[n],[0]]==MNN[m][0]:
            if MNN_initial[[n],[1]]==MNN[m][1]:
                n+=1
            else:
                MNN[m]=np.append(MNN[m],MNN_initial[[n],[1]])
                n+=1
        else:
            MNN.append(MNN_initial[n,...])
            m+=1
    MNN = [list(i) for i in MNN]        
    # MNN=np.array(MNN)    
    # print("--------------------MNN------------------------",MNN)
    return MNN

def ratio_and_classfication (shape,MNN,K):
    r = np.zeros((shape[0],1),dtype=float)
    seeds = []
    weak_points = []
    noise_1 = []
    i=0
    j=0
    while (i<shape[0]):
        if i == MNN[j][0]:
            r[[i],[0]] = len(MNN[j])/K
            if r[[i],[0]] >= 1:
                seeds.append([i, r[[i],[0]]])
            else:
                weak_points.append([i, r[[i],[0]]])
            i+=1
            if j == np.shape(MNN)[0]-1:
                j+=0
            else:
                j+=1          
        else:
            r[[i],[0]] = 0
            noise_1.append([i, 0])
            i+=1
    # seeds = [list(i) for i in seeds]
    # noise = [list(i) for i in noise]
    weak_points = np.array(weak_points)
    weak_points = weak_points[np.argsort(weak_points[:,1])[::-1]]
    seeds = np.array(seeds)
    seeds = seeds[np.argsort(seeds[:,1])[::-1]]
    noise_1 = np.array(noise_1)
    non_seeds = weak_points
    # print(r)
    print(noise_1)
    # return seeds,non_seeds

def get_seeds(MNN,K):
    seeds=[]
    for i in range(np.shape(MNN)[0]-1):
        if len(MNN[i])>=K:
            seeds.append(MNN[i])
    # print("-----------------------seeds-------------------------",seeds)
    return seeds
def get_weakpoint(MNN,K):
    weak_points=[]
    for i in range(np.shape(MNN)[0]-1):
        if len(MNN[i])>0 and len(MNN[i])<K:
            weak_points.append(MNN[i])
    # print('------------------------wp---------------------------',weak_points)
    return weak_points
def create_cluster(MNN,K):
    L=[]
    seeds = get_seeds(MNN,K)
    L.append(seeds[0])
    for i in range(np.shape(seeds)[0]):
        C_inter = []
        C = seeds[i]
        # print(L)
        # print(len(L))
        for j in range(len(L)):
            # print(j)
            # print(np.intersect1d(C,L[j]))
            if len(np.intersect1d(C,L[j])):
                C_inter = np.union1d(C,L[j]).astype(int)
                # print(C_inter)
                L=np.delete(L,j,axis=0).tolist()
                break
        if len(C_inter):
            C = np.union1d(C,C_inter).astype(int)
            # L.tolist().append(C)
            # L=np.array(L)
            L.append(C)
        else:
            L.append(C)
        #     L=np.array(L)
        #     L.tolist().append(C)
    # print('-------------------create_cluster-----------------------',L)
    return L
def Assign_WP (MNN,K):
    Q = get_weakpoint(MNN,K)
    L = create_cluster(MNN,K)
    noise_2 = []
    for i in range(len(Q)):
        count = []
        for j in range(len(L)):
            num = len(np.intersect1d(Q[i],L[j]))
            if num != 0:
                count.append(num)
            else:
                count.append(-1)    
        x = count.index(max(count))
        c = count[x]
        if c !=-1:
            if [Q[i][0]] not in L[x]:
                L[x]=np.concatenate((L[x],[Q[i][0]]))
            else:
                pass
            
        else:
            noise_2.append(Q[i][0])
    # print('---------------------------ass_weak------------------------',L)
    return L
def connectivity(L,K,k):
    S = []
    for n in range(len(L)):
        s = []
        for m in range(len(L)):
            if m != n:
                c_i =  np.intersect1d(L[n],L[m]).astype(int)
                if len(c_i) >= k:
                    s.append([m,len(c_i),n])
            else: pass
        # print(s)
        # print("s1",s[1])
        if len(s) == 0:
            pass
        else:
            s = np.array(s)
            # print('s',s)
            # print(np.argsort(s[:,1])[::-1])
            s = s[np.argsort(s[:,1])[::-1]].tolist() 
            s_m = s[0]
            S.append(s_m)
    # print("S",S)
    # print('---------connect----------',L)
    c_new = []
    if len(S) == 0:
        pass
    else:
        for q in range(len(S)):
            for p in range(len(S)):
                if S[q][0] == p and S[q][1] != 0:
                    if S[p][0] == q and ([p,q] not in c_new):
                        c_new.append([q,p])
                    else:
                        pass
    # print(L,S,c_new)
    return c_new

def merge(L, K,k):
    c_new = connectivity(L, K,k)
    print('len L',len(L))
    for c in range(len(c_new)):
        L[c_new[c][1]] = np.union1d(L[c_new[c][0]],L[c_new[c][1]]).astype(int).tolist()
    for d in range(len(c_new)):
        L = np.delete(L,c_new[d][0]-d,axis=0).tolist()
    # print('merge',L)
    return L


def continue_merge(L, K,k):
    L_merge = L
    while(1):
        c_new = connectivity(L_merge, K,k)
        if len(c_new) ==0:
            break
        else:
            L_merge = merge(L_merge, K,k)
    # print('continue_merge',L_merge)
    return L_merge
def evaluate(K,L,Y):
    shape = np.shape(Y)
    y_pred = np.zeros((np.shape(Y)[0],1),dtype=np.int)

    for i in range(len(L)):
        for j in range(np.shape(L[i])[0]):
            d = L[i][j]
            y_pred[d] = i
    # ratio_and_classfication(shape,MNN,K)
    y_pred = np.array(y_pred).T.reshape(shape[0],)
    y_true = Y
    acc = np.round(metrics.acc(y_true, y_pred), 5)
    nmi = np.round(metrics.nmi(y_true, y_pred), 5)
    ari = np.round(metrics.ari(y_true, y_pred), 5)
    print(K, acc,nmi,ari)
    return y_pred,acc,nmi,ari

def fit (K,X,leaf_size,Y,k):
    result1 = []
    result2 = []
    times = []
    lens = []
    t1=time()
    knn = construct_tree(X,K,leaf_size)
    MNN = find_mutual_neighbor(knn)
    L = Assign_WP(MNN,K)
    print(len(L))
    print("result without merge")
    t2=time()
    y_pred1,acc1,nmi1,ari1 = evaluate(K,L,Y)
    t3=time()
    print("time used:",t3-t1)
    result1.append([[acc1, nmi1, ari1],K])
    #begin merge process
    t4=time()
    L_merge = continue_merge(L,K,k)
    print(len(L_merge))
    print('result with merge')
    y_pred2,acc2,nmi2,ari2 = evaluate(K,L_merge,Y)
    t5=time()
    print("time used:",t2-t1+t5-t4)
    result2.append([[acc2, nmi2, ari2],K])
    times.append([[t3-t1,t2-t1+t5-t4],K])
    lens.append([[len(L),len(L_merge)],K])
    return y_pred1,result1, y_pred2, result2, times,lens
def fit2 (K,knn,Y,k):
    result1 = []
    result2 = []
    t1=time()
    MNN = find_mutual_neighbor(knn)
    L = Assign_WP(MNN,K)
    print(len(L))
    print("result without merge")
    t2=time()
    y_pred1,acc1,nmi1,ari1 = evaluate(K,L,Y)
    t3=time()
    print("time used:",t3-t1)
    result1.append([[acc1, nmi1, ari1],K])
    #begin merge process
    t4=time()
    L_merge = continue_merge(L,K,k)
    print(len(L_merge))
    print('result with merge')
    y_pred2,acc2,nmi2,ari2 = evaluate(K,L_merge,Y)
    t5=time()
    print("time used:",t2-t1+t5-t4)
    result2.append([[acc2, nmi2, ari2],K])

    return y_pred1,result1, y_pred2, result2
# result = []
# X = np.random.rand(10000, 2)
# Y = t2 = np.random.randint(0, 10, (10000, 1))

# lris_df = datasets.load_iris()
# X = lris_df.data
# Y = lris_df.target


# t1=time()
# fit(55,X,10,Y)
# t2=time()
# plt.show() 
# print("time used:",t2-t1)

# lris_df = datasets.load_iris()
# X = lris_df.data
# Y = lris_df.target

# for l in range(1, 20):
#     result = []
#     for K in range (3,50):
#         result.append([fit(K,X,l,Y),K])
#     acc = [y[0] for y in result]
#     k = [x[1] for x in result]
#     plt.plot(k,acc, label={"leaf_size = " + str(l)})