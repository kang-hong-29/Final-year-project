import struct
import os
import numpy as np
import matplotlib.pyplot as plt
def read_bin(filepath):
    binfile = open(filepath, 'rb') #打开二进制文件
    size = os.path.getsize(filepath) #获得文件大小
    train_y = [] 
    for i in range(size):
        data = binfile.read(1)
        num = struct.unpack('B', data)
        train_y.append(num[0])
    train_y = np.array(train_y)
    np.savetxt("datasets/fyp/simclr/test_y1.txt",train_y)
    print(train_y,np.shape(train_y))
# read_bin("datasets/fyp/simclr/test_batch.bin")
def draw_result(result1,result2,time1,lens1,save_dir):
    m1 = [y[0] for y in result1]
    r1 = [y[0] for y in m1]
    k1 = [x[1] for x in m1]
    m2 = [m[0] for m in result2]
    r2 = [n[0] for n in m2]
    t1 = [y[0] for y in time1]
    t = [y[0] for y in t1]
    l1 = [y[0] for y in lens1]
    l = [y[0] for y in l1]
    fig = plt.figure(1)
    plt.suptitle('Performance Comaprison between DenMune and DenMune with merge process')
    ax1=plt.subplot(2,1,1)
    plt.plot(k1,[r[0] for r in r1], label='acc')
    plt.plot(k1,[r[1] for r in r1], label='nmi')
    plt.plot(k1,[r[2] for r in r1], label='ari')
    plt.xlabel('K')
    plt.ylabel('Metrics') 
    plt.legend()
    ax2=plt.subplot(2,1,2)
    plt.plot(k1,[r[0] for r in r2], label='acc')
    plt.plot(k1,[r[1] for r in r2], label='nmi')
    plt.plot(k1,[r[2] for r in r2], label='ari')
    plt.xlabel('K')
    plt.ylabel('Metrics') 
    plt.legend()
    plt.savefig(save_dir+ '/' +'/performance.png')
    plt.show()
    fig = plt.figure(2)
    plt.title('DenMune vs DenMune with merge (running time)')
    plt.plot(k1,[r[0] for r in t],label='DenMune')
    plt.plot(k1,[r[1] for r in t],label='DenMune_merge')
    plt.legend()
    plt.xlabel('K')
    plt.ylabel('Running time(s)')
    plt.show()
    plt.savefig(save_dir+ '/' +'/time.png')
    fig = plt.figure(3)
    plt.title('DenMune vs DenMune with merge (cluster number)')
    plt.plot(k1,[r[0] for r in l],label='DenMune')
    plt.plot(k1,[r[1] for r in l],label='DenMune_merge')
    plt.legend()
    plt.xlabel('K')
    plt.ylabel('Number of clusters')
    plt.show()
    plt.savefig(save_dir+ '/' +'/lens.png')
    

    
