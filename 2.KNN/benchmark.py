# 对数据集中的点云，批量执行构建树和查找，包括kdtree和octree，并评测其运行时间

import random
import math
import numpy as np
import time
import os
import struct

import octree as octree
import kdtree as kdtree
from collections import defaultdict
from result_set import KNNResultSet, RadiusNNResultSet
from scipy import spatial


def read_velodyne_bin(path):
    '''
    :param path:
    :return: homography matrix of the point cloud, N*3
    '''
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)

def scipyKdtreeSearch(tree:spatial.KDTree,result_set:KNNResultSet,point: np.ndarray):
    scipy_nn_dis,scipy_nn_idx=tree.query(point,result_set.capacity)
    for idx, distindex in enumerate(result_set.dist_index_list):
        distindex.distance=scipy_nn_dis[idx]
        distindex.index=scipy_nn_idx[idx]
    return False

def RandomVoxelGridDownsample(pointcloud,size):
    RandomPoints = []
    # 计算每个维度上的极值
    x_min=min(pointcloud[:][0])
    x_max=max(pointcloud[:][0])
    y_min=min(pointcloud[:][1])
    y_max=max(pointcloud[:][1])
    z_min=min(pointcloud[:][2])
    z_max=max(pointcloud[:][2])
    Dx=int((x_max-x_min)/size)
    Dy=int((y_max-y_min)/size)
    Dz=int((z_max-z_min)/size)
    # 用字典存储h及其对应的点
    d=defaultdict(list)
    for p in pointcloud:
        hx=int((p[0]-x_min)/size)
        hy=int((p[1]-y_min)/size)
        hz=int((p[2]-z_min)/size)
        h=hx+hy*Dx+hz*Dx*Dy
        d[h].append(p)
    # 随机选择
    RandomPoints=[]
    for key,value in d.items():
        RandomPoint=random.choice(value)
        RandomPoints.append(RandomPoint)
    
    RandomPoints = np.asarray(RandomPoints, dtype=np.float32)
    return RandomPoints

def main():
    # configuration
    leaf_size = 32
    min_extent = 0.0001
    k = 8
    radius = 1

    
    filename = r'C:\Users\Junliang\Desktop\三维点云\第二章\lesson2code\000000.bin'
    db_np = read_velodyne_bin(filename)
    #对原始点云采样，在下面的作业中对采样后的点做KNN搜索
    filter_db_np = RandomVoxelGridDownsample(db_np, 1)
    print("对原始点云采样，在下面的作业中对采样后的点做KNN搜索")
    print("filter_db_np = ")
    print(filter_db_np)
    print(filter_db_np.shape)
    
    print("octree --------------")
    #时间统计
    construction_time_sum = 0
    knn_time_sum = 0
#    radius_time_sum = 0
#    brute_time_sum = 0

    #construction
    begin_t = time.time()
    root = octree.octree_construction(db_np, leaf_size, min_extent)
    construction_time_sum += time.time() - begin_t  # 统计构建时间

    #Octree KNNsearch
    begin_t = time.time()
    for i in  range(len(filter_db_np)):    #len(db_np)
        result_set = KNNResultSet(capacity=k)
        query = filter_db_np[i,:]           #对每一个点进行KNN搜索
        octree.octree_knn_search(root, db_np, result_set, query)
    knn_time_sum += time.time() - begin_t
    
#    begin_t = time.time()
#    for i in  range(len(db_np)):    #len(db_np)
#        result_set = RadiusNNResultSet(capacity=k)
#        query = db_np[i,:]           #对每一个点进行KNN搜索
#        octree.octree_radius_search(root, db_np, result_set, query)
#    knn_time_sum += time.time() - begin_t
    

    
    print("Octree: build %.3fms, knn %.3fms" % (construction_time_sum * 1000, knn_time_sum * 1000))    


    print("kdtree --------------")
    #spatial.KDTree
    construction_time_sum = 0
    knn_time_sum = 0

    #construction
    begin_t = time.time()
    tree = spatial.KDTree(db_np, leaf_size)
    construction_time_sum += time.time() - begin_t

    #search
    begin_t = time.time()
    for ind ,p in enumerate(filter_db_np):
        result_set = KNNResultSet(capacity=k)
        scipyKdtreeSearch(tree,result_set,p)
    knn_time_sum += time.time() - begin_t
    
    print("Kdtree_spatial: build %.3fms, knn %.3fms" % (construction_time_sum * 1000, knn_time_sum * 1000))

    
    #KDtree  
    construction_time_sum = 0
    knn_time_sum = 0
#    radius_time_sum = 0
#    brute_time_sum = 0
    
    begin_t = time.time()
    root = kdtree.kdtree_construction(db_np, leaf_size)
    construction_time_sum += time.time() - begin_t

    begin_t = time.time()
    for i in range(len(filter_db_np)):   #len(db_np)
        result_set = KNNResultSet(capacity=k)
        query = filter_db_np[i,:]
        kdtree.kdtree_knn_search(root, db_np, result_set, query)
    knn_time_sum += time.time() - begin_t
    
#    begin_t = time.time()
#    for i in range(len(db_np)):   #len(db_np)
#        result_set = RadiusNNResultSet(radius=radius)
#        query = db_np[i,:]
#        kdtree.kdtree_radius_search(root, db_np, result_set, query)
#    knn_time_sum += time.time() - begin_t
    print("Kdtree: build %.3fms, knn %.3fms" % (construction_time_sum * 1000, knn_time_sum * 1000))
    
    print("Brute-force search --------------")
    #暴力算法
    Brute_time_sum = 0
    begin_t = time.time()
    for i in range(len(filter_db_np)):
        query = filter_db_np[i, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
        nn_idx = np.argsort(diff)
        nn_dist = diff[nn_idx]
    Brute_time_sum = time.time() - begin_t
    print("Brute-force search: %.3fms" % (Brute_time_sum * 1000))



if __name__ == '__main__':
    main()