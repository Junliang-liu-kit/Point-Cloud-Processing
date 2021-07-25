# 文件功能：
#     1. 从数据集中加载点云数据
#     2. 从点云数据中滤除地面点云
#     3. 从剩余的点云中提取聚类

import numpy as np
import os
import struct
from sklearn import cluster, datasets, mixture
from itertools import cycle, islice
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import open3d as o3d 
from pyntcloud import PyntCloud
from pandas import DataFrame
import math
import random
#import Spectral as sp
from collections import defaultdict
from scipy import spatial
from sklearn.neighbors import KDTree # KDTree 进行搜索


# 功能：从kitti的.bin格式点云文件中读取点云
# 输入：
#     path: 文件路径
# 输出：
#     点云数组
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

# 功能：从点云文件中滤除地面点
# 输入：
#     data: 一帧完整点云
# 输出：
#     segmengted_cloud: 删除地面点之后的点云
def ground_segmentation(data, sigma_num, outlier_ratio_num):
    # 作业1
    # 屏蔽开始
    # 数据初始化
    idx_segmented = []
    segmented_cloud = []
    iters = 100   #最大迭代次数
    sigma = sigma_num   #数据和模型之间可接受的最大差值
    ##最好模型的参数估计和内点数目,平面表达方程为   aX + bY + cZ +D= 0
    best_a = 0
    best_b = 0
    best_c = 0
    best_d = 0
    pretotal = 0  #上一次inlier的点数
    #希望的到正确模型的概率
    p =0.99
    n = len(data)  #点的数目
    outlier_ratio = outlier_ratio_num #e :outlier_ratio
    for i in range(iters):
        ground_cloud = []
        idx_ground = []
        #step1 选择可以估计出模型的最小数据集，地面分离是平面拟合，三个点
        sample_index = random.sample(range(n),3)     #重数据集中随机选取3个点
        point1 = data[sample_index[0]]
        point2 = data[sample_index[1]]
        point3 = data[sample_index[2]]
        #step2 求解模型
        ##求解法向量
        point1_2 = (point1-point2)
        point1_3 = (point1-point3)
        N = np.cross(point1_3,point1_2)  #向量叉乘求解 平面法向量
        ##slove model 求解模型的a,b,c,d
        a = N[0]
        b = N[1]
        c = N[2]
        d = -N.dot(point1)
        #step3 将所有数据带入模型，计算出“内点”的数目；(累加在一定误差范围内的适合当前迭代推出模型的数据)
        total_inlier = 0
        pointn_1 = (data - point1)  #sample（三点）外的点 与 sample内的三点其中一点 所构成的向量
        distance = abs(pointn_1.dot(N))/ np.linalg.norm(N)    #求距离
        ##使用距离判断inliner
        idx_ground = (distance <= sigma)
        total_inlier = np.sum(idx_ground == True)
        ##判断当前的模型是否比之前估算的模型好
        if total_inlier > pretotal:
            iters = math.log(1 - p) / math.log(1 - pow(total_inlier / n, 3))  #Iteration number
            pretotal = total_inlier
            #获取最好的 abcd 模型参数
            best_a = a
            best_b = b
            best_c = c
            best_d = d
            
            
        #判断
        if total_inlier > n*(1-outlier_ratio):
            break
    print("iters = %f" %iters)
    #提取分割后得点
    idx_segmented = np.logical_not(idx_ground)
    ground_cloud = data[idx_ground]
    segmented_cloud = data[idx_segmented]



    # 屏蔽结束

    print('origin data points num:', data.shape[0])
    print('segmented data points num:', segmented_cloud.shape[0])
    return ground_cloud,segmented_cloud


# 功能：从点云中提取聚类
# 输入：
#     data: 点云（滤除地面后的点云）
# 输出：
#     clusters_index： 一维数组，存储的是点云中每个点所属的聚类编号（参考上一章内容容易理解）

def Clustering(data, eps, Minpts):
    # 作业2
    
    #屏蔽开始
    
    """
    基于密度的点云聚类
    :param d_bbox: 点与点之间的距离矩阵
    :param eps:  最大搜索直径阈值
    :param Minpts:  最小包含其他对象数量阈值
    :return: 返回聚类结果，是一个嵌套列表,每个子列表就是这个区域的对象的序号
    """
    n = len(data)
    # 构建kd_tree
    leaf_size = 4
#    root = kdtree.kdtree_construction(data,leaf_size=leaf_size)
    tree = KDTree(data,leaf_size)
    #step1 初始化核心对象集合T,聚类个数k,聚类集合C, 未访问集合P
    T = set()    #set 集合
    k = 0        #类初始化
    cluster_index = np.zeros(n,dtype=int)      #聚类集合
    unvisited = set(range(n))   #初始化未访问集合
    #step2 通过判断，通过kd_tree radius NN找出所有核心点
    
    nearest_idx = tree.query_radius(data, eps)  # 进行radius NN搜索,半径为epsion,所有点的最临近点储存在 nearest_idx中
    for d in range(n):
#        result_set = RadiusNNResultSet(radius=eps)  #进行radius NN搜索,半径为epsion
#        kdtree.kdtree_radius_search(root,data,result_set,data[d])
#        nearest_idx = result_set.radius_nn_output_index()
        if len(nearest_idx) >= Minpts:     #临近点数 > min_sample,加入核心点
            T.add(d)    #最初得核心点
    #step3 聚类
    while len(T):     #visited core ，until all core points were visited
        unvisited_old = unvisited     #更新为访问集合
        core = list(T)[np.random.randint(0,len(T))]    #从 核心点集T 中随机选取一个 核心点core
        unvisited = unvisited - set([core])      #把核心点标记为 visited,从 unvisited 集合中剔除
        visited = []
        visited.append(core)

        while len(visited):
            new_core = visited[0]
            #kd-tree radius NN 搜索邻近
#            result_set = RadiusNNResultSet(radius=eps)  # 进行radius NN搜索,半径为epsion
#            kdtree.kdtree_radius_search(root, data, result_set, data[new_core])
#            new_core_nearest = result_set.radius_nn_output_index()   #获取new_core 得邻近点
            if new_core in T:
                S = unvisited & set(nearest_idx[new_core])    #当前 核心对象的nearest 与 unvisited 的交集
                visited +=  (list(S))                     #对该new core所能辐射的点，再做检测
                unvisited = unvisited - S          #unvisited 剔除已 visited 的点
            visited.remove(new_core)                     #new core 已做检测，去掉new core

        cluster = unvisited_old -  unvisited    #原有的 unvisited # 和去掉了 该核心对象的密度可达对象的visited就是该类的所有对象
        T = T - cluster  #去掉该类对象里面包含的核心对象,差集
        cluster_index[list(cluster)] = k
        k += 1   #类个数
    noise_cluster = unvisited
    cluster_index[list(noise_cluster)] = -1    #噪声归类为 1
    print(cluster_index)
    print("生成的聚类个数：%d" %k)
    
    #屏蔽结束
    
    return cluster_index




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

# 功能：显示聚类点云，每个聚类一种颜色
# 输入：
#      data：点云数据（滤除地面后的点云）
#      cluster_index：一维数组，存储的是点云中每个点所属的聚类编号（与上同）
    
def plot_clusters(segmented_ground, segmented_cloud, cluster_index):
    """
    Visualize segmentation results using Open3D

    Parameters
    ----------
    segmented_cloud: numpy.ndarray
        Segmented surrounding objects as N-by-3 numpy.ndarray
    segmented_ground: numpy.ndarray
        Segmented ground as N-by-3 numpy.ndarray
    cluster_index: list of int
        Cluster ID for each point

    """
    def colormap(c, num_clusters):
        """
        Colormap for segmentation result

        Parameters
        ----------
        c: int
            Cluster ID
        C

        """
        # outlier:
        if c == -1:
            color = [1]*3
        # surrouding object:
        else:
            color = [0] * 3
            color[c % 3] = c/num_clusters

        return color

    # ground element:
    pcd_ground = o3d.geometry.PointCloud()
    pcd_ground.points = o3d.utility.Vector3dVector(segmented_ground)
    pcd_ground.colors = o3d.utility.Vector3dVector(
        [
            [0, 0, 255] for i in range(segmented_ground.shape[0])
        ]
    )

    # surrounding object elements:
    pcd_objects = o3d.geometry.PointCloud()
    pcd_objects.points = o3d.utility.Vector3dVector(segmented_cloud)
    num_clusters = max(cluster_index) + 1
    pcd_objects.colors = o3d.utility.Vector3dVector(
        [
            colormap(c, num_clusters) for c in cluster_index
        ]
    )

    # visualize:
    o3d.visualization.draw_geometries([pcd_ground, pcd_objects])

def main():
    iteration_num = 1    #文件数

    # for i in range(iteration_num):
    filename = r'C:\Users\Junliang\Desktop\三维点云\第四章\HomeworkIVclustering\000006.bin'         #数据集路径
    print('clustering pointcloud file:', filename)

    origin_points_np = read_velodyne_bin(filename)   #读取数据点
    origin_points = RandomVoxelGridDownsample(origin_points_np, 0.08)    #对原始点云采样，在下面的作业中对采样后的点做KNN搜索
    print(origin_points)
    print(origin_points.shape)
    
#    origin_points_df = DataFrame(origin_points,columns=['x', 'y', 'z'])  # 选取每一列 的 第0个元素到第二个元素   [0,3)
#    point_cloud_pynt = PyntCloud(origin_points_df)  # 将points的数据 存到结构体中
#    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)  # 实例化
#    #
#    # o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云


    # # 地面分割
    ground_points, segmented_points = ground_segmentation(data=origin_points, sigma_num=0.4, outlier_ratio_num=0.6)
    
    ground_points_df = DataFrame(ground_points, columns=['x', 'y', 'z'])  # 选取每一列 的 第0个元素到第二个元素   [0,3)
    point_cloud_pynt_ground = PyntCloud(ground_points_df)  # 将points的数据 存到结构体中
    point_cloud_o3d_ground = point_cloud_pynt_ground.to_instance("open3d", mesh=False)  # 实例化
    point_cloud_o3d_ground.paint_uniform_color([0, 0, 255])
    
    
    #显示segmentd_points示地面点云
    segmented_points_df = DataFrame(segmented_points, columns=['x', 'y', 'z'])  # 选取每一列 的 第0个元素到第二个元素   [0,3)
    point_cloud_pynt_segmented = PyntCloud(segmented_points_df)  # 将points的数据 存到结构体中
    point_cloud_o3d_segmented = point_cloud_pynt_segmented.to_instance("open3d", mesh=False)  # 实例化
    point_cloud_o3d_segmented.paint_uniform_color([255, 0, 0])
    
    o3d.visualization.draw_geometries([point_cloud_o3d_ground])
    o3d.visualization.draw_geometries([point_cloud_o3d_ground,point_cloud_o3d_segmented])
    # 显示聚类结果
    # 显示segmentd_points 分割后每个类的点云
    
    cluster_index = Clustering(segmented_points,eps=0.5,Minpts=15)
    plot_clusters(ground_points, segmented_points, cluster_index)

if __name__ == '__main__':
    main()
