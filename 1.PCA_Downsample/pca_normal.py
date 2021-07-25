# 实现PCA分析和法向量计算，并加载数据集中的文件进行验证

import open3d as o3d 
import os
import numpy as np
from pyntcloud import PyntCloud
from pandas import DataFrame

# 功能：计算PCA的函数
# 输入：
#     data：点云，NX3的矩阵
#     correlation：区分np的cov和corrcoef，不输入时默认为False
#     sort: 特征值排序，排序是为了其他功能方便使用，不输入时默认为True
# 输出：
#     eigenvalues：特征值
#     eigenvectors：特征向量
def PCA(data, correlation=False, sort=True):
    # 作业1
    # 屏蔽开始
     # 1、均值归一化
    mean_vals = np.mean(data,axis=0)#对列求均值
    mid = data-mean_vals    #去中心化
    # 2、求SVD矩阵
    H = np.dot(mid.T,mid)
    eigenvectors,eigenvalues,eigenvectors_T = np.linalg.svd(H)

    # 屏蔽结束

    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors


def main():
    # 指定点云路径
    # cat_index = 10 # 物体编号，范围是0-39，即对应数据集中40个物体
    # root_dir = '/Users/renqian/cloud_lesson/ModelNet40/ply_data_points' # 数据集路径
    # cat = os.listdir(root_dir)
    # filename = os.path.join(root_dir, cat[cat_index],'train', cat[cat_index]+'_0001.ply') # 默认使用第一个点云

    # 加载原始点云"/Users/renqian/Downloads/program/cloud_data/11.ply"
    point_cloud_raw = np.genfromtxt(r'C:\Users\Junliang\Desktop\三维点云\point_cloud_txt\pc\airplane_0542.txt', delimiter=",")  #为 xyz的 N*3矩阵
    point_cloud_raw = DataFrame(point_cloud_raw[:, 0:3])  # 选取每一列 的 第0个元素到第二个元素   [0,3)
    point_cloud_raw.columns = ['x', 'y', 'z']  # 给选取到的数据 附上标题
    point_cloud_pynt = PyntCloud(point_cloud_raw)  # 将points的数据 存到结构体中

    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)  # 实例化
    o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云

    # 从点云中获取点，只对点进行处理
    points = point_cloud_pynt.points
    print('total points number is:', points.shape[0])

    # 用PCA分析点云主方向
    w, v = PCA(points)
    point_cloud_vector = v[:, 2] #点云主方向对应的向量
    print('the main orientation of this pointcloud is: ', point_cloud_vector)
    # TODO: 此处只显示了点云，还没有显示PCA
    # o3d.visualization.draw_geometries([point_cloud_o3d])
    
    # 循环计算每个点的法向量
    pcd_tree = o3d.geometry.KDTreeFlann(point_cloud_o3d)
    normals = []
    # 作业2
    # 屏蔽开始
    print(point_cloud_raw.shape[0])      #打印当前点数 20000个点
    for i in range(point_cloud_raw.shape[0]):
        # search_knn_vector_3d函数 ， 输入值[每一点，x]      返回值 [int, open3d.utility.IntVector, open3d.utility.DoubleVector]
        [_,idx,_] = pcd_tree.search_knn_vector_3d(point_cloud_o3d.points[i],10)      #取10个临近点进行曲线拟合
        # asarray和array 一样 但是array会copy出一个副本，asarray不会，节省内存
        k_nearest_point = np.asarray(point_cloud_o3d.points)[idx, :]  # 找出每一点的10个临近点，类似于拟合成曲面，然后进行PCA找到特征向量最小的值，作为法向量
        w, v = PCA(k_nearest_point)
        normals.append(v[:, 2])

    # 屏蔽结束
    normals = np.array(normals, dtype=np.float64)
    # TODO: 此处把法向量存放在了normals中
    point_cloud_o3d.normals = o3d.utility.Vector3dVector(normals)
    o3d.visualization.draw_geometries([point_cloud_o3d])


if __name__ == '__main__':
    main()
