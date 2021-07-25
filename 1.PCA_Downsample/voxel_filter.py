# 实现voxel滤波，并加载数据集中的文件进行验证

import open3d as o3d 
import os
import numpy as np
from pyntcloud import PyntCloud
from collections import defaultdict
from pandas import DataFrame

def read_txt(filename):
    points = []
    with open(filename, 'r') as f:
        for line in f: 
            value = line.split(',')
            points.append([float(x) for x in value[:3]])
    points = np.array(points)
    return points

# 功能：对点云进行voxel滤波
# 输入：
#     point_cloud：输入点云
#     leaf_size: voxel尺寸
def voxel_filter(point_cloud, leaf_size):
    filtered_points = []
    # 作业3
    # 屏蔽开始
    # CentriodVoxelGridDownsample
    # 计算每个维度上的极值
    x_min=min(point_cloud[:][0])
    x_max=max(point_cloud[:][0])
    y_min=min(point_cloud[:][1])
    y_max=max(point_cloud[:][1])
    z_min=min(point_cloud[:][2])
    z_max=max(point_cloud[:][2])
    Dx=int((x_max-x_min)/leaf_size)
    Dy=int((y_max-y_min)/leaf_size)
    Dz=int((z_max-z_min)/leaf_size)
    # 用字典存储h及其对应的点
    d=defaultdict(list)
    for p in point_cloud:
        hx=int((p[0]-x_min)/leaf_size)
        hy=int((p[1]-y_min)/leaf_size)
        hz=int((p[2]-z_min)/leaf_size)
        h=hx+hy*Dx+hz*Dx*Dy
        d[h].append(p)
    # 求每个h对应的点的中心
    Centriod_Points=[]
    for key,value in d.items():
        Centriod_Point=np.mean(value,axis=0)
        Centriod_Points.append(Centriod_Point)

 
    # 屏蔽结束

    # 把点云格式改成array，并对外返回
    filtered_points = np.array(Centriod_Points, dtype=np.float64)
    return filtered_points

def main():
    # # 从ModelNet数据集文件夹中自动索引路径，加载点云
    # cat_index = 10 # 物体编号，范围是0-39，即对应数据集中40个物体
    # root_dir = '/Users/renqian/cloud_lesson/ModelNet40/ply_data_points' # 数据集路径
    # cat = os.listdir(root_dir)
    # filename = os.path.join(root_dir, cat[cat_index],'train', cat[cat_index]+'_0001.ply') # 默认使用第一个点云
    # point_cloud_pynt = PyntCloud.from_file(file_name)

    # 加载自己的点云文件"/Users/renqian/Downloads/program/cloud_data/11.ply"
    point_cloud_raw = np.genfromtxt(r'C:\Users\Junliang\Desktop\三维点云\point_cloud_txt\pc\airplane_0542.txt', delimiter=",")  #为 xyz的 N*3矩阵
    point_cloud_raw = DataFrame(point_cloud_raw[:, 0:3])  # 选取每一列 的 第0个元素到第二个元素   [0,3)
    point_cloud_raw.columns = ['x', 'y', 'z']  # 给选取到的数据 附上标题
    point_cloud_pynt = PyntCloud(point_cloud_raw)  # 将points的数据 存到结构体中


    # 转成open3d能识别的格式
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    #o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云

    # 调用voxel滤波函数，实现滤波
    point_cloud = read_txt(r'C:\Users\Junliang\Desktop\三维点云\point_cloud_txt\pc\airplane_0542.txt')
    filtered_cloud = voxel_filter(point_cloud, 0.05)
    point_cloud_o3d = o3d.geometry.PointCloud()
    point_cloud_o3d.points = o3d.utility.Vector3dVector(filtered_cloud)
    print(point_cloud_o3d)
    # 显示滤波后的点云
    o3d.visualization.draw_geometries([point_cloud_o3d])

if __name__ == '__main__':
    main()
