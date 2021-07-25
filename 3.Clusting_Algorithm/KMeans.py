# 文件功能： 实现 K-Means 算法

import numpy as np
import random
import matplotlib.pyplot as plt
from result_set import KNNResultSet, RadiusNNResultSet
import  kdtree as kdtree

# 二维点云显示函数
def Point_Show(point,color):
    x = []
    y = []
    point = np.asarray(point)
    for i in range(len(point)):
        x.append(point[i][0])
        y.append(point[i][1])
    plt.scatter(x, y,color=color)
    plt.show()


class K_Means(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, n_clusters=2, tolerance=0.0001, max_iter=300):
        self.k_ = n_clusters
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter

    def fit(self, data):
        # 作业1
        # 屏蔽开始
        self.centers_ = data[random.sample(range(data.shape[0]),self.k_)]    #random.sample(list,num)
        old_centers = np.copy(self.centers_)                                          #存储old_centers
        #step2 E-Step(expectation)：N个点、K个中心，求N个点到K个中心的nearest-neighbor
        #kd-tree config
        leaf_size = 1
        k = 1  # 结果每个点选取属于自己的类中心
        for _ in range(self.max_iter_):
            labels = [[] for i in range(self.k_)]        #用于分类所有数据点
            root = kdtree.kdtree_construction(self.centers_ , leaf_size=leaf_size)    #对中心点进行构建kd-tree
            for i in range(data.shape[0]):       #对每一个点在4个中心点中进行 1-NN的搜索
                result_set = KNNResultSet(capacity=k)
                query =  data[i]
                kdtree.kdtree_knn_search(root, self.centers_, result_set, query)     #返回对应中心点的索引
                # labels[result_set.output_index].append(data[i])
                #print(result_set)
                output_index = result_set.knn_output_index()[0]                 #获取最邻近点的索引
                labels[output_index].append(data[i])             #将点放入类中
            #step3 M-Step(maximization)：更新中心点的位置，把属于同一个类的数据点求一个均值，作为这个类的中心值
            for i in range(self.k_):     #求K类里，每个类的的中心点
                points = np.array(labels[i])
                self.centers_[i] = points.mean(axis=0)       #取点的均值，作为新的聚类中心
                # print(points)
                # print(self.centers_[i])
            if np.sum(np.abs(self.centers_ - old_centers)) < self.tolerance_ * self.k_:  # 如果前后聚类中心的距离相差小于self.tolerance_ * self.k_ 输出
                break
            old_centers = np.copy(self.centers_)     #保存旧中心点
        self.fitted = True
        # 屏蔽结束

    def predict(self, p_datas):
        result = []
        # 作业2
        # 屏蔽开始
        if not self.fitted:
            print('Unfitter. ')
            return result
        for point in p_datas:
            diff = np.linalg.norm(self.centers_ - point, axis=1)     #使用二范数求解每个点对新的聚类中心的距离
            result.append(np.argmin(diff))                           #返回离该点距离最小的聚类中心，标记rnk = 1
        # 屏蔽结束
        return result

if __name__ == '__main__':
    db_size = 10
    dim = 2
    x = np.random.rand(db_size,dim)
    x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
#    x = np.genfromtxt(r"point.txt",delimiter="").reshape((-1,2))
    Point_Show(x,color="blue")
    k_means = K_Means(n_clusters=2)    #计算迭代后的中心点
    k_means.fit(x)                     #计算每个点属于哪个类
    cat = k_means.predict(x)
    print(cat)
    cluster = [[] for i in range(2)]        #用于分类所有数据点
    for i in range(len(x)):
        if cat[i] == 0:
            cluster[0].append(x[i])
        else:cluster[1].append(x[i])
    Point_Show(cluster[0],"red")
    Point_Show(cluster[1], "yellow")
