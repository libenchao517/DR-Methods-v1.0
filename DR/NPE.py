################################################################################
# 本文件用于实现NPE算法
################################################################################
# 导入模块
import numpy as np
from sklearn.neighbors import NearestNeighbors
################################################################################
# Neighborhood Preserving Embedding
class Neighborhood_Preserving_Embedding:
    """
    [1] He X, Cai D, Yan S, et al. Neighborhood Preserving Embedding [C].
     In: IEEE International Conference on Computer Vision. IEEE, 2005, 1208-1213.
    [2] Pan S J, Wan L C, Liu H L, et al. Quantum algorithm for neighborhood
    preserving embedding[J]. Chinese Physics B, 2022, 31(6): 060304.
    [3] https://github.com/thomasverardo/NPE
    """
    def __init__(
            self,
            n_components = 2,
            n_neighbors=10,transform_mode="embedding",
            alpha = 1e-5
    ):
        """
        初始化函数
        :param n_components: 低维维度
        :param n_neighbors: 近邻数
        :param transform_mode: 投影模式
               mapping: 返回投影矩阵
               embedding: 返回低维矩阵
        :param alpha: 超参数
        """
        self.n_components=n_components
        self.n_neighbors=n_neighbors
        self.transform_mode=transform_mode
        self.alpha=alpha
        self.knn = NearestNeighbors(
            n_neighbors=n_neighbors+1,
            algorithm='auto', n_jobs=-1
        )

    def Calculate_K_NN(self, data):
        """
        计算连接图
        :param data: 原始数据矩阵 [N, D]
        :return: 近邻图和索引
        """
        KNN = self.knn.fit(data)
        Q_distances = []
        Q_indices = []
        for i in range(len(data)):
            distance, index = KNN.kneighbors([data[i]])
            Q_distances.append(distance[0][1:])
            Q_indices.append(index[0][1:])
        return np.array(Q_distances), np.array(Q_indices)

    def Calculate_W(self, data, indices):
        """
        计算权重矩阵
        :param data: 原始数据矩阵 [N, D]
        :param indices: 紧邻索引
        :return: 权重矩阵
        """
        W = []
        I = np.ones(self.n_neighbors)
        for i in range(len(data)):
            xi = data[i]
            C = []
            for j in range(self.n_neighbors):
                xj = data[indices[i][j]]
                C_aux = []
                for m in range(self.n_neighbors):
                    xm = data[indices[i][m]]
                    C_jk = (xi - xj).T @ (xi - xm)
                    C_aux.append(C_jk)
                C.append(C_aux)
            C = np.array(C)
            C = C + self.alpha * np.eye(*C.shape)
            w = np.linalg.inv(C) @ I
            w = w / (I.T @ np.linalg.inv(C) @ I)
            w_zeros = np.zeros(len(data))
            np.put(w_zeros, indices[i], w)
            W.append(w_zeros)
        W = np.array(W)
        return W

    def Calculate_A(self, data, W):
        """
        计算投影矩阵
        :param data: 原始数据矩阵 [N, D]
        :param W: 权重矩阵 [N, N]
        :return: 投影矩阵 [D, d]
        """
        I = np.eye(len(data))
        M = (I - W).T @ (I - W)
        eig_values, eig_vectors = np.linalg.eig(M)
        if np.iscomplex(eig_vectors).any():
            eig_values, eig_vectors = eig_values.real, eig_vectors.real
        index_ = np.argsort(np.abs(eig_values))[1:self.n_components+1]
        z = eig_vectors[:, index_]
        a = np.linalg.inv((data.T @ data) + (self.alpha * np.eye(data.shape[1]))) @ data.T
        self.A = a @ z
        return self.A

    def fit(self, data):
        """
        :param data: 原始数据矩阵 [N, D]
        :return: 投影矩阵 [D, d]
        """
        dist, indices = self.Calculate_K_NN(data)
        W=self.Calculate_W(data, indices)
        A=self.Calculate_A(data, W)
        return A

    def fit_transform(self, data):
        """
        主函数
        :param data: 原始数据矩阵 [N, D]
        :return:
        """
        if data.shape[0] <= self.n_neighbors:
            self.n_neighbors = data.shape[0] - 1
            self.knn = NearestNeighbors(n_neighbors=data.shape[0], algorithm='auto', n_jobs=-1)
        A = self.fit(data)
        if self.transform_mode=="mapping":
            return A
        elif self.transform_mode=="embedding":
            return data @ A
