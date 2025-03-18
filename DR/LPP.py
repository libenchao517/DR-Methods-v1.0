################################################################################
# 本文件用于实现LPP算法PNN-LPP算法
################################################################################
# 导入模块
import sys
import scipy
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph
################################################################################
# Locality Preserving Projection
class LocalityPreservingProjection:
    """
    [1] He X, Niyogi P. Locality Preserving Projections [C].
    In: Advances in Neural Information Processing Systems.
    MIT PRESS, 2004, 153-160.
    """
    def __init__(
            self,
            n_components=2,
            n_neighbors=5,
            weight_width=1.0,
            transform_mode = "embedding"
    ):
        """
        初始化函数
        :param n_components: 低维维度
        :param n_neighbors: 近邻数
        :param weight_width: 高斯核函数超参数
        :param transform_mode: 投影模式
               mapping: 返回投影矩阵
               embedding: 返回低维矩阵
        """
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.weight_width = weight_width
        self.transform_mode = transform_mode

    def rbf(self, dist):
        """
        高斯核函数
        :param dist: 距离矩阵 [N, N]
        :return: 核矩阵 [N, N]
        """
        return np.exp(-(dist / self.weight_width))

    def Calculate_W(self, data):
        """
        计算基于邻域的权重矩阵
        :param data: 原始数据矩阵 [N, D]
        :return: 权重矩阵 [N, N]
        """
        dist = pairwise_distances(data)
        dist[dist < 0] = 0
        N = dist.shape[0]
        rbf_dist = self.rbf(dist)
        W = np.zeros([N, N])
        for i in range(N):
            index_ = np.argsort(dist[i])[1:1 + self.n_neighbors] #邻居索引
            W[i, index_] = rbf_dist[i, index_]
            W[index_, i] = rbf_dist[index_, i]
        return W

    def Calculate_A(self, X, W):
        """
        计算投影矩阵
        :param X: 原始数据矩阵 [N, D]
        :param W: 权重矩阵 [N, N]
        :return: 投影矩阵 [D, d]
        """
        N = X.shape[0]
        D = np.zeros_like(W)
        for i in range(N):
            D[i, i] = np.sum(W[i])
        L = D - W
        XDXT = np.dot(np.dot(X.T, D), X)
        XLXT = np.dot(np.dot(X.T, L), X)
        eig_val, eig_vec = np.linalg.eig(np.dot(np.linalg.pinv(XDXT), XLXT))
        sort_index_ = np.argsort(np.abs(eig_val))
        eig_val = eig_val[sort_index_]
        j = 0
        while eig_val[j] < 1e-6:
            j += 1
        sort_index_ = sort_index_[j: j + self.n_components]
        self.components_ = eig_vec[:, sort_index_]
        return self.components_

    def fit(self, X):
        """
        :param X: 原始数据矩阵 [N, D]
        :return:
        """
        W = self.Calculate_W(X)
        self.Calculate_A(X, W)

    def fit_transform(self, data):
        """
        主函数
        :param data: 原始数据矩阵 [N, D]
        :return:
        """
        self.fit(data)
        if self.transform_mode == "mapping":
            return self.components_
        elif self.transform_mode == "embedding":
            Y = np.dot(data, self.components_)
            if np.allclose(np.imag(Y), 0):
                Y = Y.real
            return Y

################################################################################
# Probabilistic Nearest Neighbor Locality Preserving Projection
class PNN_LPP:
    """
    [1] Neto A C, Levada A L M. Probabilistic Nearest Neighbors
    Based Locality Preserving Projections for Unsupervised Metric
    Learning[J]. Journal of Universal Computer Science, 2024, 30(5): 603.
    [2] https://github.com/alexandrelevada/PNN_LPP
    """
    def __init__(
            self,
            n_components = 2,
            t=1.0,
            n_neighbors=15,
            transform_mode="embedding",
            mode = 'distancias'
    ):
        """
        初始化函数
        :param n_components: 低维维度
        :param t: 高斯核函数超参数
        :param n_neighbors: 近邻数
        :param transform_mode: 投影模式
               mapping: 返回投影矩阵
               embedding: 返回低维矩阵
        :param mode: 计算权重矩阵的方法
        """
        self.n_components = n_components
        self.t = t
        self.n_neighbors = n_neighbors
        self.transform_mode = transform_mode
        self.mode = mode

    def fit_transform(self, data):
        A = self.fit(data)
        if self.transform_mode == "mapping":
            return A
        elif self.transform_mode == "embedding":
            return np.dot(data, A)

    def fit(self, X):
        n=X.shape[0]
        nn = round(np.sqrt(n))
        # 计算k近邻图
        if self.mode.lower() == 'distancias':
            knnGraph = kneighbors_graph(X, n_neighbors=self.n_neighbors, mode='distance')
            knnGraph.data = np.exp(-(knnGraph.data ** 2) / self.t)
        else:
            knnGraph = kneighbors_graph(X, n_neighbors=self.n_neighbors, mode='connectivity')
        # 计算权重
        W = knnGraph.toarray()
        W1 = np.zeros(W.shape)
        P = np.zeros(W.shape)
        for i in range(n):
            distancias = W[i, :]
            order = distancias.argsort()
            distancias.sort()
            for j in range(n):
                if (distancias[nn + 1] - distancias[1]) != 0:
                    W1[i, j] = (distancias[nn + 1] - distancias[j]) / (distancias[nn + 1] - distancias[1])  # PNN
                else:
                    W1[i, j] = 0
                if W1[i, j] == 0:
                    W1[i, j] = 10 ** (-15)
            P[i, order[1:nn + 1]] = W1[i, 1:nn+ 1]
        W = P.copy()
        # 计算投影矩阵
        D = np.diag(W.sum(1))
        L = D - W
        X = X.T
        M1 = np.dot(np.dot(X, D), X.T)
        M2 = np.dot(np.dot(X, L), X.T)
        if np.linalg.cond(M1) < 1 / sys.float_info.epsilon:
            M = np.dot(np.linalg.inv(M1), M2)
        else:
            M1 = M1 + 0.00001 * np.eye(M1.shape[0])
            M = np.dot(np.linalg.inv(M1), M2)
        lambdas, alphas = scipy.linalg.eigh(M, eigvals=(0, self.n_components - 1))
        return alphas
