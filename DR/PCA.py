################################################################################
# 本文件用于实现PCA的相关算法
################################################################################
# 导入模块
import gc
import copy
import numpy as np
from scipy import signal
from sklearn.svm import SVC
from sklearn.decomposition import PCA
################################################################################
# Two Dimensional PCA
class TD_PCA:
    """
    Yang J, Zhang D, Frangi A F, et al.
    Two-dimensional PCA: a new approach to appearance-based face representation and recognition[J].
    IEEE transactions on pattern analysis and machine intelligence, 2004, 26(1): 131-137.
    """
    def __init__(
            self,
            mode='dimension',
            n_components=5,
            rate=0.95,
            transform_mode="embedding"
    ):
        """
        初始化函数
        :param mode: 维度确定方法
               dimension: 给定低维维度
               auto: 通过累积贡献率确定
        :param n_components: 低维维度
        :param rate: 累积贡献率
        :param transform_mode: 投影模式
               mapping: 返回投影矩阵
               embedding: 返回低维矩阵
               projection: 计算训练数据和局外样本数据的低维矩阵
        """
        self.mode = mode
        self.n_components = n_components
        self.transform_mode = transform_mode
        self.rate = rate

    def Calculate_Mean(self, data):
        """
        计算均值
        :param data: 原始数据矩阵 [N, m, n]
        :return: 均值数据 [m, n]
        """
        data_shape = data[0].shape
        mean_data = np.zeros(data_shape)
        for d in data:
            mean_data = mean_data + d
        mean_data = mean_data / float(len(data))
        return mean_data

    def Calculate_Cov(self, data, mean_data):
        """
        计算协方差矩阵
        :param data: 原始数据矩阵 [N, m, n]
        :param mean_data: 均值数据 [m, n]
        :return: 协方差矩阵 [n, n]
        """
        data_shape = mean_data.shape
        cov = np.zeros((data_shape[1], data_shape[1]), dtype=float)
        for d in data:
            diff = d - mean_data
            cov = cov + np.dot(diff.T, diff)
        return cov

    def Calculate_Eig(self, cov):
        """
        计算特征值和特征向量
        :param cov: 协方差矩阵 [n, n]
        :return: 特征值，特征向量，排序
        """
        eig_values, eig_vectors = np.linalg.eig(cov)
        if np.iscomplex(eig_vectors).any():
            eig_values, eig_vectors = eig_values.real, eig_vectors.real
        sorted_index = np.argsort(eig_values)
        eig_values.sort()
        sorted_values = eig_values[::-1]
        return sorted_values, eig_vectors, sorted_index

    def Calculate_dim(self, values):
        """
        根据累积贡献率计算低维维度
        :param values: 排序后的特征值
        :return: 低维维度
        """
        dim = 0
        sum_values = 0
        sum_temp = 0
        for v in values:
            sum_values += v
        while sum_temp < self.rate * sum_values:
            sum_temp += values[dim]
            dim += 1
        return dim

    def Calculate_embedding(self, data, A):
        """
        计算投影
        :param data: 原始数据矩阵 [N, m, n]
        :param A: 投影矩阵 [n, d]
        :return: 低维投影 [N, m, d]
        """
        embedding = []
        for d in data:
            embedding.append((d@A).flatten())
        return np.array(embedding)

    def fit(self, data):
        """
        计算投影矩阵
        :param data: 原始数据矩阵 [N, m, n]
        :return: 投影矩阵 [n, d]
        """
        mean_data = self.Calculate_Mean(data)
        cov = self.Calculate_Cov(data, mean_data)
        values, vectors, index = self.Calculate_Eig(cov)
        if self.mode.lower() == 'auto':
            self.dim = self.Calculate_dim(values)
        elif self.mode.lower() == 'dimension':
            self.dim = self.n_components
        return vectors[:, index[:-self.dim - 1:-1]]

    def fit_transform(
            self,
            data,
            oos_data=None,
            sample_height=32,
            sample_weight=32
    ):
        """
        主函数
        :param data: 原始数据矩阵 [N, m*n]
        :param oos_data: [可选]局外样本矩阵 [M, m*n]
        :param sample_height: 图像高度 m
        :param sample_weight: 图像宽度 n
        :return:
        """
        data = data.reshape(data.shape[0], sample_height, sample_weight)
        A = self.fit(data)
        if self.transform_mode.lower() == "mapping":
            return A
        elif self.transform_mode.lower() == "embedding":
            embedding = self.Calculate_embedding(data, A)
            return embedding
        elif self.transform_mode.lower() == "projection":
            oos_data = oos_data.reshape(oos_data.shape[0], sample_height, sample_weight)
            embedding = self.Calculate_embedding(data, A)
            oos_embedding = self.Calculate_embedding(oos_data, A)
            return embedding, oos_embedding

################################################################################
# Two Dimensional Two Direct PCA
class TD_TD_PCA:
    """
    Zhang D, Zhou Z H.
    (2D) 2PCA: Two-directional two-dimensional PCA for efficient face representation and recognition[J].
    Neurocomputing, 2005, 69(1-3): 224-231.
    """
    def __init__(
            self,
            mode='dimension',
            components_z=5,
            components_x=5,
            rate=0.95,
            transform_mode="embedding"
    ):
        """
        初始化函数
        :param mode: 维度确定方法
               dimension: 给定低维维度
               auto: 通过累积贡献率确定
        :param n_components: 低维维度
        :param rate: 累积贡献率
        :param transform_mode: 投影模式
               mapping: 返回投影矩阵
               embedding: 返回低维矩阵
               projection: 计算训练数据和局外样本数据的低维矩阵
        """
        """
        初始化函数
        :param mode: 维度确定方法
               dimension: 给定低维维度
               auto: 通过累积贡献率确定
        :param components_z: 纵向维度
        :param components_x: 横向维度
        :param rate: 累积贡献率
        :param transform_mode: 投影模式
               mapping: 返回投影矩阵
               embedding: 返回低维矩阵
               projection: 计算训练数据和局外样本数据的低维矩阵
        """
        self.mode = mode
        self.components_z = components_z
        self.components_x = components_x
        self.rate = rate
        self.transform_mode = transform_mode

    def Calculate_Mean(self, data):
        """
        计算均值
        :param data: 原始数据矩阵 [N, m, n]
        :return: 均值数据 [m, n]
        """
        data_shape = data[0].shape
        mean_data = np.zeros(data_shape)
        for d in data:
            mean_data += d
        mean_data = mean_data / float(len(data))
        return mean_data

    def Calculate_Cov(self, data, mean_data):
        """
        计算协方差矩阵
        :param data: 原始数据矩阵 [N, m, n]
        :param mean_data: 均值数据 [m, n]
        :return:
         cov_z [m, m]
         cov_x [n, n]
        """
        data_shape = mean_data.shape
        cov_z = np.zeros((data_shape[0], data_shape[0]), dtype=float)
        cov_x = np.zeros((data_shape[1], data_shape[1]), dtype=float)
        for d in data:
            diff = d - mean_data
            cov_z = cov_z + np.dot(diff, diff.T)
            cov_x = cov_x + np.dot(diff.T, diff)
        return cov_z, cov_x

    def Calculate_Eig(self, cov):
        """
        计算特征值和特征向量
        :param cov: 协方差矩阵 [n, n]
        :return: 特征值，特征向量，排序
        """
        eig_values, eig_vectors = np.linalg.eig(cov)
        sorted_index = np.argsort(eig_values)
        eig_values.sort()
        sorted_values = eig_values[::-1]
        return sorted_values, eig_vectors, sorted_index
    def Calculate_dim(self, values):
        """
        根据累积贡献率计算低维维度
        :param values: 排序后的特征值
        :return: 低维维度
        """
        dim = 0
        sum_values = 0
        sum_temp = 0
        for v in values:
            sum_values += v
        while sum_temp < self.rate * sum_values:
            sum_temp += values[dim]
            dim += 1
        return dim
    def Calculate_embedding(self, data, Z, X):
        """
        计算投影
        :param data: 原始数据矩阵 [N, m, n]
        :param Z: 纵向投影矩阵 [m, dim_z]
        :param X: 横向投影矩阵 [n, dim_x]
        :return: 低维投影 [N, dim_z, dim_x]
        """
        embedding = []
        for d in data:
            embedding.append((Z.T@d@X).flatten())
        return np.array(embedding)

    def fit(self, data):
        """
        计算投影矩阵
        :param data: 原始数据矩阵 [N, m, n]
        :return:
        横向投影矩阵 Z [m, dim_z]
        纵向投影矩阵 X [n, dim_x]
        """
        mean_data = self.Calculate_Mean(data)
        cov_z, cov_x = self.Calculate_Cov(data, mean_data)
        z_values, z_vectors, z_index = self.Calculate_Eig(cov_z)
        x_values, x_vectors, x_index = self.Calculate_Eig(cov_x)
        if self.mode.lower() == 'auto':
            self.dim_z = self.Calculate_dim(z_values)
            self.dim_x = self.Calculate_dim(x_values)
        elif self.mode.lower() == 'dimension':
            self.dim_z = self.components_z
            self.dim_x = self.components_x
        return z_vectors[:, z_index[:-self.dim_z-1:-1]], x_vectors[:, x_index[:-self.dim_x-1:-1]]

    def fit_transform(self, data, sample_height=32, sample_weight=32):
        """
        主函数
        :param data: 原始数据矩阵 [N, m*n]
        :param oos_data: [可选]局外样本矩阵 [M, m*n]
        :param sample_height: 图像高度 m
        :param sample_weight: 图像宽度 n
        :return:
        """
        data = data.reshape(data.shape[0], sample_height, sample_weight)
        Z, X = self.fit(data)
        if self.transform_mode == "mapping":
            return Z, X
        elif self.transform_mode == "embedding":
            embedding = self.Calculate_embedding(data, Z, X)
            return embedding

################################################################################
# PCA-Net
class PCANet:
    """
    [1] Chan T H, Jia K, Gao S, et al.
    PCANet: A simple deep learning baseline for image classification[J].
    IEEE transactions on image processing, 2015, 24(12): 5017-5032.
    [2] https://github.com/IshitaTakeshi/PCANet
    """
    def __init__(
            self,
            k1, k2, L1, L2, block_size,
            overlapping_radio=0,
            linear_classifier='svm',
            spp_parm=None,
            dim_reduction=None
    ):
        """
        初始化函数
        :param k1:
        :param k2:
        :param L1:
        :param L2:
        :param block_size:
        :param overlapping_radio:
        :param linear_classifier:
        :param spp_parm:
        :param dim_reduction:
        """
        self.k1 = k1
        self.k2 = k2
        self.L1 = L1
        self.L2 = L2
        self.block_size = block_size
        self.overlapping_radio = overlapping_radio
        self.l1_filters = None
        self.l2_filters = None
        if linear_classifier == 'svm':
            self.classifier = SVC()
        else:
            self.classifier = None
        self.spp_parm = spp_parm
        if dim_reduction:
            self.dim_reduction = dim_reduction
        else:
            self.dim_reduction = None

    def mean_remove_img_patches(self, img, width, height):
        """
        去平均化
        :param img:
        :param width:
        :param height:
        :return:
        """
        in_img = copy.deepcopy(img)
        del img
        cap_x_i = np.empty((self.k1 * self.k2, width * height))
        idx = 0
        for i in range(width):
            for j in range(height):
                patten = in_img[i: i + self.k1, j:j + self.k2].copy()
                cap_x_i[:, idx] = patten.flatten()
                idx += 1
        cap_x_i -= np.mean(cap_x_i, axis=0)
        return cap_x_i

    def get_filter(self, train_data, num_filter, rgb=False):
        """
        训练滤波器
        :param train_data:
        :param num_filter:
        :param rgb:
        :return:
        """
        if rgb:
            num_chn = train_data.shape[3]
        img_num, img_width, img_height = train_data.shape[0], train_data.shape[1], train_data.shape[2]
        patch_width = self.k1
        patch_height = self.k2
        img_patch_height = img_height - patch_height + 1
        img_patch_width = img_width - patch_width + 1
        if rgb:
            cap_c = np.zeros((num_chn * patch_width * patch_height, num_chn * patch_width * patch_height))
        else:
            cap_c = np.zeros((patch_width * patch_height, patch_width * patch_height))
        for n in range(img_num):
            if rgb:
                im = np.array([self.mean_remove_img_patches(train_data[n][:, :, i], img_patch_width, img_patch_height) for i in range(num_chn)]).reshape((num_chn * patch_width * patch_height, -1))
                cap_c += np.matmul(im, im.T)
            else:
                im = self.mean_remove_img_patches(train_data[n], img_patch_width, img_patch_height)
                cap_c += np.matmul(im, im.T)
            if n % 10000 == 0:
                gc.collect()
        vals, vecs = np.linalg.eig(cap_c / img_num * im.shape[1])
        idx_w_l1 = np.argsort(np.real(vals))[:-(num_filter + 1):-1]
        cap_w_l1 = np.real(vecs[:, idx_w_l1])
        if rgb:
            filters = cap_w_l1.T.reshape(num_filter, patch_width, patch_height, num_chn)
        else:
            filters = cap_w_l1.T.reshape(num_filter, patch_width, patch_height)
        return filters

    def get_historgram(self, decimal_result):
        """
        计算直方图
        :param decimal_result:
        :return:
        """
        histo_bins = range(2 ** self.L2)
        img_width, img_height = decimal_result.shape[1], decimal_result.shape[2]
        step_size = int(self.block_size * (1 - self.overlapping_radio))
        img_patch_height = img_height - self.block_size + 1
        img_patch_width = img_width - self.block_size + 1
        for l in range(self.L1):
            for i in range(0, img_patch_width, step_size):
                for j in range(0, img_patch_height, step_size):
                    patten = decimal_result[i: i + self.block_size, j:j + self.block_size]
                    histogram, _ = np.histogram(patten, histo_bins)

    def extract_features(self, img, rgb=False):
        """
        提取特征
        :param img:
        :param rgb:
        :return:
        """
        if rgb:
            conv_result = np.empty((self.L1, self.L2, img.shape[0], img.shape[1]))
            for i in range(len(self.l1_filters)):
                l1_result = np.empty(img.shape)
                for ch in range(img.shape[2]):
                    l1_result[:, :, ch] = signal.convolve2d(img[:, :, ch], self.l1_filters[i, :, :, ch], 'same')
                l1_result = np.sum(l1_result, axis=-1)
                for j in range(len(self.l2_filters)):
                    conv_result[i, j, :, :] = signal.convolve2d(l1_result, self.l2_filters[j], 'same')
        else:
            conv_result = np.empty((self.L1, self.L2, img.shape[0], img.shape[1]))
            for i in range(len(self.l1_filters)):
                l1_result = signal.convolve2d(img, self.l1_filters[i], 'same')
                for j in range(len(self.l2_filters)):
                    conv_result[i, j, :, :] = signal.convolve2d(l1_result, self.l2_filters[j], 'same')
        binary_result = np.where(conv_result > 0, 1, 0)
        decimal_result = np.zeros((self.L1, img.shape[0], img.shape[1]))
        for i in range(len(self.l2_filters)):
            decimal_result += (2 ** i) * binary_result[:, i, :, :]
        histo_bins = range(2 ** self.L2 + 1)
        img_width, img_height = decimal_result.shape[1], decimal_result.shape[2]
        step_size = int(self.block_size * (1 - self.overlapping_radio))
        img_patch_height = img_height - self.block_size + 1
        img_patch_width = img_width - self.block_size + 1
        if self.spp_parm:
            feature_width = len(range(0, img_patch_width, step_size))
            feature_height = len(range(0, img_patch_height, step_size))
            feature = []
            for l in range(self.L1):
                before_spp = np.empty((feature_width, feature_height, len(histo_bins)-1))
                spp_idx_i = 0
                for i in range(0, img_patch_width, step_size):
                    spp_idx_j = 0
                    for j in range(0, img_patch_height, step_size):
                        patten = decimal_result[l, i: i + self.block_size, j:j + self.block_size]
                        histogram, _ = np.histogram(patten, histo_bins)
                        before_spp[spp_idx_i, spp_idx_j, :] = histogram
                        spp_idx_j += 1
                    spp_idx_i += 1
                after_spp = []
                for side in self.spp_parm:
                    W = feature_width // side
                    H = feature_height // side
                    for side_i in range(side):
                        for side_j in range(side):
                            after_spp.append(before_spp[side_i*W:(side_i+1)*W, side_j*H:(side_j+1)*H:, :].max(axis=(0, 1)))
                feature.append(after_spp)
            if self.dim_reduction:
                feature = np.array(feature).swapaxes(0, 1)
                dim_reduction_to = self.dim_reduction // feature.shape[1]
                after_pca = []
                for i in range(feature.shape[0]):
                    pca = PCA(n_components=dim_reduction_to, copy=False)
                    after_pca.append(pca.fit_transform(feature[i]))
                return np.array(after_pca).reshape((-1))
            else:
                return np.array(feature).reshape((-1))
        else:
            feature = []
            for l in range(self.L1):
                for i in range(0, img_patch_width, step_size):
                    for j in range(0, img_patch_height, step_size):
                        patten = decimal_result[l, i: i + self.block_size, j:j + self.block_size]
                        histogram, _ = np.histogram(patten, histo_bins)
                        feature.append(histogram)
            return np.array(feature).reshape((-1))

    def fit(self, train_data, train_labels):
        """
        训练器
        :param train_data:
        :param train_labels:
        :return:
        """
        if len(train_data.shape) == 4:
            rgb = True
            num_chr = train_data.shape[3]
        else:
            rgb = False
        self.l1_filters = self.get_filter(train_data, self.L1, rgb)
        if rgb:
            l1_conv_result = np.empty((train_data.shape[0] * self.l1_filters.shape[0], train_data.shape[1], train_data.shape[2], train_data.shape[3]))
        else:
            l1_conv_result = np.empty((train_data.shape[0] * self.l1_filters.shape[0], train_data.shape[1], train_data.shape[2]))
        l1_conv_idx = 0
        for image in train_data:
            for kernel in self.l1_filters:
                if rgb:
                    for chn in range(num_chr):
                        l1_conv_result[l1_conv_idx, :, :, chn] = signal.convolve2d(image[:, :, chn], kernel[:, :, chn], 'same')
                else:
                    l1_conv_result[l1_conv_idx, :, :] = signal.convolve2d(image, kernel, 'same')
                l1_conv_idx += 1
        if rgb:
            l1_conv_result = np.sum(l1_conv_result, axis=-1)
        self.l2_filters = self.get_filter(l1_conv_result, self.L2)
        features = []
        for i in range(len(train_data)):
            if i % 1000 == 0:
                gc.collect()
            feature = self.extract_features(train_data[i], rgb)
            features.append(feature)
        self.classifier.fit(features, train_labels)

    def predict(self, test_data):
        """
        预测器
        :param test_data:
        :return:
        """
        if len(test_data.shape) == 4:
            rgb = True
        else:
            rgb = False
        test_features = []
        for i in range(len(test_data)):
            test_features.append(self.extract_features(test_data[i], rgb))
        predictions = self.classifier.predict(test_features)
        return predictions
