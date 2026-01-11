## DR-Methods
DR-Methods软件包中提供了一些常见的降维方法。下面是这些降维方法的案例以及参考文献。

### 案例
- Two Dimensional PCA [1]
```python
from DR import TD_PCA
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', parser='auto', version=1)
images, target = mnist["data"].to_numpy(), mnist["target"].to_numpy()
embedding = TD_PCA().fit_transform(data=images, sample_weight=28, sample_height=28)
```

- Two Dimensional Two Direct PCA [2]
```python
from DR import TD_TD_PCA
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', parser='auto', version=1)
images, target = mnist["data"].to_numpy(), mnist["target"].to_numpy()
embedding = TD_TD_PCA().fit_transform(data=images, sample_weight=28, sample_height=28)
```

- PCA-Net [3-4]
```python
from DR import PCANet
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
mnist = fetch_openml('mnist_784', parser='auto', version=1)
images, target = mnist["data"].to_numpy(), mnist["target"].to_numpy()
train_images, test_images, train_target, test_target = train_test_split(images, target, train_size=0.5)
train_images = train_images.reshape((train_images.shape[0], 28, 28))
test_images = test_images.reshape((test_images.shape[0], 28, 28))
pcanet = PCANet(k1=3, k2=3, L1=5, L2=5, block_size=10, overlapping_radio=0.1)
pcanet.fit(train_images, train_target)
pcanet.predict(test_images)
```

- Locality Preserving Projection [5]
```python
from DR import LocalityPreservingProjection
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', parser='auto', version=1)
images, target = mnist["data"].to_numpy(), mnist["target"].to_numpy()
embedding = LocalityPreservingProjection().fit_transform(data=images)
```

- PNN LPP [6-7]
```python
from DR import PNN_LPP
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', parser='auto', version=1)
images, target = mnist["data"].to_numpy(), mnist["target"].to_numpy()
embedding = PNN_LPP().fit_transform(data=images)
```

- Neighborhood Preserving Embedding [8-10]
```python
from DR import Neighborhood_Preserving_Embedding
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', parser='auto', version=1)
images, target = mnist["data"].to_numpy(), mnist["target"].to_numpy()
embedding = Neighborhood_Preserving_Embedding().fit_transform(data=images)
```

### 参考文献
```
[1] Yang J, Zhang D, Frangi A F, et al. Two-dimensional PCA: a new approach to appearance-based face representation and recognition[J]. IEEE transactions on pattern analysis and machine intelligence, 2004, 26(1): 131-137.
[2] Zhang D, Zhou Z H. (2D) 2PCA: Two-directional two-dimensional PCA for efficient face representation and recognition[J]. Neurocomputing, 2005, 69(1-3): 224-231.
[3] Chan T H, Jia K, Gao S, et al. PCANet: A simple deep learning baseline for image classification[J]. IEEE transactions on image processing, 2015, 24(12): 5017-5032.
[4] https://github.com/IshitaTakeshi/PCANet
[5] He X, Niyogi P. Locality Preserving Projections [C]. In: Advances in Neural Information Processing Systems. MIT PRESS, 2004, 153-160.
[6] Neto A C, Levada A L M. Probabilistic Nearest Neighbors Based Locality Preserving Projections for Unsupervised Metric Learning[J]. Journal of Universal Computer Science, 2024, 30(5): 603.
[7] https://github.com/alexandrelevada/PNN_LPP
[8] He X, Cai D, Yan S, et al. Neighborhood Preserving Embedding [C]. In: IEEE International Conference on Computer Vision. IEEE, 2005, 1208-1213.
[9] Pan S J, Wan L C, Liu H L, et al. Quantum algorithm for neighborhood preserving embedding[J]. Chinese Physics B, 2022, 31(6): 060304.
[10] https://github.com/thomasverardo/NPE
```
