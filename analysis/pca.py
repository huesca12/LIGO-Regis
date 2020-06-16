# IMPORTS
# import external packages
from sklearn.decomposition import PCA

# import local modules
from decorators.timer import time_func


@time_func
def PCA3(pcaDf, mainDf):
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(pcaDf)

    mainDf["pca-one"] = pca_result[:, 0]
    mainDf["pca-two"] = pca_result[:, 1]
    mainDf["pca-three"] = pca_result[:, 2]
    print("Explained variation per principal component: {}".format(pca.explained_variance_ratio_))
