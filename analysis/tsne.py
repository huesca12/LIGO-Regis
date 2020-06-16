# IMPORTS
# import external packages
from sklearn.manifold import TSNE
import time

# import local modules
from decorators.timer import time_func


@time_func
def tsnePCA3(mainDf):
    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
    tsne_pca_results = tsne.fit_transform(mainDf.iloc[:, [8, 9, 10]])
    mainDf["tsne-one"] = tsne_pca_results[:, 0]
    mainDf["tsne-two"] = tsne_pca_results[:, 1]
