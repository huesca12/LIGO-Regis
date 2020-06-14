from sklearn.manifold import TSNE
import time

def tsnePCA3(mainDf):
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
    tsne_pca_results = tsne.fit_transform(mainDf.iloc[:,[8,9,10]])
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    mainDf['tsne-one'] = tsne_pca_results[:,0]
    mainDf['tsne-two'] = tsne_pca_results[:,1]
