import scipy.stats
import matplotlib

#from sklearn import svm

step = 10
norm_features = pl.clusterNormFeatures[::step]
clusters = pl.clusters
partition = pl.partition[::step]
GMM = pl.classifier

def fit_pca(N, original_data, kernel='rbf'):
    import sklearn.decomposition

    data = original_data
    #a = np.arange(original_data.shape[0])
    #np.random.shuffle(a)
    #data = original_data[a[:N]]
    #data = np.asarray(data, dtype=np.float32)

    pca = sklearn.decomposition.PCA(N)
    #pca = sklearn.decomposition.KernelPCA(kernel=kernel)
    pca.fit(data)

    return pca

pca = fit_pca(2, norm_features)
pca_clusters = pca.transform(clusters)
pca_features = pca.transform(norm_features)

def plot_gmm(axes, classifier, clusters, X, Y=None):
    import matplotlib.pylab as pl
    import numpy as np

    def make_ellipses(gmm, ax, cm):
        global pca
        import matplotlib as mpl
        #for n, color in enumerate('rgb'):
        for n in xrange(gmm.means.shape[0]): 
            var = gmm.covars[n][np.identity(gmm.means.shape[1], dtype=bool)]
            v = pca.transform(var)
            #v, w = np.linalg.eigh(gmm.covars[n][:2, :2])
            #u = w[0] / np.linalg.norm(w[0])
            #angle = np.arctan(u[1] / u[0])
            #angle = 180 * angle / np.pi  # convert to degrees
            #v *= 9
            #v /= 10.0
            means = pca.transform(gmm.means[n, :])
            ell = mpl.patches.Ellipse(means, v[0], v[1], 0,
                                      color=cm(n), alpha=0.5)
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(0.5)
            ax.add_artist(ell)

    cm = pl.cm.Paired
    pl.set_cmap(cm)

    h = pl.subplot(1, 1, 1)
    make_ellipses(classifier, h, cm)

    # Plot also the training points
    if Y is not None:
        pl.scatter(X[:,0], X[:,1], c=Y)
    else:
        pl.scatter(X[:,0], X[:,1])

    pl.xticks(())
    pl.yticks(())

    pl.scatter(clusters[:,0], clusters[:,1], s=200, marker='x', color='black')

plotWindow = plot(show_toolbar=True)
fig, axes = plotWindow.get_figure_and_axes(0)
plot_gmm(axes, GMM, pca_clusters, pca_features, partition)
plotWindow.show()
