import scipy.stats
import matplotlib

from sklearn import svm

step = 20
norm_features = pl.clusterNormFeatures[::step]
clusters = pl.clusters
partition = pl.partition[::step]
svm.SVC(
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

def plot_decision_surface(axes, clusters, X, Y=None):
    import matplotlib.pylab as pylab
    import numpy as np

    def kmeans_predict(clusters, X):
        from ..core import distance
        dist_m = distance.minkowski_dist(clusters, X)
        print 'dist_m:', dist_m.shape
        pred = np.argmin(dist_m, axis=0)
        print 'pred:', pred.shape
        return pred

    # step size in the mesh
    h = (np.max(X, axis=0) - np.min(X, axis=0) ) / 100.0
    # create a mesh to plot in
    x_min = np.min(X, axis=0) - 1
    x_max = np.max(X, axis=0) + 1
    xx, yy = np.meshgrid(np.arange(x_min[0], x_max[0], h[0]),
                         np.arange(x_min[1], x_max[1], h[1]))
    Z = kmeans_predict(clusters, np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    pylab.set_cmap(pylab.cm.Paired)
    pylab.axes(axes)
    pylab.contourf(xx, yy, Z, cmap=pylab.cm.Paired)
    #pylab.axis('off')

    # Plot also the training points
    if Y is not None:
        pylab.scatter(X[:,0], X[:,1], c=Y)
    else:
        pylab.scatter(X[:,0], X[:,1])
    pylab.scatter(clusters[:,0], clusters[:,1], s=200, marker='x', color='white')

plotWindow = plot(show_toolbar=True)
fig, axes = plotWindow.get_figure_and_axes(0)
#plot_decision_surface(axes, clf, data[:, [svm_indices[0], svm_indices[1]]], labels)
plot_decision_surface(axes, pca_clusters, pca_features, partition)
plotWindow.show()
