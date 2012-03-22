import scipy.stats
import matplotlib

from sklearn import svm

from . import print_engine

featureIds = pl.featureIds
for i,fId in enumerate(featureIds):
    for k,v in pdc.objFeatureIds.iteritems():
        if v == fId:
            print '%d: %s (%d)' % (i, k, fId)
            break

treatment1 = 'CB1_a'
replicate1 = 0
treatment2 = 'CA_b'
replicate2 = 1

mask1 = pl.mask_and(pl.get_valid_cell_mask(), pl.get_treatment_cell_mask(pdc.treatmentByName[treatment1]))
mask2 = pl.mask_and(pl.get_valid_cell_mask(), pl.get_treatment_cell_mask(pdc.treatmentByName[treatment2]))
if replicate1 >= 0:
    mask1 = pl.mask_and(mask1, pl.get_replicate_cell_mask(replicate1))
if replicate2 >= 0:
    mask2 = pl.mask_and(mask2, pl.get_replicate_cell_mask(replicate2))

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

features = pdc.objFeatures[pl.mask_or(mask1, mask2)][:, featureIds]
#pca = fit_pca(2, features)
#features1 = pca.transform(pdc.objFeatures[mask1][:, featureIds])
#features2 = pca.transform(pdc.objFeatures[mask2][:, featureIds])
features1 = pdc.objFeatures[mask1][:, featureIds]
features2 = pdc.objFeatures[mask2][:, featureIds]

N_min = np.min([features1.shape[0], features2.shape[0]])

data1 = features1[:N_min]
labels1 = np.ones((data1.shape[0],))
indices = np.arange(0, data1.shape[0])
np.random.shuffle(indices)
X1 = data1[indices[:1000]]

assert np.sum(np.logical_and(mask1, mask2)) == 0

data2 = features2[:N_min]
data = np.vstack([data1, data2])
labels2 = 2 * np.ones((data2.shape[0],))
labels = np.hstack([labels1, labels2])
indices = np.arange(0, data2.shape[0])
np.random.shuffle(indices)
X2 = data2[indices[:1000]]

Y1 = labels1[:X1.shape[0]]
Y2 = labels2[:X2.shape[0]]

print 'data1:', data1.shape[0]
print 'data2:', data2.shape[0]

X = np.vstack([X1, X2])
Y = np.hstack([Y1, Y2])

# Train a SVM
clf = svm.SVC(kernel='linear')
clf.fit(X, Y)
score = clf.score(data, labels)
print 'SVM error:', score
#if data1.shape[0] > data2.shape[0]:
#    alpha = data2.shape[0] / float(data1.shape[0])
#else:
#    alpha = data1.shape[0] / float(data2.shape[0])
#print 'Score:', (score - alpha) / (1 - alpha)

svm_weights = (clf.coef_**2).sum(axis=0)
svm_weights /= svm_weights.max()
svm_indices = np.argsort(svm_weights)[::-1]
#print svm_indices

#plotWindow = plot(show_toolbar=True)
##plotWindow.set_caption(0, 'Feature weights for %s and %s' % (tr1.name, tr2.name))
#fig, axes = plotWindow.get_figure_and_axes(0)
#x_indices = np.arange(X.shape[-1])
#b2 = axes.bar(x_indices, svm_weights, width=.8, color='r', align='center')
#axes.set_xlim(-1, x_indices[-1]+1)
#plotWindow.show()

def plot_decision_surface(axes, clf, X, Y):
    import matplotlib.pylab as pylab
    import numpy as np
    # step size in the mesh
    h = (np.max(X, axis=0) - np.min(X, axis=0)) / 100.0
    # create a mesh to plot in
    x_min = np.min(X, axis=0) - 1
    x_max = np.max(X, axis=0) + 1
    xx, yy = np.meshgrid(np.arange(x_min[0], x_max[0], h[0]),
                         np.arange(x_min[1], x_max[1], h[1]))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    pylab.set_cmap(pylab.cm.Paired)
    pylab.axes(axes)
    pylab.contourf(xx, yy, Z, cmap=pylab.cm.Paired)
    #pylab.axis('off')

    # Plot also the training points
    pylab.scatter(X[:,0], X[:,1], c=Y)

plotWindow = plot(show_toolbar=True)
fig, axes = plotWindow.get_figure_and_axes(0)
#plot_decision_surface(axes, clf, data[:, [svm_indices[0], svm_indices[1]]], labels)
#plot_decision_surface(axes, clf, data, labels)
#plotWindow.show()
