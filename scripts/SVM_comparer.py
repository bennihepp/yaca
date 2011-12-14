import scipy.stats
import matplotlib
import matplotlib.pylab as pylab

from sklearn import svm

from . import print_engine

featureIds = pl.featureIds
for i,fId in enumerate(featureIds):
    for k,v in pdc.objFeatureIds.iteritems():
        if v == fId:
            print '%d: %s (%d)' % (i, k, fId)
            break

groups = grouping.get_groups('treatment', pdc, pl.validCellMask)
names, masks = zip(*groups)

d_matrix = np.empty((len(groups), len(groups)))
for index1,(name1,mask1) in enumerate(groups):
    print 'processing group %s...' % name1
    #tr1_mask = pl.mask_and(pl.get_treatment_cell_mask(tr1.index), pl.get_valid_cell_mask())
    #nan_mask = np.isnan(pdc.objFeatures[mask1][:, featureIds])
    #nan_mask = np.any(nan_mask, axis=1)
    #tmp_mask = mask1[mask1]
    #tmp_mask[nan_mask] = False
    #mask1[mask1] = tmp_mask
    data1 = pdc.objFeatures[mask1][:, featureIds]
    labels1 = np.ones((data1.shape[0],))
    indices = np.arange(0, data1.shape[0])
    np.random.shuffle(indices)
    X1 = data1[indices[:1000]]
    for index2,(name2,mask2) in enumerate(groups):
        #tr2_mask = pl.mask_and(pl.get_treatment_cell_mask(tr2.index), pl.get_valid_cell_mask())
        if index1 != index2:
            assert np.sum(np.logical_and(mask1, mask2)) == 0
        #nan_mask = np.isnan(pdc.objFeatures[mask2][:, featureIds])
        #nan_mask = np.any(nan_mask, axis=1)
        #tmp_mask = mask1[mask2]
        #tmp_mask[nan_mask] = False
        #mask1[mask2] = tmp_mask
        data2 = pdc.objFeatures[mask2][:, featureIds]
        data = np.vstack([data1, data2])
        labels2 = 2 * np.ones((data2.shape[0],))
        labels = np.hstack([labels1, labels2])
        indices = np.arange(0, data2.shape[0])
        np.random.shuffle(indices)
        X2 = data2[indices[:1000]]
        X = np.vstack([X1, X2])
        Y1 = labels1[:X1.shape[0]]
        Y2 = labels2[:X2.shape[0]]
        Y = np.hstack([Y1, Y2])

        # Compare to the weights of an SVM
        clf = svm.SVC(kernel='linear')
        clf.fit(X, Y)
        score = clf.score(data, labels)
        print 'SVM error:', score
        #pred = clf.predict(data)
        #match = np.sum(pred == labels)
        #print match, labels.shape[0]
        #print match / float(labels.shape[0])

        #svm_weights = (clf.coef_**2).sum(axis=0)
        #svm_weights /= svm_weights.max()

        d_matrix[index1, index2] = score

#d_matrix = np.array([d_matrix, d_matrix.T])
#d_matrix = np.max(d_matrix, axis=0)
print d_matrix

plotWindow = plot(show_toolbar=True)
#plotWindow.set_caption(0, 'Feature weights for %s and %s' % (tr1.name, tr2.name))
fig, axes = plotWindow.get_figure_and_axes(0)
#x_indices = np.arange(X.shape[-1])
#b1 = axes.bar(x_indices-.45, scores, width=.3, color='g')
#b2 = axes.bar(x_indices-.15, svm_weights, width=.3, color='r')
#fig.legend((b1, b2), ('Univariate score ($-Log(p_{value})$)', 'SVM weight'))
#print x_indices.shape
#print scores.shape
#print scores
#axes.grid(True)
axes.set_xticks(range(len(names)))
axes.set_yticks(range(len(names)))
axes.set_xticklabels(names)
axes.set_yticklabels(names)
aximg = axes.imshow(d_matrix, interpolation='nearest')
fig.colorbar(aximg)
plotWindow.show()

def plot_decision_surface(axes, clf, X, Y):
    import numpy as np
    h=.02 # step size in the mesh
    # create a mesh to plot in
    x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
    y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    axes.set_cmap(pylab.cm.Paired)
    pylab.axes(axes)
    pylab.contourf(xx, yy, Z)
    pylab.axis('off')

    # Plot also the training points
    pylab.scatter(X[:,0], X[:,1], c=Y)

plotWindow = plot(show_toolbar=True)
fig, axes = plotWindow.get_figure_and_axes(0)
plot_decision_surface(axes, svm, data, labels)
plotWindow.show()

cdm = np.array([d_matrix, d_matrix.T])
cdm = np.max(cdm, axis=0)
cdm = cdm - 0.5
cdm[np.identity(cdm.shape[0], dtype=bool)] = 0.0
plotWindow = plot(show_toolbar=True)
fig, axes = plotWindow.get_figure_and_axes(0)
import hcluster
cdm = hcluster.squareform(cdm)
Z = hcluster.linkage(cdm, 'average')
print_engine.draw_dendrogram(axes, Z, labels=names)
plotWindow.show()
