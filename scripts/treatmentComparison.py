import scipy.stats
import matplotlib

tr1 = pdc.treatments[pdc.treatmentByName['mock']]
tr2 = pdc.treatments[pdc.treatmentByName['sc_a']]
tr1_mask = pl.mask_and(pl.get_treatment_cell_mask(tr1.index), pl.get_valid_cell_mask(), pl.get_replicate_cell_mask(0))
tr2_mask = pl.mask_and(pl.get_treatment_cell_mask(tr2.index), pl.get_valid_cell_mask(), pl.get_replicate_cell_mask(1))
print np.sum(tr1_mask), np.sum(tr2_mask)

featureIds = pl.featureIds
for i,fId in enumerate(featureIds):
    for k,v in pdc.objFeatureIds.iteritems():
        if v == fId:
            print '%d: %s (%d)' % (i, k, fId)
            break

data1 = pdc.objFeatures[tr1_mask][:, featureIds]
data2 = pdc.objFeatures[tr2_mask][:, featureIds]
data = np.vstack([data1, data2])
labels1 = np.zeros((data1.shape[0],))
labels2 = np.ones((data2.shape[0],))
labels = np.hstack([labels1, labels2])
X1 = data1[:1000]
X2 = data2[-1000:]
X = np.vstack([X1, X2])
Y1 = labels1[:X1.shape[0]]
Y2 = labels2[:X2.shape[0]]
Y = np.hstack([Y1, Y2])

from sklearn.feature_selection import SelectFpr, f_classif

selector = SelectFpr(f_classif, alpha=0.1)
selector.fit(X, Y)
scores = -np.log10(selector._pvalues)
scores /= scores.max()

from sklearn import svm
# Compare to the weights of an SVM
clf = svm.SVC(kernel='linear')
clf.fit(X, Y)
print 'SVM error:', clf.score(data, labels)
pred = clf.predict(data)
match = numpy.sum(pred == labels)
print match, labels.shape[0]
print match / float(labels.shape[0])

svm_weights = (clf.coef_**2).sum(axis=0)
svm_weights /= svm_weights.max()


plotWindow = plot(show_toolbar=True)
plotWindow.set_caption(0, 'Feature weights for %s and %s' % (tr1.name, tr2.name))
fig, axes = plotWindow.get_figure_and_axes(0)
x_indices = np.arange(X.shape[-1])
b1 = axes.bar(x_indices-.45, scores, width=.3, color='g')
b2 = axes.bar(x_indices-.15, svm_weights, width=.3, color='r')
fig.legend((b1, b2), ('Univariate score ($-Log(p_{value})$)', 'SVM weight'))
print x_indices.shape
print scores.shape
print scores
plotWindow.show()
