import scipy.stats
import matplotlib

from sklearn import svm, metrics, cross_validation

from ..core import grouping
import print_engine

featureIds = pl.featureIds
for i,fId in enumerate(featureIds):
    for k,v in pdc.objFeatureIds.iteritems():
        if v == fId:
            print '%d: %s (%d)' % (i, k, fId)
            break

groups = grouping.get_groups('treatment,replicate', pdc, pl.validCellMask)
names, masks = zip(*groups)
joined_groups = grouping.join_groups(groups)
data = pdc.objFeatures[pl.validCellMask][joined_groups][:, featureIds]
labels = 2*pdc.objFeatures[pl.validCellMask][joined_groups][:, pdc.objTreatmentFeatureId] + 1*pdc.objFeatures[pl.validCellMask][joined_groups][:, pdc.objReplicateFeatureId]

svm = svm.SVC(kernel='linear')

bs = cross_validation.Bootstrap(data.shape[0], 2)
cms = cross_validation.cross_val_score(svm, data, labels, cv=bs, score_func=metrics.confusion_matrix)

for cm in cms:

    plotWindow = plot(show_toolbar=True)
    fig, axes = plotWindow.get_figure_and_axes(0)
    axes.set_xticks(range(len(names)))
    axes.set_yticks(range(len(names)))
    axes.set_xticklabels(names)
    axes.set_yticklabels(names)
    aximg = axes.imshow(cm, interpolation='nearest')
    fig.colorbar(aximg)
    plotWindow.show()
