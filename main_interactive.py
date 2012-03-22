import sys, os
import time
import numpy

from src.core import pipeline
from src.core import importer
from src.core import analyse
from src.core import headless_cluster_configuration
from src.core import headless_channel_configuration

import src.main_gui
from src.gui.gallery_window import GalleryWindow
from src.gui.gui_utils import CellPixmapFactory,CellFeatureTextFactory

from src.core import parameter_utils as utils

import hcluster as hc



skip_next = 0


def print_help():

    sys.stderr.write("""Usage: python %s [options]
Necessary options:
  --project-file <filename>     Load specified pipeline file
""" % sys.argv[0])


if len(sys.argv) > 1:
    for i in xrange(1, len(sys.argv)):

        arg = sys.argv[i]

        if skip_next > 0:
            skip_next -= 1
            continue

        if arg == '--project-file':
            project_file = sys.argv[i+1]
            skip_next = 1
        elif arg == '--help':
            print_help()
            sys.exit(0)
        else:
            sys.stderr.write('Unknown option: %s\n' % arg)
            print_help()
            sys.exit(-1)


#from PyQt4.QtCore import *
#from PyQt4.QtGui import *

#from gui.main_window import MainWindow


#g = gui_window.GUI(pdc, objFeatures, mahalPoints, mahalFeatureNames, channelDescription, partition, sorting, inverse_sorting, clusters, cluster_count, sys.argv)

yaca_importer = importer.Importer()

headlessClusterConfiguration = headless_cluster_configuration.HeadlessClusterConfiguration()
headlessChannelConfiguration = headless_channel_configuration.HeadlessChannelConfiguration()

print 'Loading project file...'
utils.load_module_configuration(project_file)
print 'Project file loaded'


modules = utils.list_modules()
for module in modules:

    if not utils.all_parameters_set(module):

        print 'Not all required parameters for module %s have been set' % module
        print 'Unable to start pipeline'
        sys.exit(1)

    elif not utils.all_requirements_met(module):

        print 'Not all requirements for module %s have been fulfilled' % module
        print 'Unable to start pipeline'
        sys.exit(1)

pdc = yaca_importer.get_pdc()
clusterConfiguration = headlessClusterConfiguration.clusterConfiguration

pl = pipeline.Pipeline(pdc, clusterConfiguration)
pl.run_quality_control()

pl.run_pre_filtering(analyse.FILTER_MODE_xMEDIAN)


import mixture
import matplotlib
from matplotlib import pyplot as plt


treatmentId = pdc.treatmentByName['S24A[40]']

mask1 = pl.mask_and(
    pl.get_control_treatment_cell_mask(),
    pl.get_valid_cell_mask(),
    pl.get_non_control_cell_mask()
)
mask2 = pl.mask_and(
    pl.get_treatment_cell_mask(treatmentId),
    pl.get_valid_cell_mask(),
    pl.get_non_control_cell_mask()
)
mask = pl.mask_or(mask1, mask2)

features = pdc.objFeatures

print 'Feature set: %s' % str(clusterConfiguration[1][0])
featureIds = []
featureNames = clusterConfiguration[1][1]
for featureName in featureNames:
    featureIds.append(pdc.objFeatureIds[featureName])
featureIds = numpy.array(featureIds)

features = features[: , featureIds]

from src.core.batch import utils as batch_utils

BOOTSTRAP_COUNT = 300
RESAMPLE_MAX = 3
bootstrap_dist = numpy.empty((BOOTSTRAP_COUNT, features.shape[1]))

dist = numpy.empty((BOOTSTRAP_COUNT, features.shape[1]))

obs1 = pdc.objFeatures[mask1][: , featureIds]
edf1 = batch_utils.compute_edf(obs1)
obs3 = pdc.objFeatures[mask2][: , featureIds]
edf3 = batch_utils.compute_edf(obs3)
obs2 = numpy.empty(obs1.shape)
for i in xrange(BOOTSTRAP_COUNT):
    #sample_ids = numpy.random.randint(0, obs1.shape[0], obs1.shape[0])
    #obs2 = obs1[sample_ids]
    resample_counts = numpy.zeros((obs1.shape[0],))
    for j in xrange(obs2.shape[0]):
        l = numpy.random.randint(0, obs1.shape[0])
        while resample_counts[l] >= RESAMPLE_MAX:
            l = numpy.random.randint(0, obs1.shape[0])
        resample_counts[l] += 1
        obs2[j] = obs1[l]
    edf2 = batch_utils.compute_edf(obs2)
    for k in xrange(features.shape[1]):
        support1 = edf1[k]
        support2 = edf2[k]
        support3 = edf3[k]
        bootstrap_dist[i, k] = batch_utils.compute_edf_distance(support1, support2)
        dist[i, k] = batch_utils.compute_edf_distance(support2, support3)

"""masks = [mask1, mask2]
edfs = batch_utils.compute_edfs(pl, masks, features)
#dist_matrix = batch_utils.compute_edf_distances(edfs)
dist = numpy.empty((features.shape[1],))
for k in xrange(features.shape[1]):
    support1 = edfs[0][k]
    support2 = edfs[1][k]
    dist[k] = batch_utils.compute_edf_distance(support1, support2)"""

max_dist = numpy.max(dist, axis=0)
mean_dist = numpy.mean(dist, axis=0)
median_dist = numpy.median(dist, axis=0)

max_bootstrap_dist = numpy.max(bootstrap_dist, axis=0)
mean_bootstrap_dist = numpy.mean(bootstrap_dist, axis=0)
median_bootstrap_dist = numpy.median(bootstrap_dist, axis=0)

feature_mask = median_dist - median_bootstrap_dist  > 1e-8
#masked_ids = numpy.arange(feature_mask.shape[0])[feature_mask]
masked_ids = featureIds[feature_mask]
print 'informative features:'
print masked_ids
for feature_id in masked_ids:
    fname = None
    for fn,fid in pdc.objFeatureIds.iteritems():
        if fid == feature_id:
            fname = fn
            break
    print 'feature (%d): %s' % (feature_id, fname)

def select_best_features(feature_quality, feature_mask, num_of_features=3):
    if num_of_features <= 0:
        best_feature_mask = feature_mask
    else:
        sorted_feature_indices = numpy.argsort(feature_quality)
        #best_feature_mask = median_dist - median_bootstrap_dist  > 1e-8
        best_feature_mask = feature_mask.copy()
        best_feature_mask[sorted_feature_indices[: -num_of_features]] = False
    return best_feature_mask

feature_quality = median_dist - median_bootstrap_dist
best_feature_mask = select_best_features(feature_quality, feature_mask, 2)
best_feature_ids = featureIds[best_feature_mask]
print 'best informative features:'
print best_feature_ids
for feature_id in best_feature_ids:
    fname = None
    for fn,fid in pdc.objFeatureIds.iteritems():
        if fid == feature_id:
            fname = fn
            break
    print 'feature (%d): %s' % (feature_id, fname)


xlabels = []
for fid in featureIds:
    xlabels.append(str(fid))
left = 2 * numpy.arange(obs1.shape[1])
plt.bar(left, median_bootstrap_dist, facecolor='red', align='center')
left = left + 1.0
plt.bar(left, median_dist, facecolor='green', align='center')
t = 2 * numpy.arange(obs1.shape[1]) + 0.5
plt.axes().set_xticks(t)
plt.axes().set_xticklabels(xlabels)
plt.show()

best_features = features[: , best_feature_mask]
#features = features[: 1000]
#features = features[: , : 3]
mu1 = numpy.mean(best_features[mask1], axis=0)
mu2 = numpy.mean(best_features[mask2], axis=0)
sigma1 = numpy.cov(best_features[mask1], rowvar=0)
sigma2 = numpy.cov(best_features[mask2], rowvar=0)
p1 = numpy.sum(mask1) / float(numpy.sum(mask))
p2 = 1.0 - p1
#best_features = best_features[mask]


from scikits.learn.mixture import GMM

NUMBER_OF_COMPONENTS = 3
COVARIANCE_TYPE = 'diag'

gmm = GMM(NUMBER_OF_COMPONENTS, COVARIANCE_TYPE)

gmm.fit(best_features[mask], 40)

print gmm.weights

print gmm.means

print gmm.covars

estimated_labels = gmm.predict(best_features[mask])


def plot_mixture_model(clf, observations, estimated_labels, true_labels):
    #colors = 'rym'
    #for i,sub_observations in enumerate(observations):
    #    sub_observations = observations[i]
    #    color = colors[i]
    #    plt.plot(sub_observations[:, 0], sub_observations[:, 1], '%s.' % color, alpha=0.5)
    colors = 'gbyrcm'
    import itertools
    color_iter = itertools.cycle(colors)
    for i, (label,color) in enumerate(zip(xrange(numpy.min(estimated_labels), numpy.max(estimated_labels) + 1), color_iter)):
        #color = colors[i]
        label_mask = (estimated_labels == label)
        sub_observations = observations[label_mask]
        plt.plot(sub_observations[:, 0], sub_observations[:, 1], '%s.' % color, alpha=0.5)
    colors = ['green', 'blue', 'yellow', 'red', 'cyan', 'magenta']
    color_iter = itertools.cycle(colors)
    for i,(mu, sigma, color) in enumerate(zip(clf.means, clf.covars, color_iter)):
        #color = colors[i]
        #mu = component[0].mu
        #sigma = component[0].sigma
        print 'sigma:', sigma
        #eigvalues,eigvector = numpy.linalg.eig(sigma)
        #norm_acos = eigvector[0, 0] / (eigvector[0, 0]**2 + eigvector[0, 1]**2)
        #angle = numpy.math.acos(norm_acos)
        #angle_degrees = numpy.math.degrees(angle)
        #print 'angle:', angle_degrees
        #ell = matplotlib.patches.Ellipse(mu[:2], eigvalues[0], eigvalues[1], angle_degrees, color=color, alpha=0.3)
        #plt.axes().add_artist(ell)
        v,w = numpy.linalg.eigh(sigma)
        u = w[0] / numpy.linalg.norm(w[0])
        angle = numpy.arctan(u[1] / u[0])
        angle_degrees = numpy.math.degrees(angle)
        print 'angle:', angle_degrees
        v *= 9
        print 'v:', v
        ell = matplotlib.patches.Ellipse(mu, v[0], v[1], 180.0 + angle_degrees, color=color)
        ell.set_clip_box(plt.axes().bbox)
        ell.set_alpha(0.5)
        ell.set(zorder=100)
        plt.axes().add_artist(ell)
    plt.show()

plot_mixture_model(gmm, best_features[mask], estimated_labels, None)



"""
def plot_mixture_model(components, observations, estimated_labels, true_labels):
    #colors = 'rym'
    #for i,sub_observations in enumerate(observations):
    #    sub_observations = observations[i]
    #    color = colors[i]
    #    plt.plot(sub_observations[:, 0], sub_observations[:, 1], '%s.' % color, alpha=0.5)
    colors = 'gbyrcm'
    for i, label in enumerate(xrange(numpy.min(estimated_labels), numpy.max(estimated_labels) + 1)):
        color = colors[i]
        label_mask = (estimated_labels == label)
        sub_observations = observations[label_mask]
        plt.plot(sub_observations[:, 0], sub_observations[:, 1], '%s.' % color, alpha=0.5)
    colors = ['green', 'blue', 'yellow', 'red', 'cyan', 'magenta']
    for i,component in enumerate(components):
        color = colors[i]
        mu = component[0].mu
        sigma = component[0].sigma
        print 'sigma:', sigma
        #eigvalues,eigvector = numpy.linalg.eig(sigma)
        #norm_acos = eigvector[0, 0] / (eigvector[0, 0]**2 + eigvector[0, 1]**2)
        #angle = numpy.math.acos(norm_acos)
        #angle_degrees = numpy.math.degrees(angle)
        #print 'angle:', angle_degrees
        #ell = matplotlib.patches.Ellipse(mu[:2], eigvalues[0], eigvalues[1], angle_degrees, color=color, alpha=0.3)
        #plt.axes().add_artist(ell)
        v,w = numpy.linalg.eigh(sigma)
        u = w[0] / numpy.linalg.norm(w[0])
        angle = numpy.arctan(u[1] / u[0])
        angle_degrees = numpy.math.degrees(angle)
        print 'angle:', angle_degrees
        v *= 9
        print 'v:', v
        ell = matplotlib.patches.Ellipse(mu, v[0], v[1], 180.0 + angle_degrees, color=color)
        ell.set_clip_box(plt.axes().bbox)
        ell.set_alpha(0.5)
        ell.set(zorder=100)
        plt.axes().add_artist(ell)
    plt.show()


NUMBER_OF_COMPONENTS = 3

data = mixture.DataSet()
data.fromArray(best_features[mask])
sigma = numpy.cov(best_features[mask], rowvar=0) / NUMBER_OF_COMPONENTS
data_min, data_max = numpy.min(best_features[mask], axis=0), numpy.max(best_features[mask], axis=0)
components = []
weights = []
for n in xrange(NUMBER_OF_COMPONENTS):
    mu = data_min + (data_max - data_min) * n
    component = mixture.MultiNormalDistribution(best_features.shape[1], mu, sigma)
    components.append(component)
    weights.append(1.0 / NUMBER_OF_COMPONENTS)
##mu = numpy.ones((best_features.shape[1],))
#n1 = mixture.MultiNormalDistribution(best_features.shape[1], mu1, sigma1)
##mu = -numpy.ones((best_features.shape[1],))
##sigma = numpy.identity(best_features.shape[1])
#n2 = mixture.MultiNormalDistribution(best_features.shape[1], mu2, sigma2)
#m = mixture.MixtureModel(2, [p1, p2], [n1, n2])
m = mixture.MixtureModel(NUMBER_OF_COMPONENTS, weights, components)

data.internalInit(m)

m.modelInitialization(data)
m.EM(data, 15, 0.1)

estimated_cl = m.classify(data)
true_cl = -numpy.ones(estimated_cl.shape)
true_cl[mask1[mask]] = 0
true_cl[mask2[mask]] = 1

accuracy1 = numpy.sum(true_cl == estimated_cl) / float(true_cl.shape[0])
accuracy2 = numpy.sum((1.0 - true_cl) == estimated_cl) / float(true_cl.shape[0])
print 'accuracy1:', accuracy1
print 'accuracy2:', accuracy2

plot_mixture_model(m.components, best_features[mask], estimated_cl, true_cl)
plot_mixture_model(m.components, best_features[mask], true_cl, estimated_cl)

channelDescription, channelMapping = headlessChannelConfiguration.getChannelMappingAndDescription()
featureDescription = dict(pdc.objFeatureIds)
gallery_window = GalleryWindow(pl, featureDescription, channelMapping, channelDescription, singleImage=False)
focusObjId = -1
featureFactory = CellFeatureTextFactory(pdc)
pixmapFactory = CellPixmapFactory(pdc, channelMapping)
featureDescription = dict(pdc.objFeatureIds)
gallery_window.update_feature_description(featureDescription)

caption = 'Showing control cells'
gallery_window.update_caption(caption)
selectionIds = pdc.objFeatures[: , pdc.objObjectFeatureId][mask1]
gallery_window.on_selection_changed(focusObjId, selectionIds, pixmapFactory, featureFactory)
gallery_window.show()

caption = 'Showing treated cells'
gallery_window.update_caption(caption)
selectionIds = pdc.objFeatures[: , pdc.objObjectFeatureId][mask2]
gallery_window.on_selection_changed(focusObjId, selectionIds, pixmapFactory, featureFactory)
gallery_window.show()

for i, label in enumerate(xrange(numpy.min(estimated_cl), numpy.max(estimated_cl) + 1)):
    estimated_mask = estimated_cl == label
    caption = 'Showing cells for estimated label %d' % label
    selectionIds = pdc.objFeatures[: , pdc.objObjectFeatureId][mask][estimated_mask]
    gallery_window.on_selection_changed(focusObjId, selectionIds, pixmapFactory, featureFactory)
    gallery_window.update_caption(caption)
    gallery_window.show()

"""
