# -*- coding: utf-8 -*-

"""
cluster_profiles.py -- Computation and manipulation of cluster profiles.
"""

# This software is distributed under the FreeBSD License.
# See the accompanying file LICENSE for details.
# 
# Copyright 2011 Benjamin Hepp

import sys

import numpy

import distance

def compute_cluster_profiles(masks, points, clusters,
                             exp_factor=10.0, minkowski_p=2):
    if exp_factor < 0.0:
        return __compute_cluster_profiles(
                    masks, points, clusters, minkowski_p)
    else:
        clusterProfiles = numpy.zeros((len(masks), clusters.shape[0]))
        for i in xrange(len(masks)):
            sys.stdout.write('\rcomputing profile {} out of {}'.format(
                i + 1, len(masks)))
            sys.stdout.flush()
            mask = masks[i]
            p = points[mask]
            clusterProfiles[i] = __compute_cluster_profile(
                    p, clusters, exp_factor, minkowski_p)
        sys.stdout.write('\n')
        sys.stdout.flush()
        return clusterProfiles

# New variant as documented in the thesis
def __compute_cluster_profiles(masks, points, clusters, minkowski_p):
    cluster_profiles = numpy.zeros((len(masks), clusters.shape[0]))
    # calculate the distance of all pairs of cluster centroids ...
    cluster_dist_m = distance.minkowski_cdist(
            clusters, clusters, minkowski_p)
    # ... and find the minimum
    min_cluster_dist = numpy.min(cluster_dist_m[numpy.invert(
            numpy.identity(cluster_dist_m.shape[0], dtype=bool)
   )])
    # Calculate the distance of all the samples to the k-th cluster
    # centroid. Rows represent samples, columns represent clusters.
    dist_m = distance.minkowski_cdist(points, clusters, minkowski_p)
    # Compute cluster profile for each point and normalize it.
    profiles = numpy.exp(-dist_m/min_cluster_dist)
    profiles = profiles / numpy.sum(profiles, axis=1)[:,numpy.newaxis]
    # Compute the mean cluster profile for each set of points.
    for i in xrange(len(masks)):
        mask = masks[i]
        p = points[mask]
        cluster_profiles[i] = numpy.sum(profiles[mask], axis=0) / numpy.sum(mask)
    return cluster_profiles

def __compute_cluster_profile(points, clusters, exp_factor, minkowski_p=2):

    if points.shape[0] == 0:
        return numpy.zeros((clusters.shape[0],))

    # calculate the distance of all the samples to the k-th cluster centroid.
    # rows represent clusters, columns represent samples
    dist_m = distance.minkowski_cdist(clusters, points, minkowski_p)

    #print 'dist_m:', dist_m[:, 0]

    if exp_factor < 0:
        argmin = numpy.argmin(dist_m, axis=0)
        profile = numpy.zeros((clusters.shape[0],))
        for i in xrange(clusters.shape[0]):
            profile[i] += numpy.sum(argmin == i)
        return profile


    sim_m = 1.0 / dist_m

    #print 'sim_m:', sim_m[: , 0]

    mask = numpy.isinf(sim_m)
    col_mask = numpy.any(mask, axis=0)

    sim_m[:, col_mask] = 0.0
    sim_m[mask] = 1.0

    #print 'sum(sim_m):', numpy.sum(sim_m)

    if exp_factor > 0:

        sim_m = sim_m / numpy.max(sim_m)

        sim_m = (numpy.exp(exp_factor * sim_m) - 1) / (numpy.exp(exp_factor) - 1)

    # normalize the similarity matrix
    sim_m = sim_m / numpy.sum(sim_m, axis=0)

    #print 'sim_m:', sim_m[: , 0]

    #mask = numpy.isnan(sim_m, axis=)
    #sim_m[mask] = 1.0

    profile = numpy.sum(sim_m, axis=1)

    #print 'sum(profile):', numpy.sum(profile), 'sum(sim_m):', numpy.sum(sim_m), 'points.shape[0]:', points.shape[0]

    return profile


def compute_cluster_similarity_matrix(clusters, minkowski_p=2.0):

    #dist_m = distance.minkowski_cdist(clusters, clusters, minkowski_p)
    #sim_m = numpy.exp(-dist_m)
    #return sim_m
    """max_dist = numpy.max(dist)
    if max_dist > 0.0:
        sim = 1.0 - dist / max_dist
    else:
        sim = 1.0 - dist
    return sim"""
    """sim = 1.0 / dist
    finite_mask = numpy.isfinite(sim)
    max_sim = numpy.max(sim[finite_mask])
    sim /= max_sim
    sim[numpy.invert(finite_mask)] = 1.0
    return sim"""
    cluster_dist_m = distance.minkowski_cdist(clusters, clusters, minkowski_p)
    min_cluster_dist = numpy.min(cluster_dist_m[
            numpy.invert(numpy.identity(cluster_dist_m.shape[0], dtype=bool))
   ])
    sim = numpy.exp(-cluster_dist_m/min_cluster_dist)
    return sim

def compute_treatment_similarity_map(distanceMap, profile_metric):

    if profile_metric == 'summed_minmax':
        return 1.0 - distanceMap
    elif profile_metric == 'l2_norm' or profile_metric == 'chi2_norm' or profile_metric == 'l1_norm' or profile_metric == 'chi1_norm' or profile_metric == 'quadratic_chi':
        masked_map = distanceMap[numpy.isfinite(distanceMap)]
        if masked_map.shape[0] == 0:
            return distanceMap
        else:
            max_v = numpy.max(masked_map)
            #print 'max_v:', max_v
            #print 'distanceMap/max_v:', (distanceMap / max_v)
            if max_v > 0.0:
                return 1.0 - distanceMap / max_v
            else:
                return 1.0 - distanceMap
    else:
        raise Exception('No such profile metric: %s' % profile_metric)


def compute_treatment_distance_map(clusterProfiles, profile_metric, profile_threshold=0.0, binSimilarityMatrix=None, normalizationFactor=0.5):

    profileHeatmap = numpy.zeros((clusterProfiles.shape[0], clusterProfiles.shape[0]))
    profileHeatmap[numpy.identity(profileHeatmap.shape[0], dtype=bool)] = 0.0

    for i in xrange(clusterProfiles.shape[0]):

        profile1 = clusterProfiles[i]
        norm_profile1 = profile1 / float(numpy.sum(profile1))

        sys.stdout.write('\rprofile #{} out of {}'.format(i + 1, clusterProfiles.shape[0]))
        sys.stdout.flush()
        #print 'norm_profile1:', norm_profile1

        if profile_threshold > 0.0:
            max = numpy.max(profile1)
            threshold_mask = profile1 < max * profile_threshold
            norm_profile1[threshold_mask] = 0.0
            norm_profile1 = norm_profile1 / float(numpy.sum(norm_profile1))

        for j in xrange(i):

            profile2 = clusterProfiles[j]
            norm_profile2 = profile2 / float(numpy.sum(profile2))

            if profile_threshold > 0.0:
                max = numpy.max(profile2)
                threshold_mask = profile2 < max * profile_threshold
                norm_profile2[threshold_mask] = 0.0
                norm_profile2 = norm_profile2 / float(numpy.sum(norm_profile2))

            if profile_metric == 'summed_minmax':

                min_match = numpy.sum(numpy.min([norm_profile1, norm_profile2], axis=0)**2)
                max_match = numpy.sum(numpy.max([norm_profile1, norm_profile2], axis=0)**2)

                match = min_match / max_match

                dist = 1.0 - match

            elif profile_metric == 'l2_norm':

                # L2-norm
                dist = numpy.sqrt(numpy.sum((norm_profile1 - norm_profile2)**2))

            elif profile_metric == 'l1_norm':

                # L1-norm
                dist = numpy.sum(numpy.abs(norm_profile1 - norm_profile2))

            elif profile_metric == 'chi1_norm':

                # Chi1
                dist =  numpy.abs(norm_profile1 - norm_profile2) / (norm_profile1 + norm_profile2)
                dist[numpy.logical_and(norm_profile1 == 0, norm_profile2 == 0)] = 0.0

                dist = numpy.sum(dist)

            elif profile_metric == 'chi2_norm':

                # chi-square
                dist =  (norm_profile1 - norm_profile2) ** 2 / (norm_profile1 + norm_profile2)
                dist[numpy.logical_and(norm_profile1 == 0, norm_profile2 == 0)] = 0.0

                dist = numpy.sum(dist)

            elif profile_metric == 'quadratic_chi':
            
                if binSimilarityMatrix is None:
                    binSimilarityMatrix = numpy.identity(
                        clusterProfiles.shape[1])

                # quadratic-chi
                A = binSimilarityMatrix
                #A = numpy.identity(norm_profile1.shape[0])
                m = normalizationFactor

                P = norm_profile1
                Q = norm_profile2

                """
                This variant is adapted from the MatLab sources on
                http://www.cs.huji.ac.il/~ofirpele/QC/code/

                Z = numpy.dot((P+Q), A)
                Z[Z==0] = 1.0
                Z = Z ** m
                D = (P-Q) / Z
                #E = D*A*D
                E = numpy.dot(numpy.dot(D, A), D)
                dist = numpy.max([E, numpy.zeros(E.shape)], axis=0)"""

                D = numpy.empty((P.shape[0],))
                for k in xrange(P.shape[0]):
                    D[k] = 0.0
                    for c in xrange(P.shape[0]):
                        D[k] += ((P[c] + Q[c]) * A[c,k]) ** m
                N = numpy.empty((P.shape[0],))
                for k in xrange(P.shape[0]):
                    N[k] = P[k] - Q[k]
                zero_mask = D == 0.0
                D[zero_mask] = 1.0
                N[zero_mask] = 0.0
                M = 0.0
                for k in xrange(N.shape[0]):
                    for l in xrange(N.shape[0]):
                        M += (N[k] / D[k]) * (N[l] / D[l]) * A[k,l]
                dist = M

            elif profile_metric == 'summed_difference':

                P = norm_profile1
                Q = norm_profile2
                D = (P - Q)
                dist = numpy.sum(numpy.abs(D))

            else:
                raise Exception('No such profile metric: %s' % profile_metric)

            profileHeatmap[i, j] = dist
            profileHeatmap[j, i] = dist

    sys.stdout.write('\n')    
    sys.stdout.flush()
    return profileHeatmap

def compute_map_quality(map):
    diagonal_mask = numpy.identity(map.shape[0], dtype=bool)
    non_diagonal_mask = numpy.invert(diagonal_mask)
    diagonal_mean = numpy.mean(map[diagonal_mask])
    non_diagonal_mean = numpy.mean(map[non_diagonal_mask])
    quality = non_diagonal_mean - diagonal_mean
    return quality

#binSimilarityMatrix = batch_utils.compute_cluster_similarity_matrix(pl.nonControlClusters)
#distanceHeatmap = batch_utils.compute_treatment_distance_map(clusterProfiles, profile_metric, 0.0, binSimilarityMatrix, normalizationFactor)
#similarityHeatmap = batch_utils.compute_treatment_similarity_map(distanceHeatmap, profile_metric)

#if random_split_dist != None and random_split_dist.ndim == 0:
    #random_split_dist = None
#if replicate_split_dist != None and replicate_split_dist.ndim == 0:
    #replicate_split_dist = None

#heatmap = distanceHeatmap.copy()
#diagonal_mask = numpy.identity(heatmap.shape[0], dtype=bool)  
#if random_split_dist != None and random_split_dist.shape[0] == heatmap.shape[0]:
    #heatmap[diagonal_mask] = random_split_dist

#map_quality = compute_map_quality(heatmap)

#template_format_dict['group'] = group_name

#print 'Printing cluster profiles...'

#print 'heatmap:', heatmap.shape

#heatmap_kwargs = { 'lower' : True, 'diagonal' : True }
#pdf_filename = os.path.join(path_prefix, profile_file_template % template_format_dict) + '.pdf'
#print 'pdf_filename:', pdf_filename
#batch_utils.create_path(pdf_filename)
##sum_minmax_profileHeatmap, l2_chi2_norm_profileHeatmap, treatments = print_engine.print_cluster_profiles_and_heatmap(pl.pdc, clusterProfiles, pdf_filename, True, 0.0)
##print_engine.print_cluster_profiles_and_heatmap(labels, clusterProfiles, similarityHeatmap, pdf_filename, True, 0.0, similarity_map_clustering_method, xlabel='Treatment
#print_engine.print_cluster_profiles_and_heatmap(labels, clusterProfiles, heatmap, pdf_filename, True, 0.0, similarity_map_clustering_method, binSimilarityMatrix=binSimilarityMatrix, xlabel='Treatment', ylabel='Treatment', heatmap_kwargs=heatmap_kwargs, random_split_dist=random_split_dist, replicate_split_dist=replicate_split_dist, replicate_distance_threshold=replicate_distance_threshold, cluster_mask=cluster_mask, map_fontsize=pdf_map_fontsize, map_label_fontsize=pdf_map_label_fontsize, cluster_label_fontsize=pdf_cluster_label_fontsize)

#"""map = similarityHeatmap.copy()
#map[numpy.identity(map.shape[0], dtype=bool)] = random_split_match

#pdf_filename = os.path.join(path_prefix, profile_pdf_file_template % template_format_dict)
#batch_utils.create_path(pdf_filename)
#pdfDocument = print_engine.PdfDocument(pdf_filename)
#pdfDocument.next_page(2, 1)
#print_engine.draw_cluster_profiles(pdfDocument, labels, clusterProfiles)
#pdfDocument.next_page(1, 1)
#print_engine.draw_modified_treatment_similarity_map(pdfDocument.next_plot(), map, (0, 0), labels)

#print 'Clustering treatment similarity map...'
#cdm = 1.0 - similarityHeatmap
#cdm[numpy.identity(cdm.shape[0], dtype=bool)] = 0.0
#cdm = hc.squareform(cdm)
#Z = hc.linkage(cdm, similarity_map_clustering_method)
##print_engine.print_dendrogram(os.path.splitext(pdf_filename)[0] + '_dendrogram.pdf', Z, labels)

#pdfDocument.next_page()
#print_engine.draw_dendrogram(pdfDocument.next_plot(), Z, labels)
#pdfDocument.close()"""

#xls_title = 'Profile heatmap with method=%(method)s, index=%(index)d, param1=%(param1)d, param2=%(param2)d, param3=%(param3)d, param4=%(param4)d, exp_factor=%(exp_factor).1f' \
            #% template_format_dict
#xls_filename = os.path.join(path_prefix, profile_file_template % template_format_dict) + '.xls'
#batch_utils.create_path(xls_filename)
#batch_utils.write_profileHeatmapCSV(xls_title, labels, heatmap, xls_filename)

#pic_filename = os.path.join(path_prefix, profile_file_template % template_format_dict) + '.pic'
#batch_utils.create_path(pic_filename)
#f = open(pic_filename, 'w')
#import cPickle
#p = cPickle.Pickler(f)
#p.dump({ 'labels' : labels, 'masks' : masks, 'map_quality' : map_quality, 'clusterProfiles' : clusterProfiles, 'similarityHeatmap' : similarityHeatmap, 'distanceHeatmap' : distanceHeatmap, 'template_format_dict' : template_format_dict, 'heatmap' : heatmap })
#f.close()
