# -*- coding: utf-8 -*-

"""
ccluster_profiles_mp.pyx -- Fast multiprocessing computation and
                            manipulation of cluster profiles.
"""

# This software is distributed under the FreeBSD License.
# See the accompanying file LICENSE for details.
# 
# Copyright 2011 Benjamin Hepp

from __future__ import division

import sys
import math

import multiprocessing

import numpy

import ccluster_profiles_mp_worker

def compute_treatment_distance_map(
    clusterProfiles,
    profile_metric_index=0,
    profile_threshold=0.0,
    binSimilarityMatrix=None,
    normalizationFactor=0.5,
    num_of_processes=6):

    if binSimilarityMatrix is None:
        binSimilarityMatrix = numpy.identity(
            clusterProfiles.shape[1],
            dtype=numpy.float)

    print 'spawning processes...'
    chunk_size = int(math.ceil(clusterProfiles.shape[0] / float(num_of_processes)))
    manager = multiprocessing.Manager()
    #pool = multiprocessing.Pool(num_of_processes)
    #queue = manager.Queue()
    queue = multiprocessing.Queue()
    pool = []
    results = []
    for i in xrange(num_of_processes):
        start_index = i * chunk_size
        stop_index = min((i + 1) * chunk_size, clusterProfiles.shape[0])
        if not start_index < stop_index:
            continue
        #print 'start={}, stop={}'.format(start_index, stop_index)
        args = (queue, start_index, stop_index, clusterProfiles,
                profile_metric_index, profile_threshold,
                binSimilarityMatrix, normalizationFactor)
        #results.append((
            #pool.apply_async(compute_treatment_distance_map_worker, args=args),
            #start_index, stop_index))
        p = multiprocessing.Process(
            target=ccluster_profiles_mp_worker.\
                compute_treatment_distance_map_worker,
            args=args)
        pool.append(p)
        p.start()
    print 'waiting for processes...'
    profileHeatmap = numpy.empty(
        (clusterProfiles.shape[0], clusterProfiles.shape[0]))
    profileHeatmap[:,:] = numpy.nan
    i = 0
    j = 0
    #while j < clusterProfiles.shape[0]:
    while i < len(pool):
        try:
            id, load = queue.get()
        except IOError as e:
            # this is a dirty hack but I don't see another way to
            # use this with PyQt4
            if e.errno == 4:
                continue
            else:
                raise
        if id == ccluster_profiles_mp_worker.CCLUSTER_ID_CHUNK:
            v, start_index, stop_index = load
            #print 'start={}, stop={}, v={}'.format(start_index, stop_index, v)
            profileHeatmap[start_index:stop_index] = v
            #profileHeatmap[:, start_index:stop_index] = numpy.transpose(v)
            #sys.stdout.write('\rprocess {} out of {} finished'.format(
                #i + 1, num_of_processes))
            #sys.stdout.flush()
            i += 1
        elif id == ccluster_profiles_mp_worker.CCLUSTER_ID_TREATMENT:
            sys.stdout.write('\rprocessed {} out of {} treatments'.format(
                j + 1, clusterProfiles.shape[0]))
            sys.stdout.flush()
            j += 1
    tmp = numpy.tril(profileHeatmap)
    profileHeatmap = tmp + tmp.transpose()
    sys.stdout.write('\n')
    sys.stdout.flush()
    for p in pool:
        p.join()
    return profileHeatmap
