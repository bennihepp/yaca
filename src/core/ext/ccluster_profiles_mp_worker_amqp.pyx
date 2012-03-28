# -*- coding: utf-8 -*-

"""
ccluster_profiles_mp_worker_amqp.pyx -- Fast multiprocessing computation and
                                       manipulation of cluster profiles using
                                       AMQP.
"""

# This software is distributed under the FreeBSD License.
# See the accompanying file LICENSE for details.
# 
# Copyright 2011 Benjamin Hepp

#from __future__ import division

import numpy
cimport numpy
cimport cython
import cPickle

import puka

DTYPE_INT = numpy.int
ctypedef numpy.int_t DTYPE_INT_t
DTYPE_FLOAT = numpy.float
ctypedef numpy.float_t DTYPE_FLOAT_t
DTYPE_BOOL = numpy.int8
ctypedef numpy.int8_t DTYPE_BOOL_t


def compute_treatment_distance_map_worker(amqp_url, work_queue,
                                          result_queue, pub_exchange,
                                          lock,
                                          num_of_profiles,
                                          num_of_clusters,
                                          shared_profileHeatmap,
                                          shared_clusterProfiles,
                                          profile_metric_index=0,
                                          profile_threshold=0.0,
                                          binSimilarityMatrix=None,
                                          normalizationFactor=0.5):

    #import wingdbstub

    profileHeatmap = numpy.frombuffer(shared_profileHeatmap)
    profileHeatmap = profileHeatmap.reshape((num_of_profiles, num_of_profiles))
    clusterProfiles = numpy.frombuffer(shared_clusterProfiles)
    clusterProfiles \
        = clusterProfiles.reshape((num_of_profiles, num_of_clusters))

    client = puka.Client(amqp_url)
    promise = client.connect()
    client.wait(promise)

    def on_work(promise, result):
        start_index, stop_index = cPickle.loads(result['body'])
        cy_compute_treatment_distance_map_worker(
            client, result_queue,
            start_index, stop_index,
            lock,
            profileHeatmap, clusterProfiles,
            profile_metric_index, profile_threshold,
            binSimilarityMatrix, normalizationFactor)
        client.basic_ack(result)

    consume_work = client.basic_consume(queue=work_queue, prefetch_count=1,
                                        callback=on_work)

    def on_pub(promise, result):
        if result['body'] == 'QUIT':
            client.loop_break()

    promise = client.queue_declare(exclusive=True)
    result = client.wait(promise)
    pub_queue = result['queue']

    client.queue_bind(exchange=pub_exchange, queue=pub_queue)
    consume_pub = client.basic_consume(queue=pub_queue, no_ack=True,
                                       callback=on_pub)

    promise = client.basic_publish(exchange='', routing_key=result_queue,
                                   body='READY')
    client.wait(promise)

    client.loop()

    client.wait(client.close())


@cython.cdivision(True)
@cython.boundscheck(False)
def cy_compute_treatment_distance_map_worker(
    client, result_queue,
    int start_index, int stop_index,
    lock,
    numpy.ndarray[DTYPE_FLOAT_t, ndim=2] profileHeatmap,
    numpy.ndarray[DTYPE_FLOAT_t, ndim=2] clusterProfiles,
    int profile_metric_index=0,
    float profile_threshold=0.0,
    numpy.ndarray[DTYPE_FLOAT_t, ndim=2] binSimilarityMatrix=None,
    float normalizationFactor=0.5):

    if not clusterProfiles.flags.c_contiguous:
        raise ValueError('clusterProfiles must be a C contiguous array')

    cdef int i, j, k
    cdef int ntreatments = clusterProfiles.shape[0]
    cdef int nclusters = clusterProfiles.shape[1]

    #cdef numpy.ndarray[DTYPE_FLOAT_t, ndim=2] profileHeatmap
    #profileHeatmap = numpy.zeros(
        #(ntreatments, ntreatments),
        #dtype=DTYPE_FLOAT)
    #profileHeatmap[numpy.identity(ntreatments, dtype=bool)] = 0.0

    if not profileHeatmap.flags.c_contiguous:
        raise ValueError('profileHeatmap must be a C contiguous array')

    if not binSimilarityMatrix.flags.c_contiguous:
        raise ValueError('binSimilarityMatrix must be a C contiguous array')

    cdef numpy.ndarray[DTYPE_FLOAT_t, ndim=1] profile1
    cdef numpy.ndarray[DTYPE_FLOAT_t, ndim=1] norm_profile1
    cdef numpy.ndarray[DTYPE_FLOAT_t, ndim=1] profile2
    cdef numpy.ndarray[DTYPE_FLOAT_t, ndim=1] norm_profile2
    cdef numpy.ndarray[DTYPE_FLOAT_t, ndim=2] A
    cdef DTYPE_FLOAT_t m
    cdef numpy.ndarray[DTYPE_FLOAT_t, ndim=1] P
    cdef numpy.ndarray[DTYPE_FLOAT_t, ndim=1] Q
    cdef numpy.ndarray[DTYPE_FLOAT_t, ndim=1] D
    cdef numpy.ndarray[DTYPE_FLOAT_t, ndim=1] N
    cdef DTYPE_FLOAT_t M, dist
    cdef numpy.ndarray zero_mask

    cdef DTYPE_FLOAT_t v
    cdef numpy.ndarray[DTYPE_FLOAT_t, ndim=2] tmp_array
    tmp_array = numpy.empty((stop_index - start_index, ntreatments),
                            dtype=DTYPE_FLOAT)

    for i in xrange(start_index, stop_index):

        profile1 = clusterProfiles[i]
        norm_profile1 = profile1 / float(numpy.sum(profile1))

        #if profile_threshold > 0.0:
            #max = numpy.max(profile1)
            #threshold_mask = profile1 < max * profile_threshold
            #norm_profile1[threshold_mask] = 0.0
            #norm_profile1 = norm_profile1 / float(numpy.sum(norm_profile1))

        for j in xrange(i):

            profile2 = clusterProfiles[j]
            norm_profile2 = profile2 / float(numpy.sum(profile2))

            #if profile_threshold > 0.0:
                #max = numpy.max(profile2)
                #threshold_mask = profile2 < max * profile_threshold
                #norm_profile2[threshold_mask] = 0.0
                #norm_profile2 = norm_profile2 / float(numpy.sum(norm_profile2))

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

            tmp_array[i - start_index, j] = dist

        with lock:
            for i in xrange(start_index, stop_index):
                for j in xrange(i):
                    v = tmp_array[i - start_index, j]
                    profileHeatmap[i, j] = v
                    profileHeatmap[j, i] = v

        promise = client.basic_publish(exchange='', routing_key=result_queue,
                                       body=cPickle.dumps(('TREATMENT', i)))
        client.wait(promise)

    #v = profileHeatmap[start_index:stop_index]

    #result = ('CHUNK', (v, start_index, stop_index))
    result = ('CHUNK', (start_index, stop_index))
    promise = client.basic_publish(exchange='', routing_key=result_queue,
                                   body=cPickle.dumps(result))
    client.wait(promise)

    #return v
