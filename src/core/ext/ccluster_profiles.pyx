# -*- coding: utf-8 -*-

"""
ccluster_profiles.pyx -- Fast computation and manipulation of cluster profiles.
"""

# This software is distributed under the FreeBSD License.
# See the accompanying file LICENSE for details.
# 
# Copyright 2011 Benjamin Hepp

from __future__ import division

import sys

import numpy
cimport numpy
cimport cython

DTYPE_INT = numpy.int
ctypedef numpy.int_t DTYPE_INT_t
DTYPE_FLOAT = numpy.float
ctypedef numpy.float_t DTYPE_FLOAT_t
DTYPE_BOOL = numpy.int8
ctypedef numpy.int8_t DTYPE_BOOL_t

from cython.parallel import prange, parallel, threadid


@cython.cdivision(True)
@cython.boundscheck(False)
def compute_treatment_distance_map(
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

    cdef numpy.ndarray[DTYPE_FLOAT_t, ndim=2] profileHeatmap
    profileHeatmap = numpy.zeros(
        (ntreatments, ntreatments),
        dtype=DTYPE_FLOAT)
    profileHeatmap[numpy.identity(ntreatments, dtype=bool)] = 0.0

    if not profileHeatmap.flags.c_contiguous:
        raise ValueError('profileHeatmap must be a C contiguous array')

    #cdef numpy.ndarray[DTYPE_FLOAT_t, ndim=2] binSimilarityMatrix
    if binSimilarityMatrix is None:
        binSimilarityMatrix = numpy.identity(
            nclusters,
            dtype=DTYPE_FLOAT)

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

    for i in xrange(ntreatments):
    #for i in prange(ntreatments, nogil=True):

        profile1 = clusterProfiles[i]
        norm_profile1 = profile1 / float(numpy.sum(profile1))

        sys.stdout.write('\rprofile #{} out of {}'.format(i + 1, clusterProfiles.shape[0]))
        sys.stdout.flush()
        #print 'norm_profile1:', norm_profile1

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

            #if profile_metric == 'quadratic_chi':

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

            profileHeatmap[i, j] = dist
            profileHeatmap[j, i] = dist

    sys.stdout.write('\n')
    sys.stdout.flush()

    return profileHeatmap
