# -*- coding: utf-8 -*-

"""
ccluster_profiles_mp_zmq.pyx -- Fast multiprocessing computation and
                                manipulation of cluster profiles using ZMQ
"""

# This software is distributed under the FreeBSD License.
# See the accompanying file LICENSE for details.
# 
# Copyright 2011 Benjamin Hepp

from __future__ import division

import sys
import math
import tempfile

import multiprocessing

import numpy
import zmq

import ccluster_profiles_mp_worker_zmq

def compute_treatment_distance_map(
    clusterProfiles,
    profile_metric_index=0,
    profile_threshold=0.0,
    binSimilarityMatrix=None,
    normalizationFactor=0.5,
    num_of_processes=10,
    num_of_chunks=50):

    if binSimilarityMatrix is None:
        binSimilarityMatrix = numpy.identity(
            clusterProfiles.shape[1],
            dtype=numpy.float)
    
    lock = multiprocessing.Lock()    
    shared_profileHeatmap = multiprocessing.Array(
        'd', clusterProfiles.shape[0] * clusterProfiles.shape[0],
        lock=False)
    shared_clusterProfiles = multiprocessing.Array(
        'd', clusterProfiles.reshape(clusterProfiles.shape[0] \
                                     * clusterProfiles.shape[1]),
        lock=False)

    print 'setting up sockets...'
    context = zmq.Context()
    pull_socket = context.socket(zmq.PULL)
    pull_tmpfile = tempfile.NamedTemporaryFile(delete=False)
    pull_endpoint = 'ipc://{}'.format(pull_tmpfile.name)
    pull_socket.bind(pull_endpoint)
    rep_socket = context.socket(zmq.REP)
    rep_tmpfile = tempfile.NamedTemporaryFile(delete=False)
    rep_endpoint = 'ipc://{}'.format(rep_tmpfile.name)
    rep_socket.bind(rep_endpoint)

    try:

        pool = []
        results = []
        # start processes
        print 'spawning processes...'
        # start processes
        for i in xrange(num_of_processes):
            args = (pull_endpoint, rep_endpoint,
                    lock,
                    clusterProfiles.shape[0],
                    clusterProfiles.shape[1],
                    shared_profileHeatmap,
                    shared_clusterProfiles, profile_metric_index,
                    profile_threshold, binSimilarityMatrix,
                    normalizationFactor)
            p = multiprocessing.Process(
                target=ccluster_profiles_mp_worker_zmq.\
                    compute_treatment_distance_map_worker,
                args=args)
            pool.append(p)
            p.start()

        poller = zmq.Poller()
        poller.register(rep_socket, zmq.POLLIN)
        poller.register(pull_socket, zmq.POLLIN)

        # wait for workers to connect and ask for jobs
        i = 0
        k = 0
        q = 0
        chunk_size \
            = int(math.ceil(clusterProfiles.shape[0] / float(num_of_chunks)))
        profileHeatmap = numpy.empty(
            (clusterProfiles.shape[0], clusterProfiles.shape[0]))
        profileHeatmap[:,:] = numpy.nan
        while q < num_of_processes:
            try:
                socks = dict(poller.poll())
                if socks.get(rep_socket) == zmq.POLLIN:
                    obj = rep_socket.recv_pyobj()
                    if obj == 'READY':
                        start_index = i * chunk_size
                        stop_index = min((i + 1) * chunk_size, clusterProfiles.shape[0])
                        if start_index < stop_index:
                            msg = {'start_index': start_index,
                                   'stop_index': stop_index,
                            }
                            rep_socket.send_pyobj(('ARGS', msg))
                            i += 1
                        else:
                            num_of_chunks = i
                            rep_socket.send_pyobj(('QUIT', ''))
                    else:
                        rep_socket.send_pyobj(('ERROR', ''))
                if socks.get(pull_socket) == zmq.POLLIN:
                    id, load = pull_socket.recv_pyobj()
                    if id == 'CHUNK':
                        #v, start_index, stop_index = load
                        #profileHeatmap[start_index:stop_index] = v
                        pass
                    elif id == 'TREATMENT':
                        sys.stdout.write('\rprocessed {} out of {} treatments'.format(
                            k + 1, clusterProfiles.shape[0]))
                        sys.stdout.flush()
                        k += 1
                    elif id == 'QUIT':
                        q += 1
            except zmq.ZMQError:
                pass
        # cleanup results
        profileHeatmap = numpy.array(
            numpy.frombuffer(shared_profileHeatmap))
        profileHeatmap = profileHeatmap.reshape(
            (clusterProfiles.shape[0], clusterProfiles.shape[0]))
        tmp = numpy.tril(profileHeatmap)
        profileHeatmap = tmp + tmp.transpose()
        sys.stdout.write('\n')
        sys.stdout.flush()
        # wait for workers to quit
        for p in pool:
            p.join()

    finally:
        # teardown sockets
        rep_socket.close()
        rep_tmpfile.close()
        pull_socket.close()
        pull_tmpfile.close()
        context.term()

    # return result
    return profileHeatmap
