# -*- coding: utf-8 -*-

"""
ccluster_profiles_mp_amqp.pyx -- Fast multiprocessing computation and
                                manipulation of cluster profiles using AMQP
"""

# This software is distributed under the FreeBSD License.
# See the accompanying file LICENSE for details.
# 
# Copyright 2011 Benjamin Hepp

from __future__ import division

import sys
import math
import cPickle

import multiprocessing

import numpy
import puka

import ccluster_profiles_mp_worker_amqp

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

    print 'setting up message queues...'
    amqp_url = 'amqp://localhost/'
    client = puka.Client(amqp_url)
    promise = client.connect()
    client.wait(promise)
    promise = client.queue_declare(auto_delete=True, durable=False)
    result = client.wait(promise)
    work_queue = result['queue']
    promise = client.queue_declare(auto_delete=True, durable=False)
    result = client.wait(promise)
    result_queue = result['queue']
    pub_exchange = 'yaca_workers'
    promise = client.exchange_declare(pub_exchange, type='fanout',
                                      auto_delete=True, durable=False)
    result = client.wait(promise)

    try:

        pool = []
        results = []
        # start processes
        print 'spawning processes...'
        # start processes
        for i in xrange(num_of_processes):
            args = (amqp_url, work_queue, result_queue, pub_exchange,
                    lock,
                    clusterProfiles.shape[0],
                    clusterProfiles.shape[1],
                    shared_profileHeatmap,
                    shared_clusterProfiles, profile_metric_index,
                    profile_threshold, binSimilarityMatrix,
                    normalizationFactor)
            p = multiprocessing.Process(
                target=ccluster_profiles_mp_worker_amqp.\
                    compute_treatment_distance_map_worker,
                args=args)
            pool.append(p)
            p.start()

        consume_result = client.basic_consume(queue=result_queue, no_ack=True)

        # waiting for clients
        for i in xrange(num_of_processes):
            result = client.wait(consume_result)

        chunk_size \
            = int(math.ceil(clusterProfiles.shape[0] / float(num_of_chunks)))
        for i in xrange(num_of_chunks):
            start_index = i * chunk_size
            stop_index = min((i + 1) * chunk_size, clusterProfiles.shape[0])
            if stop_index <= start_index:
                num_of_chunks = i
                break
            chunk = (start_index, stop_index)
            promise = client.basic_publish(exchange='', routing_key=work_queue,
                                 body=cPickle.dumps(chunk))
            client.wait(promise)

        k = 0
        q = 0
        while q < num_of_chunks:
            result = client.wait(consume_result)
            body = result['body']
            id, load = cPickle.loads(result['body'])
            if id == 'CHUNK':
                #v, start_index, stop_index = load
                #profileHeatmap[start_index:stop_index] = v
                q += 1
            elif id == 'TREATMENT':
                sys.stdout.write('\rprocessed {} out of {} treatments'.format(
                    k + 1, clusterProfiles.shape[0]))
                sys.stdout.flush()
                k += 1

        promise = client.basic_publish(exchange=pub_exchange, routing_key='',
                                       body='QUIT')
        client.wait(promise)

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
        # teardown message queues
        client.wait(client.close())

    # return result
    return profileHeatmap
