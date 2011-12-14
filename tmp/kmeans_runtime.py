import sys
import time

import numpy as np

from core import cluster

if len(sys.argv) > 1:
    log_filename = sys.argv[1]
    log_file = open(log_filename, 'a')
else:
    log_file = None

def test_cluster_kmeans(repeat, runs, data, k):

    np.random.seed( int( time.time() ) )

    from core import cluster

    clocks = np.empty( ( repeat, runs ) )
    times = np.empty( ( repeat, runs ) )

    for i in xrange( repeat ):
        for j in xrange( runs ):
            t1 = time.time()
            c1 = time.clock()
            cluster.cluster_kmeans(data, k)
            c2 = time.clock()
            t2 = time.time()
            dt = t2 - t1
            dc = c2 - c1
            clocks[ i, j ] = c2 - c1
            times[ i, j ] = t2 - t1

    mean_clock = np.mean( clocks )
    std_clock = np.std( clocks )
    mean_time = np.mean( times )
    std_time = np.std( times )

    print '%d objects, %d features, %d clusters: clocks=%f +- %f, times=%f +- %f' % (data.shape[0], data.shape[1], k, mean_clock, std_clock, mean_time, std_time)
    if log_file is not None:
        print >> log_file, '%d objects, %d features, %d clusters: clocks=%f +- %f, times=%f +- %f' % (data.shape[0], data.shape[1], k, mean_clock, std_clock, mean_time, std_time)

    return mean_time, std_time, mean_clock, std_clock

import cPickle
f = open('/g/pepperkok/hepp/cell_objects_COP.pic', 'r')
up = cPickle.Unpickler(f)
original_data = up.load()
f.close()

repeat, runs = 3, 3
N = [1000, np.sqrt(10)*1000, 10000, np.sqrt(10)*10000, 100000]
K = [5, 10, 20, 50]

for n in N:
    n = int(n)
    for k in K:
        a = np.arange(original_data.shape[0])
        np.random.shuffle(a)
        data = original_data[a[:n]]
        data = np.asarray(data, dtype=np.float32)

        test_cluster_kmeans(repeat, runs, data, k)
