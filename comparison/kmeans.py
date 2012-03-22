import sys
import time

import numpy as np

if len(sys.argv) > 1:
    log_filename = sys.argv[1]
    log_file = open(log_filename, 'a')
else:
    log_file = None

def test_cluster_sklearn(runs, data, k):

    import sklearn.cluster

    global log_file

    np.random.seed(int(time.time()))

    clocks = np.empty((runs,))
    times = np.empty((runs,))
    inertias = np.empty((runs,))

    for i in xrange(runs):
        t1 = time.time()
        c1 = time.clock()
        KMeans = sklearn.cluster.KMeans(k)
        KMeans.fit(data)
        c2 = time.clock()
        t2 = time.time()
        dt = t2 - t1
        dc = c2 - c1
        clocks[i] = c2 - c1
        times[i] = t2 - t1
        inertias[i] = KMeans.inertia_

    mean_clock = np.mean(clocks)
    std_clock = np.std(clocks)
    mean_time = np.mean(times)
    std_time = np.std(times)

    print 'sklearn: %d objects, %d features, %d clusters: clocks=%f +- %f, times=%f +- %f' % (data.shape[0], data.shape[1], k, mean_clock, std_clock, mean_time, std_time)
    print '  inertias:', inertias
    if log_file is not None:
        print >> log_file, '%d objects, %d features, %d clusters: clocks=%f +- %f, times=%f +- %f' % (data.shape[0], data.shape[1], k, mean_clock, std_clock, mean_time, std_time)
        print >> log_file, '  inertias:', inertias

    return mean_time, std_time, mean_clock, std_clock, inertias

def test_cluster_kmeans(runs, data, k, n_init=10):

    import sklearn.cluster

    from src.core import cluster

    global log_file

    np.random.seed(int(time.time()))

    inertias = np.zeros((runs,))

    def _tolerance(X, tol):
        """Return a tolerance which is independent of the dataset"""
        variances = np.var(X, axis=0)
        return np.mean(variances) * tol
    tol = 1e-4
    tol = _tolerance(data, tol)

    # subtract of mean of x for more accurate distance computations
    data = data.copy()
    data_mean = data.mean(axis=0)
    data -= data_mean

    for i in xrange(runs):
        best_inertia = None
        for n in xrange(n_init):
            clusters = sklearn.cluster.k_means_.k_init(data, k)
            partition, clusters, inertia = cluster.cluster_kmeans(
                data,
                k,
                clusters=clusters,
                use_ccluster=False,
                clusters_tol=tol
           )
            if best_inertia is None or inertia[-1] < best_inertia:
                best_inertia = inertia[-1]
        inertias[i] = best_inertia

    print 'kmeans: %d objects, %d features, %d clusters' % (data.shape[0], data.shape[1], k)
    print '  inertias:', inertias
    if log_file is not None:
        print >> log_file, '%d objects, %d features, %d clusters' % (data.shape[0], data.shape[1], k)
        print >> log_file, '  inertias:', inertias

    return inertias

import cPickle
f = open('/g/pepperkok/hepp/cell_objects_COP.pic', 'r')
up = cPickle.Unpickler(f)
original_data = up.load()
f.close()

runs = 3
#N = [1000, np.sqrt(10)*1000, 10000, np.sqrt(10)*10000, 100000]
#K = [100, 200]
N = [1000]
K = [5]

for n in N:
    n = int(n)
    for k in K:
        a = np.arange(original_data.shape[0])
        np.random.shuffle(a)
        data = original_data[a[:n]]
        data = np.asarray(data, dtype=np.float32)

        test_cluster_kmeans(runs, data, k)

        test_cluster_sklearn(runs, data, k)
