import numpy as np

import time

np.random.seed(int(time.time()))

def test_cluster_kmeans(cluster_module, repeat, runs, N, p, k, minkowski_p=2, swap_threshold=0, min=-1.0, max=1.0):

    points = np.empty((N, p))

    clocks = np.empty((repeat, runs))
    times = np.empty((repeat, runs))

    for i in xrange(repeat):
        points = setup_cluster_kmeans(cluster_module, points, k, min, max)
        for j in xrange(runs):
            t1 = time.time()
            c1 = time.clock()
            run_cluster_kmeans(cluster_module, points, k, minkowski_p, swap_threshold)
            c2 = time.clock()
            t2 = time.time()
            dt = t2 - t1
            dc = c2 - c1
            clocks[i, j] = c2 - c1
            times[i, j] = t2 - t1

    mean_clock = np.mean(clocks)
    std_clock = np.std(clocks)
    mean_time = np.mean(times)
    std_time = np.std(times)

    return mean_time, std_time, mean_clock, std_clock

def setup_cluster_kmeans(cluster_module, points, k, min, max):

    points = np.random.uniform(min, max, points.shape)
    return points

def run_cluster_kmeans(cluster_module, points, k, minkowski_p, swap_threshold):

    cluster_module.cluster_kmeans(points, k, minkowski_p, swap_threshold)
