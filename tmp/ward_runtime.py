import numpy as np

import time

import hcluster

N = 5000

def test_cluster_ward(repeat, runs, data):

    np.random.seed(int(time.time()))

    clocks = np.empty((repeat, runs))
    times = np.empty((repeat, runs))

    for i in xrange(repeat):
        for j in xrange(runs):
            print 'a'
            t1 = time.time()
            c1 = time.clock()
            Z = hcluster.linkage(data, 'ward')
            c2 = time.clock()
            t2 = time.time()
            print 'b'
            dt = t2 - t1
            dc = c2 - c1
            clocks[i, j] = c2 - c1
            times[i, j] = t2 - t1

    mean_clock = np.mean(clocks)
    std_clock = np.std(clocks)
    mean_time = np.mean(times)
    std_time = np.std(times)

    print '%d objects, %d features: clocks=%f +- %f, times=%f +- %f' % (data.shape[0], data.shape[1], mean_clock, std_clock, mean_time, std_time)

    return mean_time, std_time, mean_clock, std_clock

import cPickle
f = open('/g/pepperkok/hepp/cell_objects_COP.pic', 'r')
up = cPickle.Unpickler(f)
data = up.load()
f.close()

a = np.arange(data.shape[0])
np.random.shuffle(a)
data = data[a[:N]]
data = np.asarray(data, dtype=np.float32)

#import h5py
#f = h5py.File('/g/pepperkok/hepp/dist_matrix_5000_COP.hdf5', mode='r')
#root = f['dist_matrix']
#dist_m_dataset = root['dist_m']
#dist_m = np.empty(dist_m_dataset.shape, dtype=np.float32)
#dist_m_dataset.read_direct(dist_m)
#f.close()
#print 'done'

repeat, runs = 3, 1
#for k in [5,10,20,50]:

print 'starting'

test_cluster_ward(repeat, runs, data)
