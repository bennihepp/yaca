import numpy as np

import time

import hcluster

N = 5000

def test_pdist(repeat, runs, data):

    np.random.seed( int( time.time() ) )

    clocks = np.empty( ( repeat, runs ) )
    times = np.empty( ( repeat, runs ) )

    for i in xrange( repeat ):
        for j in xrange( runs ):
            t1 = time.time()
            c1 = time.clock()
            dist_m = hcluster.pdist(data)
            c2 = time.clock()
            t2 = time.time()
            dt = t2 - t1
            dc = c2 - c1
            clocks[ i, j ] = c2 - c1
            times[ i, j ] = t2 - t1
            del dist_m

    mean_clock = np.mean( clocks )
    std_clock = np.std( clocks )
    mean_time = np.mean( times )
    std_time = np.std( times )

    print '%d objects, %d features: clocks=%f +- %f, times=%f +- %f' % (data.shape[0], data.shape[1], mean_clock, std_clock, mean_time, std_time)

    return mean_time, std_time, mean_clock, std_clock

import h5py
import cPickle
f = open('/g/pepperkok/hepp/cell_objects_COP.pic', 'r')
up = cPickle.Unpickler(f)
data = up.load()
f.close()

a = np.arange(data.shape[0])
np.random.shuffle(a)
data = data[a[:N]]
data = np.asarray(data, dtype=np.float32)

print 'computing distance matrix...'
dist_m = hcluster.pdist(data)
print 'done'

#f = open('/g/pepperkok/hepp/dist_matrix2_COP.pic', 'w')
#p = cPickle.Pickler(f)
#p.dump(dist_m)
#f.close()
f = h5py.File('/g/pepperkok/hepp/dist_matrix_%d_COP.hdf5' % N, mode='w' )
root = f.create_group( 'dist_matrix' )
dist_m_dataset = root.create_dataset('dist_m', data=dist_m)
f.close()
del dist_m

repeat, runs = 3, 1
#test_pdist(repeat, runs, data)
