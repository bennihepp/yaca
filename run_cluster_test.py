import sys
import os

from test import *

import numpy as np

from src.core import cluster

import cPickle
import StringIO


output_filename = None

repeat = 3
runs = 3

minkowski_p = 2
min = -1.0
max = +1.0


# first run
#N_list = [100, 1000, 10000, 20000, 40000, 50000, 100000]
#p_list = [10, 20, 40, 50, 100, 150, 200]
#k_list = [5, 10, 20, 40, 50, 100, 150, 200]
#swap_threshold_list = [0, 2, 5, 10]

# second run
#N_list = [100, 1000, 10000, 50000, 100000, 500000, 1000000]
#p_list = [50, 100, 150, 200]
#k_list = [20, 50, 100, 150, 200]
#swap_threshold_list = [0]

# third run
N_list = [100, 1000, 10000, 50000, 100000]
p_list = [50, 100]
k_list = [20, 50, 100, 200]
swap_threshold_list = [0]


#N_list = [30, 40, 50]
#p_list = [10, 20]
#k_list = [5, 10, 20]
#swap_threshold_list = [0]


skip_next = 0

def print_help():

    sys.stderr.write("""Usage: python %s [options]
Necessary options:
  --output-file <filename>      Output file for the timing measurements.
Optional options:
  --repeat <i>                  A complete test-run is done i times.
  --runs <j>                    In each test-run a random dataset is created and the algorithm is run <j> times.
  --minkowski-p <minkowski-p>   A parameter for the k-means algorithm.
  --min <min>                   The minimum value for the range of the random dataset.
  --max <max>                   The max value for the range of the random dataset.
  --N-list <list>               List of values for N.
  --p-list <list>               List of values for p.
  --k-list <list>               List of values for k.
  --swap-threshold-list <list>  List of values for swap-threshold.
""" % sys.argv[0])


def make_param_list(x):
    if type(x) == str:
        l1 = []
        x = x.split()
        for y in x:
            l1.extend(y.split(','))
        l2 = []
        for y in l1:
            l2.extend(y.split(';'))
        x = l2
    elif type(x) == float or type(x) == int:
        return [x]
    l = []
    for y in x:
        try:
            l.append(int(y))
        except:
            try:
                l.append(float(y))
            except:
                l.append(y)
    return l

if len(sys.argv) > 1:
    for i in xrange(1, len(sys.argv)):

        arg = sys.argv[i]
        if i < len(sys.argv) - 1:
            next_arg = sys.argv[i+1]
        else:
            next_arg = None

        if skip_next > 0:
            skip_next -= 1
            continue


        if arg == '--output-file':
            output_filename = next_arg
            skip_next = 1
        elif arg == '--repeat':
            repeat = int(next_arg)
            skip_next = 1
        elif arg == '--runs':
            runs = int(next_arg)
            skip_next = 1
        elif arg == '--minkowski-p':
            minkowski_p = int(next_arg)
            skip_next = 1
        elif arg == '--min':
            min = int(next_arg)
            skip_next = 1
        elif arg == '--max':
            max = int(next_arg)
            skip_next = 1
        elif arg == '--N-list':
            N_list = make_param_list(next_arg)
            skip_next = 1
        elif arg == '--p-list':
            p_list = make_param_list(next_arg)
            skip_next = 1
        elif arg == '--k-list':
            k_list = make_param_list(next_arg)
            skip_next = 1
        elif arg == '--swap-threshold-list':
            swap_threshold_list = make_param_list(next_arg)
            skip_next = 1
        elif arg == '--help':
            print_help()
            sys.exit(0)
        elif arg == '':
            continue
        else:
            sys.stderr.write('Unknown option: %s\n' % arg)
            print_help()
            sys.exit(-1)

if output_filename == None:
    print 'You need to specifiy an output file'
    print_help()
    sys.exit(1)


array_shape = (len(swap_threshold_list), len(p_list), len(k_list), len(N_list))
array_dims = ['swap_threshold', 'p', 'k', 'N']
mean_times = np.empty(array_shape)
std_times = np.empty(array_shape)
mean_clocks = np.empty(array_shape)
std_clocks = np.empty(array_shape)

for swap_threshold in swap_threshold_list:
    i1 = swap_threshold_list.index(swap_threshold)

    for p in p_list:
        i2 = p_list.index(p)

        if  os.path.isfile(output_filename + ('_p=%d_swap=%d.pic' % (p, swap_threshold))) \
        and os.path.isfile(output_filename + ('_p=%d_swap=%d.xls' % (p, swap_threshold))):

            print '\rskipping clustering for p=%d, swap_threshold=%d...' % (p, swap_threshold)

        else:


            for k in k_list:
                i3 = k_list.index(k)

                for N in N_list:
                    i4 = N_list.index(N)

                    print '\rtesting clustering for N=%d, k=%d, p=%d, swap_threshold=%d...' % (N, k, p, swap_threshold)
                    format_dict = {
                                   'swap_threshold' : swap_threshold,
                                   'p' : p,
                                   'k' : k,
                                   'N' : N
                    }



                    result = test_cluster.test_cluster_kmeans(cluster, repeat, runs, N, p, k, minkowski_p, swap_threshold, min, max)
                    mean_time, std_time, mean_clock, std_clock = result
                    mean_times[i1, i2, i3, i4] = mean_time
                    std_times[i1, i2, i3, i4] = std_time
                    mean_clocks[i1, i2, i3, i4] = mean_clock
                    std_clocks[i1, i2, i3, i4] = std_clock


            result_dict = {
                           'mean_times' : mean_times[i1, i2],
                           'std_times' : std_times[i1, i2],
                           'mean_clocks' : mean_clocks[i1, i2],
                           'std_clocks' : std_clocks[i1, i2]
                        }
            f = open(output_filename + ('_p=%d_swap=%d.pic' % (p, swap_threshold)), 'w')
            pic = cPickle.Pickler(f)
            pic.dump(result_dict)
            f.close()

            str = StringIO.StringIO()

            str.write('timing information for p=%d, swap_threshold=%d\n\n' % (p, swap_threshold))
            num_of_tabs = 1 + array_shape[3] + 1
            str.write('times mean' + ( num_of_tabs * '\t') + 'clocks mean\n')

            str1 = '\t'
            str2 = '\t'
            for i4 in xrange(array_shape[3]):
                str1 += 'N=%d\t' % (N_list[i4])
                str2 += 'N=%d\t' % (N_list[i4])
            str2 = str2[: -1] + '\n'
            str.write(str1 + '\t' + str2)

            for i3 in xrange(array_shape[2]):
                str1 = 'k=%d\t' % k_list[i3]
                str2 = 'k=%d\t' % k_list[i3]
                for i4 in xrange(array_shape[3]):
                    str1 += '%.2f\t' % (mean_times[i1, i2, i3, i4],)
                    str2 += '%.2f\t' % (mean_clocks[i1, i2, i3, i4],)
                str2 = str2[: -1] + '\n'
                str.write(str1 + '\t' + str2)
            str.write('\n')

            str.write('times stddev' + ( num_of_tabs * '\t') + 'clocks stddev\n')

            str1 = '\t'
            str2 = '\t'
            for i4 in xrange(array_shape[3]):
                str1 += 'N=%d\t' % (N_list[i4])
                str2 += 'N=%d\t' % (N_list[i4])
            str2 = str2[: -1] + '\n'
            str.write(str1 + '\t' + str2)

            for i3 in xrange(array_shape[2]):
                str1 = 'k=%d\t' % k_list[i3]
                str2 = 'k=%d\t' % k_list[i3]
                for i4 in xrange(array_shape[3]):
                    str1 += '%.2f\t' % (std_times[i1, i2, i3, i4],)
                    str2 += '%.2f\t' % (std_clocks[i1, i2, i3, i4],)
                str2 = str2[: -1] + '\n'
                str.write(str1 + '\t' + str2)
            str.write('\n\n')


            f = open(output_filename + ('_p=%d_swap=%d.xls' % (p, swap_threshold)), 'w')
            f.write(str.getvalue())
            f.close()

            str.close()

print 'finished testing clustering...exporting results...'

"""import cPickle
result_dict = {
               'mean_times' : mean_times,
               'std_times' : std_times,
               'mean_clocks' : mean_clocks,
               'std_clocks' : std_clocks
            }
f = open(output_filename + '.pic', 'w')
p = cPickle.Pickler(f)
p.dump(result_dict)
f.close()

import StringIO
str = StringIO.StringIO()

for swap_threshold in swap_threshold_list:
    for p in p_list:
        i1,i2 = swap_threshold_list.index(swap_threshold), p_list.index(p)
        str.write('timing information for p=%d, swap_threshold=%d\n' % (p, swap_threshold))
        num_of_tabs = 1 + array_shape[3] + 1
        str.write('times' + ( num_of_tabs * '\t') + 'clocks\n')

        str1 = '\t'
        str2 = '\t'
        for i4 in xrange(array_shape[3]):
            str1 += 'N=%d\t' % (N_list[i4])
            str2 += 'N=%d\t' % (N_list[i4])
        str2 = str2[: -1] + '\n'
        str.write(str1 + '\t' + str2)

        for i3 in xrange(array_shape[2]):
            str1 = 'k=%d\t' % k_list[i3]
            str2 = 'k=%d\t' % k_list[i3]
            for i4 in xrange(array_shape[3]):
                str1 += '%.2f +- %.2f\t' % (mean_times[i1, i2, i3, i4], std_times[i1, i2, i3, i4])
                str2 += '%.2f +- %.2f\t' % (mean_clocks[i1, i2, i3, i4], std_clocks[i1, i2, i3, i4])
            str2 = str2[: -1] + '\n'
            str.write(str1 + '\t' + str2)
        str.write('\n')

f = open(output_filename, 'w')
f.write(str.getvalue())
f.close()

str.close()
"""
