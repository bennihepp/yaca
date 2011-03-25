import sys
import os

from test import *

import numpy as np

from src.core import cluster

import cPickle
import StringIO

output_filename = sys.argv[ 1 ]


repeat = 5
runs = 3

minkowski_p = 2
min = -1.0
max = +1.0


# first run
#N_list = [ 100, 1000, 10000, 20000, 40000, 50000, 100000 ]
#p_list = [ 10, 20, 40, 50, 100, 150, 200 ]
#k_list = [ 5, 10, 20, 40, 50, 100, 150, 200 ]
#swap_threshold_list = [ 0, 2, 5, 10 ]

# second run
#N_list = [ 100, 1000, 10000, 50000, 100000, 500000, 1000000 ]
#p_list = [ 50, 100, 150, 200 ]
#k_list = [ 20, 50, 100, 150, 200 ]
#swap_threshold_list = [ 0 ]

# third run
N_list = [ 100, 1000, 10000, 50000, 100000 ]
p_list = [ 50, 100 ]
k_list = [ 20, 50, 100, 200 ]
swap_threshold_list = [ 0 ]


#N_list = [ 30, 40, 50 ]
#p_list = [ 10, 20 ]
#k_list = [ 5, 10, 20 ]
#swap_threshold_list = [ 0 ]

array_shape = ( len( swap_threshold_list ), len( p_list ), len( k_list ), len( N_list ) )
array_dims = [ 'swap_threshold', 'p', 'k', 'N' ]
mean_times = np.empty( array_shape )
std_times = np.empty( array_shape )
mean_clocks = np.empty( array_shape )
std_clocks = np.empty( array_shape )

for swap_threshold in swap_threshold_list:
    i1 = swap_threshold_list.index( swap_threshold )

    for p in p_list:
        i2 = p_list.index( p )

        if  os.path.isfile( output_filename + ( '_p=%d_swap=%d.pic' % ( p, swap_threshold ) ) ) \
        and os.path.isfile( output_filename + ( '_p=%d_swap=%d.xls' % ( p, swap_threshold ) ) ):

            print '\rskipping clustering for p=%d, swap_threshold=%d...' % ( p, swap_threshold )

        else:

            for k in k_list:
                i3 = k_list.index( k )

                for N in N_list:
                    i4 = N_list.index( N )

                    print '\rtesting clustering for N=%d, k=%d, p=%d, swap_threshold=%d...' % ( N, k, p, swap_threshold )
                    format_dict = {
                                   'swap_threshold' : swap_threshold,
                                   'p' : p,
                                   'k' : k,
                                   'N' : N
                    }



                    result = test_cluster.test_cluster_kmeans( cluster, repeat, runs, N, p, k, minkowski_p, swap_threshold, min, max )
                    mean_time, std_time, mean_clock, std_clock = result
                    mean_times[ i1, i2, i3, i4 ] = mean_time
                    std_times[ i1, i2, i3, i4 ] = std_time
                    mean_clocks[ i1, i2, i3, i4 ] = mean_clock
                    std_clocks[ i1, i2, i3, i4 ] = std_clock


            result_dict = {
                           'mean_times' : mean_times[ i1, i2 ],
                           'std_times' : std_times[ i1, i2 ],
                           'mean_clocks' : mean_clocks[ i1, i2 ],
                           'std_clocks' : std_clocks[ i1, i2 ]
                        }
            f = open( output_filename + ( '_p=%d_swap=%d.pic' % ( p, swap_threshold ) ), 'w' )
            pic = cPickle.Pickler( f )
            pic.dump( result_dict )
            f.close()

            str = StringIO.StringIO()

            str.write( 'timing information for p=%d, swap_threshold=%d\n' % ( p, swap_threshold ) )
            num_of_tabs = 1 + array_shape[ 3 ] + 1
            str.write( 'times' + (  num_of_tabs * '\t' ) + 'clocks\n' )

            str1 = '\t'
            str2 = '\t'
            for i4 in xrange( array_shape[ 3 ] ):
                str1 += 'N=%d\t' % ( N_list[ i4 ] )
                str2 += 'N=%d\t' % ( N_list[ i4 ] )
            str2 = str2[ : -1 ] + '\n'
            str.write( str1 + '\t' + str2 )

            for i3 in xrange( array_shape[ 2 ] ):
                str1 = 'k=%d\t' % k_list[ i3 ]
                str2 = 'k=%d\t' % k_list[ i3 ]
                for i4 in xrange( array_shape[ 3 ] ):
                    str1 += '%.2f +- %.2f\t' % ( mean_times[ i1, i2, i3, i4 ], std_times[ i1, i2, i3, i4 ] )
                    str2 += '%.2f +- %.2f\t' % ( mean_clocks[ i1, i2, i3, i4 ], std_clocks[ i1, i2, i3, i4 ] )
                str2 = str2[ : -1 ] + '\n'
                str.write( str1 + '\t' + str2 )
            str.write( '\n' )

            f = open( output_filename + ( '_p=%d_swap=%d.xls' % ( p, swap_threshold ) ), 'w' )
            f.write( str.getvalue() )
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
f = open( output_filename + '.pic', 'w' )
p = cPickle.Pickler( f )
p.dump( result_dict )
f.close()

import StringIO
str = StringIO.StringIO()

for swap_threshold in swap_threshold_list:
    for p in p_list:
        i1,i2 = swap_threshold_list.index( swap_threshold ), p_list.index( p )
        str.write( 'timing information for p=%d, swap_threshold=%d\n' % ( p, swap_threshold ) )
        num_of_tabs = 1 + array_shape[ 3 ] + 1
        str.write( 'times' + (  num_of_tabs * '\t' ) + 'clocks\n' )

        str1 = '\t'
        str2 = '\t'
        for i4 in xrange( array_shape[ 3 ] ):
            str1 += 'N=%d\t' % ( N_list[ i4 ] )
            str2 += 'N=%d\t' % ( N_list[ i4 ] )
        str2 = str2[ : -1 ] + '\n'
        str.write( str1 + '\t' + str2 )

        for i3 in xrange( array_shape[ 2 ] ):
            str1 = 'k=%d\t' % k_list[ i3 ]
            str2 = 'k=%d\t' % k_list[ i3 ]
            for i4 in xrange( array_shape[ 3 ] ):
                str1 += '%.2f +- %.2f\t' % ( mean_times[ i1, i2, i3, i4 ], std_times[ i1, i2, i3, i4 ] )
                str2 += '%.2f +- %.2f\t' % ( mean_clocks[ i1, i2, i3, i4 ], std_clocks[ i1, i2, i3, i4 ] )
            str2 = str2[ : -1 ] + '\n'
            str.write( str1 + '\t' + str2 )
        str.write( '\n' )

f = open( output_filename, 'w' )
f.write( str.getvalue() )
f.close()

str.close()
"""
