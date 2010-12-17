import sys

import numpy
import scipy.spatial



CLUSTER_METHOD_KMEANS = 0
CLUSTER_METHOD_KMEDIANS = 1
CLUSTER_METHOD_KMEDOIDS = 2

def cluster(method, points, k, minkowski_p=2, calculate_silhouette=False, callback=None):
    if method == CLUSTER_METHOD_KMEANS:
        p,c = cluster_kmeans( points, k, minkowski_p, 0, callback )
    elif method == CLUSTER_METHOD_KMEDIANS:
        p,c = cluster_kmedians( points, k, minkowski_p, 0, callback )
    elif method == CLUSTER_METHOD_KMEDOIDS:
        p,c = cluster_kmedoids( points, k, minkowski_p, callback )
    else:
        raise Exception( 'Unknown clustering method' )

    if calculate_silhouette:
        s = silhouette( points, p, c, minkowski_p )
    else:
        s = None

    return p,c,s



# Returns a clustering of all the N observations.
# points is a NxM matrix whereas each row is an observation
# k is the number of clusters
# the clustering stops if less than swap_threshold swaps have
# been performed in an iteration
def cluster_by_dist(points, k, swap_threshold = 0, callback=None):

    clusters = []
    clusters.append( points[ numpy.random.randint( 0,points.shape[0] ) ] )

    partition = numpy.zeros( points.shape[0], int )

    dist_to_cluster = None

    for i in xrange(1,k):
        dist_m = scipy.spatial.distance.cdist([clusters[-1]], points)

        if dist_to_cluster == None:
            dist_to_cluster = dist_m
        else:
            dist_to_cluster = numpy.min([dist_m,dist_to_cluster],axis=0)
        max_dist = 0.0
        max_j = 0
        for j in xrange(dist_to_cluster.shape[1]):
            if dist_to_cluster[0,j] > max_dist:
                max_j = j
        clusters.append(points[j])

    clusters = numpy.array( clusters )


    #print 'cluster_mean_count:'
    #print cluster_mean_count
    #print 'cluster_mean:'
    #print cluster_mean

    iterations = 0
    swaps = swap_threshold + 1
    while swaps > swap_threshold:

        if iterations % 1 == 0:
            if callback != None:
                #if not callback( iterations, swaps ):
                #    break
                callback( iterations, swaps )
            sys.stdout.write( '\riteration %d: swaps = %d ... ' % ( iterations, swaps ) )
            sys.stdout.flush()

        swaps = 0

        dist_to_clusters = scipy.spatial.distance.cdist( points, clusters )

        for i in xrange( points.shape[0] ):
            row = dist_to_clusters[ i , : ]
            min_dist = 0.0
            min_j = -1
            for j in xrange(row.shape[0]):
                if row[j] < min_dist or min_j == -1:
                    min_dist = row[j]
                    min_j = j
            if partition[i] != min_j:
                swaps += 1
                partition[i] = min_j

        if swaps <= swap_threshold:
            #print 'cluster_mean_count:'
            #print cluster_mean_count
            #print 'cluster_mean:'
            #print cluster_mean
            continue

        # calculate center with mahalanobis distance

        old_cluster_mean = clusters.copy()
        cluster_mean = numpy.zeros( clusters.shape, float )
        cluster_mean_count = numpy.zeros( clusters.shape[0], int )

        for i in xrange(points.shape[0]):
            cluster_mean[ partition[ i ] ] += points[i]
            cluster_mean_count[ partition[ i ] ] += 1

        cluster_mean = ( cluster_mean.transpose() / cluster_mean_count ).transpose()

        for i in xrange( cluster_mean_count.shape[0] ):
            if cluster_mean_count[i] == 0:
                cluster_mean[i] = old_cluster_mean[i]
        #print 'cluster_mean_count:'
        #print cluster_mean_count
        #print 'cluster_mean:'
        #print cluster_mean

        clusters = cluster_mean
        iterations += 1

    sys.stdout.write( 'done\n' )
    sys.stdout.flush()

    return partition, clusters



def upside_down_gaussian( d, mad, sharpness=2 ):
    tmp = 1 - numpy.exp( - ( d / mad ) ** sharpness )
    return tmp

def reverse_exponential( d, mad, sharpness=2 ):
    tmp = 1 - numpy.exp( - sharpness * d / mad )
    return tmp

def fermi_dirac_transform( d, mad, sharpness=2.0 ):

    tmp = ( d - mad ) / ( mad / sharpness )
    tmp = numpy.exp( tmp ) + 1

    tmp = 1 / tmp

    tmp = 1 - tmp

    return tmp

"""def fermi_dirac_similarity( points, reference, mad, sharpness=2.0 ):

    d = numpy.abs( points - reference )

    tmp = ( d - mad ) / ( mad / sharpness )
    tmp = numpy.exp( tmp ) + 1

    sim = 1 / tmp

    sim = numpy.sum( sim, 1 )

    return sim"""


# Returns a clustering of all the N observations.
# points is a NxM matrix whereas each row is an observation
# k is the number of clusters
# the clustering stops if less than swap_threshold swaps have
# been performed in an iteration
def cluster_by_fermi_dirac_dist(points, k, swap_threshold=0, sharpness=2, callback=None):

    clusters = []
    clusters.append( points[ numpy.random.randint( 0,points.shape[0] ) ] )

    partition = numpy.zeros( points.shape[0], int )

    dist_to_cluster = None

    for i in xrange(1,k):
        dist_m = scipy.spatial.distance.cdist([clusters[-1]], points)

        if dist_to_cluster == None:
            dist_to_cluster = dist_m
        else:
            dist_to_cluster = numpy.min([dist_m,dist_to_cluster],axis=0)
        max_dist = 0.0
        max_j = 0
        for j in xrange(dist_to_cluster.shape[1]):
            if dist_to_cluster[0,j] > max_dist:
                max_j = j
        clusters.append(points[j])

    clusters = numpy.array( clusters )


    #print 'cluster_mean_count:'
    #print cluster_mean_count
    #print 'cluster_mean:'
    #print cluster_mean

    #med = numpy.median( points, 0 )
    #mad = numpy.median( numpy.abs( points - med), 0 )

    #sim_to_clusters = numpy.empty( ( points.shape[0], clusters.shape[0] ) )


    med = numpy.median( points, 0 )
    mad = numpy.median( numpy.abs( points - med ), 0 )

    dist_to_clusters = numpy.empty( ( points.shape[0], clusters.shape[0] ) )

    iterations = 0
    swaps = swap_threshold + 1
    while swaps > swap_threshold:

        if iterations % 1 == 0:
            if callback != None:
                if not callback( iterations, swaps ):
                    break
            sys.stdout.write( '\riteration %d: swaps = %d ... ' % ( iterations, swaps ) )
            sys.stdout.flush()

        swaps = 0

        """for i in xrange( clusters.shape[0] ):
            sim_to_clusters[ : , i ] = fermi_dirac_similarity( points, clusters[ i ], mad, sharpness )"""

        for i in xrange( clusters.shape[0] ):

            d = numpy.abs( points - clusters[ i ] )
            d = reverse_exponential( d, mad, sharpness )
            dist = numpy.sum( d, 1 )
            dist_to_clusters[ : , i ] = dist

        for i in xrange(points.shape[0]):
            row = dist_to_clusters[ i , : ]
            min_dist = 0.0
            min_j = -1
            for j in xrange(row.shape[0]):
                if row[j] < min_dist or min_j == -1:
                    min_dist = row[j]
                    min_j = j
            if partition[i] != min_j:
                swaps += 1
                partition[i] = min_j\

        if swaps <= swap_threshold:
            #print 'cluster_mean_count:'
            #print cluster_mean_count
            #print 'cluster_mean:'
            #print cluster_mean
            continue

        # calculate center with mahalanobis distance

        old_clusters = clusters.copy()

        for i in xrange( clusters.shape[0] ):
            cluster_mask = partition == i
            if numpy.sum( cluster_mask ) > 0:
                cluster = numpy.median( points[ cluster_mask ], 0 )
            else:
                cluster = old_clusters[ i ]
            clusters[ i ] = cluster

        iterations += 1

    sys.stdout.write( 'done\n' )
    sys.stdout.flush()

    return partition, clusters


# Returns a kmeans-clustering of all the N observations.
# points is a NxM matrix whereas each row is an observation
# k is the number of clusters
# the clustering stops if less than swap_threshold swaps have
# been performed in an iteration
def cluster_kmeans(points, k, minkowski_p=2, swap_threshold = 0, callback=None):

    clusters = []
    clusters.append( points[ numpy.random.randint( 0,points.shape[0] ) ] )

    partition = numpy.zeros( points.shape[0], int )

    dist_to_cluster = None

    for i in xrange(1,k):
        if minkowski_p == 2:
            dist_m = scipy.spatial.distance.cdist( clusters[-1:], points, 'euclidean' )
        elif minkowski_p == 1:
            dist_m = scipy.spatial.distance.cdist( clusters[-1:], points, 'cityblock' )
        else:
            dist_m = scipy.spatial.distance.cdist( clusters[-1:], points, 'minkowski', minkowski_p )

        if dist_to_cluster == None:
            dist_to_cluster = dist_m
        else:
            dist_to_cluster = numpy.min( [ dist_m, dist_to_cluster ], axis=0 )
        max_dist = 0.0
        max_j = 0
        for j in xrange( dist_to_cluster.shape[1] ):
            if dist_to_cluster[ 0, j ] > max_dist:
                max_j = j
        clusters.append( points[ j ] )

    clusters = numpy.array( clusters )

    #print 'cluster_mean_count:'
    #print cluster_mean_count
    #print 'cluster_mean_count:'
    #print cluster_mean_count

    #print 'cluster_mean_count:'
    #print cluster_mean_count
    #print 'cluster_mean:'
    #print cluster_mean

    iterations = 0
    swaps = swap_threshold + 1
    while swaps > swap_threshold:

        if iterations % 1 == 0:
            if callback != None:
                #if not callback( iterations, swaps ):
                #    break
                callback( iterations, swaps )
            sys.stdout.write( '\riteration %d: swaps = %d ... ' % ( iterations, swaps ) )
            sys.stdout.flush()

        swaps = 0

        if minkowski_p == 2:
            dist_to_clusters = scipy.spatial.distance.cdist( points, clusters, 'euclidean' )
        elif minkowski_p == 1:
            dist_to_clusters = scipy.spatial.distance.cdist( points, clusters, 'cityblock' )
        else:
            dist_to_clusters = scipy.spatial.distance.cdist( points, clusters, 'minkowski', minkowski_p )

        min_index = numpy.argmin( dist_to_clusters, 1 )
        swaps = numpy.sum( min_index != partition )
        partition = min_index

        """for i in xrange( points.shape[0] ):
            row = dist_to_clusters[ i , : ]
            min_dist = row[ 0 ]
            min_j = 0
            for j in xrange( 1, row.shape[0] ):
                if row[j] < min_dist:
                    min_dist = row[j]
                    min_j = j
            if partition[i] != min_j:
                swaps += 1
                partition[i] = min_j"""

        if swaps <= swap_threshold:
            #print 'cluster_mean_count:'
            #print cluster_mean_count
            #print 'cluster_mean:'
            #print cluster_mean
            continue

        # calculate cluster means

        #old_clusters = clusters.copy()

        for i in xrange( clusters.shape[0] ):
            cluster_mask = ( partition[ : ] == i )
            if numpy.sum( cluster_mask ) > 0:
                cluster = numpy.mean( points[ cluster_mask ], 0 )
                clusters[ i ] = cluster
            #else:
            #    cluster = old_clusters[ i ]
            #clusters[ i ] = cluster

        iterations += 1

    sys.stdout.write( 'done\n' )
    sys.stdout.flush()

    return partition, clusters


# Returns a kmedians-clustering of all the N observations.
# points is a NxM matrix whereas each row is an observation
# k is the number of clusters
# the clustering stops if less than swap_threshold swaps have
# been performed in an iteration
def cluster_kmedians(points, k, minkowski_p=2, swap_threshold = 0, callback=None):

    clusters = []
    clusters.append( points[ numpy.random.randint( 0,points.shape[0] ) ] )

    partition = numpy.zeros( points.shape[0], int )

    dist_to_cluster = None

    for i in xrange(1,k):
        if minkowski_p == 2:
            dist_m = scipy.spatial.distance.cdist( clusters[-1:], points, 'euclidean' )
        elif minkowski_p == 1:
            dist_m = scipy.spatial.distance.cdist( clusters[-1:], points, 'cityblock' )
        else:
            dist_m = scipy.spatial.distance.cdist( clusters[-1:], points, 'minkowski', minkowski_p )

        if dist_to_cluster == None:
            dist_to_cluster = dist_m
        else:
            dist_to_cluster = numpy.min( [ dist_m, dist_to_cluster ], axis=0 )
        max_dist = 0.0
        max_j = 0
        for j in xrange( dist_to_cluster.shape[1] ):
            if dist_to_cluster[ 0, j ] > max_dist:
                max_j = j
        clusters.append( points[ j ] )

    clusters = numpy.array( clusters )

    #print 'cluster_mean_count:'
    #print cluster_mean_count
    #print 'cluster_mean:'
    #print cluster_mean

    iterations = 0
    swaps = swap_threshold + 1
    while swaps > swap_threshold:

        if iterations % 1 == 0:
            if callback != None:
                #if not callback( iterations, swaps ):
                #    break
                callback( iterations, swaps )
            sys.stdout.write( '\riteration %d: swaps = %d ... ' % ( iterations, swaps ) )
            sys.stdout.flush()

        swaps = 0

        if minkowski_p == 2:
            dist_to_clusters = scipy.spatial.distance.cdist( points, clusters, 'euclidean' )
        elif minkowski_p == 1:
            dist_to_clusters = scipy.spatial.distance.cdist( points, clusters, 'cityblock' )
        else:
            dist_to_clusters = scipy.spatial.distance.cdist( points, clusters, 'minkowski', minkowski_p )

        min_index = numpy.argmin( dist_to_clusters, 1 )
        swaps = numpy.sum( min_index != partition )
        partition = min_index

        """for i in xrange( points.shape[0] ):
            row = dist_to_clusters[ i , : ]
            min_dist = row[ 0 ]
            min_j = 0
            for j in xrange( 1, row.shape[0] ):
                if row[j] < min_dist:
                    min_dist = row[j]
                    min_j = j
            if partition[i] != min_j:
                swaps += 1
                partition[i] = min_j"""

        if swaps <= swap_threshold:
            #print 'cluster_mean_count:'
            #print cluster_mean_count
            #print 'cluster_mean:'
            #print cluster_mean
            continue

        # calculate cluster medians

        #old_clusters = clusters.copy()

        for i in xrange( clusters.shape[0] ):
            cluster_mask = ( partition[ : ] == i )
            if numpy.sum( cluster_mask ) > 0:
                cluster = numpy.median( points[ cluster_mask ], 0 )
                clusters[ i ] = cluster
            #else:
            #    cluster = old_clusters[ i ]
            #clusters[ i ] = cluster

        iterations += 1

    sys.stdout.write( 'done\n' )
    sys.stdout.flush()

    return partition, clusters

# Returns a kmedois-clustering of all the N observations.
# points is a NxM matrix whereas each row is an observation
# k is the number of clusters
# the clustering stops if less than swap_threshold swaps have
# been performed in an iteration
def cluster_kmedoids(points, k, minkowski_p=2, callback=None):

    clusters = []
    medoids = []

    medoid = numpy.random.randint( 0, points.shape[0] )
    medoids.append( medoid )
    clusters.append( points[ medoid ] )

    partition = numpy.zeros( points.shape[0], int )

    dist_to_cluster = None

    for i in xrange(1,k):
        if minkowski_p == 2:
            dist_m = scipy.spatial.distance.cdist( clusters[-1:], points, 'euclidean' )
        elif minkowski_p == 1:
            dist_m = scipy.spatial.distance.cdist( clusters[-1:], points, 'cityblock' )
        else:
            dist_m = scipy.spatial.distance.cdist( clusters[-1:], points, 'minkowski', minkowski_p )

        if dist_to_cluster == None:
            dist_to_cluster = dist_m
        else:
            dist_to_cluster = numpy.min( [ dist_m, dist_to_cluster ], axis=0 )
        max_dist = 0.0
        max_j = 0
        for j in xrange( dist_to_cluster.shape[1] ):
            if dist_to_cluster[ 0, j ] > max_dist:
                max_j = j
        clusters.append( points[ j ] )
        medoids.append( j )

    clusters = numpy.array( clusters )

    #print 'cluster_mean_count:'
    #print cluster_mean_count

    #print 'cluster_mean_count:'
    #print cluster_mean_count
    #print 'cluster_mean:'
    #print cluster_mean

    medoids_changed = True

    iterations = 0
    while medoids_changed:

        if iterations % 1 == 0:
            if callback != None:
                #if not callback( iterations, swaps ):
                #    break
                callback( iterations, swaps )
            sys.stdout.write( '\riteration %d: swaps = %d ... ' % ( iterations, swaps ) )
            sys.stdout.flush()

        if minkowski_p == 2:
            dist_to_clusters = scipy.spatial.distance.cdist( points, clusters, 'euclidean' )
        elif minkowski_p == 1:
            dist_to_clusters = scipy.spatial.distance.cdist( points, clusters, 'cityblock' )
        else:
            dist_to_clusters = scipy.spatial.distance.cdist( points, clusters, 'minkowski', minkowski_p )

        min_index = numpy.argmin( dist_to_clusters, 1 )

        """min_dist = dist_to_clusters[ : , 0 ]
        min_index = numpy.zeros( min_dist.shape, int )
        for j in xrange( 1, clusters.shape[0] ):
            dist = dist_to_clusters[ : , j ]
            mask = dist < min_dist:
            min_dist[ mask ] = dist[ mask ]
            min_index[ mask ] = j"""

        #swaps = numpy.sum( min_index != partition )
        partition = min_index

        """for i in xrange( points.shape[0] ):
            row = dist_to_clusters[ i , : ]
            min_dist = row[ 0 ]
            min_j = 0
            for j in xrange( 1, row.shape[0] ):
                if row[j] < min_dist:
                    min_dist = row[j]
                    min_j = j
            if partition[i] != min_j:
                swaps += 1
                partition[i] = min_j"""

        #E = numpy.sum( min_dist )

        min_i = -1
        min_j = -1
        min_dE = 0.0

        for i in xrange( clusters.shape[0] ):

            partition_mask = partition[:] == i

            for j in xrange( points.shape[0] ):

                if j == i:
                    continue

                if minkowski_p == 2:
                    dist = scipy.spatial.distance.cdist( points, [ points[ j ] ], 'euclidean' )
                elif minkowski_p == 1:
                    dist = scipy.spatial.distance.cdist( points, [ points[ j ] ], 'cityblock' )
                else:
                    dist = scipy.spatial.distance.cdist( points, [ points[ j ] ], 'minkowski', minkowski_p )

                mask = dist[ : ] < min_dist[ : ]
                dE1 = + numpy.sum( dist[ mask ] ) - numpy.sum( min_dist[ mask ] )

                mask = partition_mask
                mask = numpy.logical_and( mask, dist[ mask ] > min_dist[ mask ] )

                mask2 = numpy.any( dist_to_clusters[ mask ] < dist[ mask ], 1 )
                mask[ mask ] = mask2

                min_index = numpy.argmin( dist_to_clusters[ mask ], 1 )
                dE2 = + numpy.sum( numpy.min( dist_to_clusters[ partition_mask ], 1 ) ) - numpy.sum( min_dist[ mask ] )

                dE = dE1 + dE2
                if dE < 0 and dE < min_dE:
                    min_i = i
                    min_j = j
                    min_dE = dE

        # found new medoid
        if dE < 0:
            medoids[ min_i ] = min_j
            clusters[ min_i ] = points[ min_j ]
            medoids_changed = True
        else:
            medoids_changed = False

        iterations += 1


    if minkowski_p == 2:
        dist_to_clusters = scipy.spatial.distance.cdist( points, clusters, 'euclidean' )
    elif minkowski_p == 1:
        dist_to_clusters = scipy.spatial.distance.cdist( points, clusters, 'cityblock' )
    else:
        dist_to_clusters = scipy.spatial.distance.cdist( points, clusters, 'minkowski', minkowski_p )

    partition = numpy.argmin( dist_to_clusters, 1 )

    sys.stdout.write( 'done\n' )
    sys.stdout.flush()

    return partition, clusters


def silhouette(points, partition, clusters, minkowski_p=2):

    print 'calculating silhouette...'

    s = numpy.empty( ( points.shape[0], ) )

    for i in xrange( points.shape[0] ):

        if minkowski_p == 2:
            dist_to_cluster = scipy.spatial.distance.cdist( points, [ points[ i ] ], 'euclidean' )
        elif minkowski_p == 1:
            dist_to_cluster = scipy.spatial.distance.cdist( points, [ points[ i ] ], 'cityblock' )
        else:
            dist_to_cluster = scipy.spatial.distance.cdist( points, [ points[ i ] ], 'minkowski', minkowski_p )

        dist_to_cluster = dist_to_cluster[ : , 0 ]

        avg_cluster_dist = numpy.empty( ( clusters.shape[0], ) )

        min_avg_cluster_dist = 0.0
        min_j = -1

        for j in xrange( min_j + 1, clusters.shape[0] ):

            cluster_mask = partition[:] == j

            avg_cluster_dist[ j ] = numpy.mean( dist_to_cluster[ cluster_mask ] )

            if min_j < 0 or ( partition[ i ] != j and avg_cluster_dist[ j ] < min_avg_cluster_dist ):
                min_j = j
                min_avg_cluster_dist = avg_cluster_dist[ j ]

        s[ i ] = min_avg_cluster_dist - avg_cluster_dist[ partition[ i ] ]
        s[ i ] /= max( min_avg_cluster_dist, avg_cluster_dist[ partition[ i ] ] )

    print 'finished calculating silhouette'

    return s


def sample_reference_datasets(points, B):
    #print 'sampling reference datasets...'
    low = numpy.min( points, 0 )
    high = numpy.max( points, 0 )
    references = []
    for b in xrange( B ):
        ref = numpy.empty( points.shape )
        for i in xrange( points.shape[1] ):
            ref[ : , i ] = numpy.random.uniform( low[ i ], high[ i ], points.shape[ 0 ] )
        references.append( ref )
    #print 'finished sampling reference datasets'
    return references

def within_cluster_distance(points, partition, clusters, minkowski_p=2, cluster_callback=None):

    if minkowski_p != 2:
        raise Exception( 'Not implemented yet' )

    # THIS IS WRONG FOR NON-EUCLIDEAN METRICS!!!

    #print 'calculating within cluster distance...'

    c = numpy.empty( ( points.shape[0], clusters.shape[1] ) )
    cluster_size = numpy.empty( ( points.shape[0], ) )

    for i in xrange( clusters.shape[0] ):
        mask = partition[ : ] == i
        cluster_size[ mask ] = numpy.sum( mask )
        c[ mask ] = clusters[ i ]

    W = numpy.sum( numpy.sum( ( points - c )**2, 1 ) / cluster_size, 0 )

    #print 'finished calculating within cluster distance'

    return W

def gap_statistic(points, num_of_clusters, B=5, cluster_callback=None):
    minkowski_p = 2
    references = sample_reference_datasets(points, B)

    W = numpy.empty( B + 1 )
    all = [points] + references
    for i in xrange( len( all ) ):
        dataset = all[ i ]
        partition,clusters,silhouette = cluster(
                CLUSTER_METHOD_KMEANS,
                dataset,
                num_of_clusters,
                minkowski_p,
                False,
                cluster_callback
        )
        W[ i ] = within_cluster_distance( dataset, partition, clusters, minkowski_p, cluster_callback )
    W = numpy.log( W )
    gap = numpy.sum( W[ 1 : ] ) / B - W[ 0 ]
    stddev = numpy.std( W[ 1 : ] )
    sk = stddev * numpy.sqrt( 1 + 1/float(B) )
    return gap, sk

def determine_num_of_clusters(points, max_num_of_clusters, B=5, cluster_callback=None):
    gaps = numpy.empty( ( max_num_of_clusters, ) )
    sk = numpy.empty( gaps.shape )
    for num_of_clusters in xrange( 1, max_num_of_clusters + 1 ):
        print 'calculating gap statistic for %d clusters...' % num_of_clusters
        gap, stddev = gap_statistic( points, num_of_clusters, B, cluster_callback )
        gaps[ num_of_clusters - 1 ] = gap
        sk[ num_of_clusters - 1 ] = stddev
    best_num_of_clusters = -1
    for num_of_clusters in xrange( 1, max_num_of_clusters ):
        k = num_of_clusters - 1
        print 'k=%d, gaps[k]=%f, sk[k]=%f, gaps[k+1]=%f, sk[k+1]=%f' % (k, gaps[k], sk[k], gaps[k+1], sk[k+1])
        if gaps[ k ] >= gaps[ k + 1 ] - sk[ k + 1 ]:
            best_num_of_clusters = num_of_clusters
            break

    return best_num_of_clusters, gaps, sk
