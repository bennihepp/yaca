import sys

import numpy
import scipy.spatial

import distance

import hcluster as hc


CLUSTER_METHOD_KMEANS = 0
CLUSTER_METHOD_KMEDIANS = 1
CLUSTER_METHOD_KMEDOIDS = 2

def do_hcluster(method, points, id, minkowski_p=2):

    PATH = '/g/pepperkok/hepp/hcluster/'

    import pickle

    try:

        file = open( PATH+'hcluster_Z_%s_%d_%d.pic' % ( method, minkowski_p, id ), 'r' )
        Z = pickle.load( file )
        file.close()

    except:

        if method in [ 'single','complete','average','weighted' ]:
            print 'computing distances...'
            if minkowski_p == 2:
                dist_m = hc.pdist( points )
            else:
                dist_m = hc.pdist( points, 'minkowski', minkowski_p )
        else:
            if minkowski_p != 2:
                print 'WARNING: Using euclidean metric!!!'
            dist_m = points

        print 'clustering...'
        Z = hc.linkage( dist_m, method )
        print 'done'

        file = open( PATH+'hcluster_%s_%d_%d.pic' % ( method, minkowski_p, id ), 'w' )
        pickle.dump( [ points, dist_m, Z ], file )
        file.close()
    
        file = open( PATH+'hcluster_Z_%s_%d_%d.pic' % ( method, minkowski_p, id ), 'w' )
        pickle.dump( Z, file )
        file.close()

    return Z


def get_hcluster_methods():
    return [
        ( 'Random', 'random' ),
        ( 'KD-Tree', 'kd-tree' ),
        ( 'Seeding', 'seed' ),
        ( 'CLink', 'complete' ),
        ( 'SLink', 'single' ),
        ( 'ALink', 'average' ),
        ( 'Weighted', 'weighted' ),
        ( 'Ward', 'ward' ),
        ( 'HMedian', 'median' ),
        ( 'HCentroid', 'centroid' ),
        ( 'K-Means', 'k-means' ),
        ( 'K-Medians', 'k-medians' ),
        ( 'K-Medoids', 'k-medoids' )
    ]


class kd_tree_node(object):
    def __init__(self, value=None, left=None, right=None, id=None, dim=None):
        self.value = value
        self.left = left
        self.right = right
        self.id = id
        self.dim = dim

def __kd_tree(points, ids, depth, max_depth, partition):

    node = kd_tree_node()

    if depth < max_depth:

        k = points.shape[1]
        dim = depth % k

        print ' depth=%d, dim=%d, points.shape=%d' % ( depth, dim, points.shape[0] )

        node.dim = dim

        points_dim = points[ :, dim ]
    
        #masked_ids = ids[ mask ]
        #masked_points = points[ mask ]
    
        sorted_arg = numpy.argsort( points_dim )
        median_id = sorted_arg[ sorted_arg.shape[0] / 2 ]
        #abs_median_id = ids[ median_id ]
    
        node.value = points_dim[ median_id ]
    
        left_mask = points_dim < node.value
        right_mask = points_dim >= node.value
    
        if ( numpy.sum( left_mask ) == 0 ) or ( numpy.sum( right_mask ) == 0 ):

            print 'increasing max_depth for dimension %d' % dim
            max_depth += 1

            del node
            return __kd_tree( points, ids, depth+1, max_depth, partition )

        else:

            node.left = __kd_tree( points[ left_mask ], ids[ left_mask ], depth+1, max_depth, partition )
            node.right = __kd_tree( points[ right_mask ], ids[ right_mask ], depth+1, max_depth, partition )

    else:

        print ' depth=%d, left node' % depth

        print '  leaf node'

        partition_id = numpy.max( partition ) + 1

        node.id = partition_id

        partition[ ids ] = partition_id

    return node

def kd_tree(points, max_depth):

    partition = - numpy.ones( ( points.shape[0], ), dtype=int )

    ids = numpy.arange( points.shape[0] )

    tree = __kd_tree( points, ids, 0, max_depth, partition )

    return tree, partition


def __partition_along_kd_tree(node, points, ids, depth):

    dim = node.dim

    print ' depth=%d, dim=%d' % ( depth, dim )

    points_dim = points[ :, dim ]

    left_mask = points_dim < node.value
    right_mask = points_dim >= node.value
    
    if node.left != None and node.right != Nonde:

        __partition_along_kd_tree( node.left, points[ left_mask ], ids[ left_mask ], depth+1 )
        __partition_along_kd_tree( node.right, points[ right_mask ], ids[ right_mask ], depth+1 )

    else:

        partition[ ids ] = node.id

def partition_along_kd_tree(tree, points):

    partition = - numpy.ones( ( points.shape[0], ), dtype=int )

    ids = numpy.arange( points.shape[0] )

    __partition_along_kd_tree( tree, points, ids, 0)

    return partition

def sample_random_point(observations):
    min = numpy.min( observations, axis=0 )
    max = numpy.max( observations, axis=0 )
    point = numpy.empty( ( observations.shape[1], ) )
    for dim in xrange( observations.shape[1] ):
        point[ dim ] = numpy.random.uniform( min[ dim ], max[ dim ] )
    return point

def sample_random_points(observations, num_of_points):
    min = numpy.min( observations, axis=0 )
    max = numpy.max( observations, axis=0 )
    points = numpy.empty( ( num_of_points, observations.shape[1] ) )
    for dim in xrange( observations.shape[1] ):
        points[ :, dim ] = numpy.random.uniform( min[ dim ], max[ dim ], points.shape[0] )
    return points

def hcluster_special(method, points, param1, param2, param3, id ,minkowski_p):

    if method == 'seed':
        print 'using %d seeds' % param1
        Z = __cluster_hierarchical_seeds( points, param1, points.shape[0], minkowski_p )
    elif method in [ 'random' ]:
        c = sample_random_points( points, param1 )
        p = partition_along_clusters_kmeans( points, c )
        return p,c,None
    elif method in [ 'k-means', 'k-medians', 'k-medoids' ]:
        p,c,z = hcluster( method, points, param1, param2, id, minkowski_p )
        new_z = numpy.empty( ( z.shape[0], z.shape[1] + 2 ) )
        new_z[ :,:z.shape[1] ] = z
        z = new_z
        print z.shape
        return p,c,z
    elif method == 'kd-tree':
        tree,partition = kd_tree( points, param1 )
        num_of_leafs = numpy.max( partition ) + 1
        clusters = numpy.empty( ( num_of_leafs, points.shape[1] ) )
        clusters[:] = numpy.nan
        return partition, clusters, tree
    else:
        Z = do_hcluster( method, points, id, minkowski_p )

    new_Z = numpy.empty( ( Z.shape[0], Z.shape[1] + 4 ) )
    new_Z[:,:4] = Z
    Z = new_Z

    partition = numpy.arange( ( points.shape[0] ) )

    print 'scanning clustering (param2=%d, param3=%d)...' % ( param2, param3 )

    """devs = -1 * numpy.ones( ( 2*points.shape[0], ) )

    ESS = 0.0
    centroids = numpy.empty( ( 2*points.shape[0], points.shape[1] ) )
    centroids[ : points.shape[0] ] = points
    ESS_array = numpy.zeros( ( 2*points.shape[0], ) )
    for i in xrange( Z.shape[0] ):
        index1, index2, dist, count = Z[ i, :4 ]
        #if dist > dist_threshold:
        #    break
        new_index = i + points.shape[0]
        #print 'i=%d, index1=%d, index2=%d, new_index=%d, dist=%f, count=%d' % ( i, index1,index2,new_index,dist,count )
        mask1 = partition[:] == index1
        mask2 = partition[:] == index2
        mask = numpy.logical_or( mask1, mask2 )
        partition[ mask ] = new_index
        devs[ new_index ] = numpy.sum( numpy.std( points[ mask ], axis=0 ) ** 2 )
        devs[ index1 ] = 0.0
        devs[ index2 ] = 0.0
        dev = numpy.sum( devs )
        dev_mask = devs > 0.0
        dev = numpy.sum( devs[ dev_mask ] )
        dev_count = numpy.sum( dev_mask )
        Z[i,4] = dev
        if dev_count > 0:
            Z[i,5] = dev / float( dev_count )
        else:
            Z[i,5] = -1.0
        Z[i,6] = dev_count
        centroids[ new_index ] = ( numpy.sum( mask1 ) * centroids[ index1 ] + numpy.sum( mask2 ) * centroids[ index2 ] ) / numpy.sum( mask )
        dist = distance.minkowski_dist( centroids[ new_index ], points[ mask ] )
        dESS = - ESS_array[ index1 ] - ESS_array[ index2 ] + numpy.sum( dist ** 2 )
        ESS += dESS
        Z[i,4] = points.shape[0] - i - 1
        Z[i,5] = ESS ** 2
        Z[i,7] = ESS ** 2 * ( points.shape[0] - i - 1 )

    print 'Z[-10:,4]:', Z[-10:,4]"""

    partition = numpy.arange( ( points.shape[0] ) )
    part_of_big_cluster = numpy.zeros( ( points.shape[0], ), dtype=bool )
    num_of_big_clusters = 0
    num_of_clusters = points.shape[0]

    max_num_of_big_clusters = 0
    max_partition = None

    if param2 == -1:
        param2 = 1

    for i in xrange( Z.shape[0] ):
        index1, index2, dist, count = Z[ i, :4 ]
        #if dist > dist_threshold:
        #    break
        new_index = i + points.shape[0]
        #print 'i=%d, index1=%d, index2=%d, new_index=%d, dist=%f, count=%d' % ( i, index1,index2,new_index,dist,count )
        mask1 = partition[:] == index1
        mask2 = partition[:] == index2
        mask = numpy.logical_or( mask1, mask2 )

        if numpy.sum( mask ) != count:
            print 'count=%d   !=   numpy.sum(mask)=%d' % ( count, numpy.sum( mask ) )

        already_big_cluster1 = numpy.any( part_of_big_cluster[ mask1 ] )
        already_big_cluster2 = numpy.any( part_of_big_cluster[ mask2 ] )
        already_big_cluster = already_big_cluster1 or already_big_cluster2

        num_of_clusters -= 1

        prev_num_of_big_clusters = num_of_big_clusters

        if not already_big_cluster:
            if numpy.sum( mask ) >= param3:
                already_big_cluster = True
                num_of_big_clusters += 1
                print 'new:', numpy.sum( mask ), num_of_big_clusters, numpy.sum( mask1 ), numpy.sum( mask2 )

        if already_big_cluster:
            part_of_big_cluster[ mask ] = True

        if already_big_cluster1 and already_big_cluster2:
            num_of_big_clusters -= 1

        if num_of_big_clusters < prev_num_of_big_clusters:
            if prev_num_of_big_clusters > max_num_of_big_clusters and prev_num_of_big_clusters <= param2:
                max_partition = partition.copy()
                max_num_of_big_clusters = prev_num_of_big_clusters

        partition[ mask ] = new_index

        if num_of_big_clusters >= param2:
            print 'break <1> at num_of_big_clusters=%d, num_of_clusters=%d' % ( num_of_big_clusters, num_of_clusters )
            break

        if num_of_clusters <= param2:
            print 'break <2> at num_of_big_clusters=%d, num_of_clusters=%d' % ( num_of_big_clusters, num_of_clusters )
            break

        #if numpy.all( part_of_big_cluster ):
        #    break

    if max_num_of_big_clusters > num_of_big_clusters:
        partition = max_partition

    print 'done'

    print 'min=%d, max=%d' % ( numpy.min( partition ), numpy.max( partition ) )

    new_partition = -1 * numpy.ones( ( points.shape[0], ), dtype=int )
    """partition_ids = numpy.zeros( ( numpy.max( partition ) + 1, ), dtype=bool )
    n = 0
    for i in xrange( partition.shape[0] ):
        partition_id = partition[ i ]
        if not partition_ids[ partition_id ]:
            partition_ids[ partition_id ] = True
            mask = partition[:] == partition_id
            new_partition[ mask ] = n
            n += 1"""

    n = 0
    for i in xrange( numpy.max( partition ) + 1 ):
        mask = partition == i
        if numpy.sum( mask ) > 0:
            new_partition[ mask ] = n
            n += 1

    partition = new_partition

    print 'n=%d, min=%d, max=%d' % ( n, numpy.min( partition ), numpy.max( partition ) )

    clusters = compute_centroids_from_partition( points, partition )
    #clusters = numpy.zeros( ( numpy.max( partition ) + 1, points.shape[1] ) )

    print 'found %d clusters' % ( numpy.max( partition ) + 1 )

    print points.shape[0], n

    return partition, clusters, Z
    

def hcluster(method, points, param1, param2, id, minkowski_p=2):

    partition = None
    clusters = None
    Z = None

    if method == 'seed':
        print 'using %d seeds' % param1
        Z = __cluster_hierarchical_seeds( points, param1, points.shape[0], minkowski_p )
    elif method in [ 'k-means', 'k-medians', 'k-medoids' ]:
        partition,clusters,s = cluster( method, points, param1, minkowski_p )
    else:
        Z = do_hcluster( method, points, id, minkowski_p )

    if partition == None:

        if Z != None:
    
            new_Z = numpy.empty( ( Z.shape[0], Z.shape[1] + 3 ) )
            new_Z[:,:4] = Z
            Z = new_Z
    
            partition = numpy.arange( ( points.shape[0] ) )
    
            print 'scanning clustering (param2=%d)...' % param2
    
            devs = -1 * numpy.ones( ( 2*points.shape[0], ) )
    
            for i in xrange( Z.shape[0] ):
                index1, index2, dist, count = Z[ i, :4 ]
                #if dist > dist_threshold:
                #    break
                new_index = i + points.shape[0]
                #print 'i=%d, index1=%d, index2=%d, new_index=%d, dist=%f, count=%d' % ( i, index1,index2,new_index,dist,count )
                mask1 = partition[:] == index1
                mask2 = partition[:] == index2
                mask = numpy.logical_or( mask1, mask2 )
                partition[ mask ] = new_index
                devs[ new_index ] = numpy.sum( numpy.std( points[ mask ], axis=0 ) ** 2 )
                devs[ index1 ] = 0.0
                devs[ index2 ] = 0.0
                dev = numpy.sum( devs )
                dev_mask = devs > 0.0
                dev = numpy.sum( devs[ dev_mask ] )
                dev_count = numpy.sum( dev_mask )
                Z[i,4] = dev
                if dev_count > 0:
                    Z[i,5] = dev / float( dev_count )
                else:
                    Z[i,5] = -1.0
                Z[i,6] = dev_count
    
    
            partition = numpy.arange( ( points.shape[0] ) )
    
            if param2 == -1:
                param2 = Z.shape[0]
    
            for i in xrange( max( 0, Z.shape[0] - param2 + 1 ) ):
                index1, index2, dist, count = Z[ i, :4 ]
                #if dist > dist_threshold:
                #    break
                new_index = i + points.shape[0]
                #print 'i=%d, index1=%d, index2=%d, new_index=%d, dist=%f, count=%d' % ( i, index1,index2,new_index,dist,count )
                mask1 = partition[:] == index1
                mask2 = partition[:] == index2
                mask = numpy.logical_or( mask1, mask2 )
                partition[ mask ] = new_index
    
            print 'done'
    
            print 'min=%d, max=%d' % ( numpy.min( partition ), numpy.max( partition ) )
    
            new_partition = -1 * numpy.ones( ( points.shape[0], ), dtype=int )
            partition_ids = numpy.zeros( ( numpy.max( partition ) + 1, ), dtype=bool )
            n = 0
            for i in xrange( partition.shape[0] ):
                partition_id = partition[ i ]
                if not partition_ids[ partition_id ]:
                    partition_ids[ partition_id ] = True
                    mask = partition[:] == partition_id
                    new_partition[ mask ] = n
                    n += 1
    
            partition = new_partition
    
            print 'min=%d, max=%d' % ( numpy.min( partition ), numpy.max( partition ) )
    
            clusters = compute_centroids_from_partition( points, partition )
            #clusters = numpy.zeros( ( numpy.max( partition ) + 1, points.shape[1] ) )
    
            print 'found %d clusters' % ( numpy.max( partition ) + 1 )

    else:

        if Z == None:

            Z = numpy.empty( ( points.shape[0], 6 ) )
    
            n = 0
            prange = numpy.arange( points.shape[0] )
            for i in xrange( numpy.max( partition ) + 1 ):
                mask = partition == i
                count = numpy.sum( mask )
                #print count
                if count > 0:
                    l = i
                    masked_prange = prange[ mask ]
                    for j in xrange( 1, count ):
                        Z[ n, 0 ] = l
                        Z[ n, 1 ] = masked_prange[ j ]
                        Z[ n, 2 ] = distance.minkowski_dist( points[ prange[ j ] ], points[ prange[ 0 ] ] )
                        Z[ n, 3 ] = j + 1
                        l = n + points.shape[0]
                        n += 1

    print points.shape[0], n

    return partition, clusters, Z

def compute_centroids_from_partition(points, partition):

    num_of_centroids = numpy.max( partition ) + 1
    centroids = numpy.empty( ( num_of_centroids, points.shape[1] ) )

    for i in xrange( num_of_centroids ):
        mask = partition == i
        centroids[ i ] = compute_centroid( points[ mask ] )

    return centroids


def cluster(method, points, k, minkowski_p=2, calculate_silhouette=False, callback=None):

    if type( method ) == str:
        map = {
            'k-means' : CLUSTER_METHOD_KMEANS,
            'k-medians' : CLUSTER_METHOD_KMEDIANS,
            'k-medoids' : CLUSTER_METHOD_KMEDOIDS
        }
        method = map[ method ]

    if method == CLUSTER_METHOD_KMEANS:
        #p,c,w = cluster_kmeans_modified2( points, k, minkowski_p, 0, callback )
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

    #for i in xrange( k ):
    #    print 'cluster %d: %d objects' % ( i, numpy.sum( p[:] == i ) )

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


def __compute_feature_importance(points, point_mask, all_mads, all_mask):
    """Calculate the feature importance of a sub-population of cells

    Input parameters:
        - points: A nxm numpy array of all the samples.
          Rows are samples and columns are features.
        - point_mask: A 1-dimensional numpy array of length n masking
          the sub-population of interest.
        - all_mads: The median-absolute-deviation of each feature along all samples
        - all_mask: A mask of features that have a mad > 0
    Output parameter:
        A 1-dimensional numpy array of length m containing the importance of each feature
        for the sub-population."""

    # calculate weights

    importance = numpy.ones( ( points.shape[1], ) )

    number_of_points = numpy.sum( point_mask )

    if number_of_points > 1:

        medians = numpy.median( points[ point_mask ], axis=0 )
        mads = numpy.median( numpy.abs( points[ point_mask ] - medians ), axis=0 )

        mask = mads[:] > 0
        mask = numpy.logical_and( mask, all_mask )

        importance[ mask ] = 1 - mads[ mask ] / all_mads[ mask ]

        min_weight = numpy.min( importance[ mask ] )
        importance[ mask ] = importance[ mask ] - min_weight
    
        max_weight = numpy.max( importance[ mask ] )
        if max_weight > 0.0:
            importance[ mask ] = importance[ mask ] / max_weight

        importance[ numpy.invert( mask ) ] = -1.0

    else:

        importance[ : ] = 1.0

    return importance

def compute_feature_importance(points, point_mask):
    """Calculate the feature importance of a sub-population of cells

    Input parameters:
        - points: A nxm numpy array of all the samples.
          Rows are samples and columns are features.
        - point_mask: A 1-dimensional numpy array of length n masking
          the sub-population of interest.
    Output parameter:
        A 1-dimensional numpy array of length m containing the importance of each feature
        for the sub-population."""

    # calculate weights

    importance = numpy.ones( ( points.shape[1], ) )

    all_medians = numpy.median( points, axis=0 )
    all_mads = numpy.median( numpy.abs( points - all_medians ), axis=0 )

    all_mask = all_mads[:] > 0

    return __compute_feature_importance( points, point_mask, all_mads, all_mask )

def compute_feature_weighting(partition, points, clusters, minkowski_p=2):
    """Calculate the feature weighting within each cluster

    Input parameters:
        - partition: The partition of the samples along the clusters, as
          returned from a clustering routine
        - points: A nxm numpy array of the samples to be distributed.
          Rows are samples and columns are features.
        - clusters: A kxm numpy array of the clusters along which the
          samples should be distributed. Rows are clusters and
          columns are features.
        - minkowski_p: The power p of the minkowski metric,
          e.g. 2 for euclidean, 1 for city-block
    Output parameter:
        A kxm numpy array. Each row represents the feature weighting for one cluster."""

    # calculate weights

    weights = numpy.ones( clusters.shape )

    all_medians = numpy.median( points, axis=0 )
    all_mads = numpy.median( numpy.abs( points - all_medians ), axis=0 )

    all_mask = all_mads[:] > 0

    # update feature weights for each cluster
    for k in xrange( clusters.shape[0] ):

        #mask = all_stddevs_mask

        p_mask = partition[:] == k

        weights[ k ] = __compute_feature_importance( points, p_mask, all_mads, all_mask )

    return weights


def partition_along_clusters_kmeans( points, clusters, minkowski_p=2):

        # calculate the distance of all the samples to the k-th cluster centroid
        dist_m = distance.minkowski_cdist( clusters, points, minkowski_p )

        # find the cluster with the nearest centroid for each sample
        partition = numpy.argmin( dist_m, 0 )

        # return a one-dimensional numpy array of length n. For each sample the index
        # of the cluster with the nearest centroid is specified
        return partition

def partition_along_clusters(new_points, points, partition, clusters, weights=None, minkowski_p=2):
    """Partition the samples along the clusters so that for each sample
    the change in the Error summed square (ESS) is minimal
    
    Input parameters:
        - points: A nxm numpy array of the samples to be distributed.
          Rows are samples and columns are features.
        - clusters: A kxm numpy array of the clusters along which the
          samples should be distributed. Rows are clusters and
          columns are features.
        - minkowski_p: The power p of the minkowski metric,
          e.g. 2 for euclidean, 1 for city-block
    Output parameter:
        A one-dimensional numpy array of length n. For each sample the index
        of the cluster with the nearest centroid is specified."""

    """# calculate the distance of all the samples to the k-th cluster centroid
    if weights == None:
        dist_m = distance.minkowski_cdist( clusters, points, minkowski_p )
    else:
        dist_m = numpy.empty( ( clusters.shape[0], points.shape[0] ) )
        for k in xrange( clusters.shape[0] ):
            dist_m[ k ] = \
                    distance.weighted_minkowski_dist( clusters[k], points, weights[k], minkowski_p )

    # find the cluster with the nearest centroid for each sample
    partition = numpy.argmin( dist_m, 0 )

    # return a one-dimensional numpy array of length n. For each sample the index
    # of the cluster with the nearest centroid is specified
    return partition"""

    print 'partitioning along the clusters...'

    new_partition = numpy.empty( ( new_points.shape[0], ), dtype=int )

    CLUSTER_SIZES = numpy.empty( ( points.shape[0], ) )
    CENTROIDS = numpy.empty( points.shape )
    PMASKS = numpy.empty( ( clusters.shape[0], points.shape[0] ), dtype=bool )
    ESS = numpy.empty( ( clusters.shape[0], ) )

    for i in xrange( clusters.shape[0] ):
        pmask = partition == i
        PMASKS[ i ] = pmask
        CLUSTER_SIZES[ pmask ] = numpy.sum( pmask )
        CENTROIDS[ pmask ] = clusters[ i ]
        ess = points[ pmask ] - clusters[ i ]
        ess = numpy.sum( ess ** 2, axis=1 )
        ess = numpy.sum( ess )
        ESS[ i ] = ess

    NEW_CENTROIDS = numpy.empty( points.shape )
    NEW_ESS = numpy.empty( ( clusters.shape[0], ) )

    for i in xrange( new_points.shape[0] ):

        NEW_CENTROIDS = ( CLUSTER_SIZES * CENTROIDS.transpose() ).transpose() + new_points[ i ]
        NEW_CENTROIDS = ( NEW_CENTROIDS.transpose() / ( CLUSTER_SIZES + 1 ) ).transpose()
        ess = points - NEW_CENTROIDS
        ess += ( new_points[ i ] - NEW_CENTROIDS )
        ess = numpy.sum( ess ** 2, axis=1 )
        for j in xrange( clusters.shape[0] ):
            NEW_ESS[ j ] = numpy.sum( ess[ PMASKS[ j ] ] )
        DESS = NEW_ESS - ESS
        j = numpy.argmin( DESS )
        new_partition[ i ] = j

    return new_partition

def find_nearest_cluster(points, clusters, weights=None, minkowski_p=2):
    """For each sample find the nearest cluster centroid

    Input parameters:
        - points: A nxm numpy array of the samples to be distributed.
          Rows are samples and columns are features.
        - clusters: A kxm numpy array of the clusters along which the
          samples should be distributed. Rows are clusters and
          columns are features.
        - minkowski_p: The power p of the minkowski metric,
          e.g. 2 for euclidean, 1 for city-block
    Output parameter:
        A one-dimensional numpy array of length n. For each sample the index
        of the cluster with the nearest centroid is specified."""

    # calculate the distance of all the samples to the k-th cluster centroid
    if weights == None:
        dist_m = distance.minkowski_cdist( clusters, points, minkowski_p )
    else:
        dist_m = numpy.empty( ( clusters.shape[0], points.shape[0] ) )
        for k in xrange( clusters.shape[0] ):
            dist_m[ k ] = \
                    distance.weighted_minkowski_dist( clusters[k], points, weights[k], minkowski_p )

    # find the cluster with the nearest centroid for each sample
    partition = numpy.argmin( dist_m, 0 )

    # return a one-dimensional numpy array of length n. For each sample the index
    # of the cluster with the nearest centroid is specified
    return partition


def compute_intra_cluster_distances(partition, points, clusters, weights=None, minkowski_p=2):
    """Calculate the distances of the samples to their corresponding cluster centroid

    Input parameters:
        - partition: The partition of the samples along the clusters, as
          returned from a clustering routine
        - points: A nxm numpy array of the samples to be distributed.
          Rows are samples and columns are features.
        - clusters: A kxm numpy array of the clusters along which the
          samples should be distributed. Rows are clusters and
          columns are features.
        - minkowski_p: The power p of the minkowski metric,
          e.g. 2 for euclidean, 1 for city-block
    Output parameter:
        A one-dimensional numpy array of length n containing the distance
        to the corresponding cluster centroid for each sample"""

    # this will keep the distances to be computed
    distances = numpy.empty( ( points.shape[0], ) )

    for k in xrange( clusters.shape[0] ):


        # calculate the distance of all the samples to the k-th cluster centroid
        if weights == None:
            dist_m = distance.minkowski_dist( clusters[k], points, minkowski_p )
        else:
            dist_m = distance.weighted_minkowski_dist( clusters[k], points, weights[k], minkowski_p )
        #dist_m = distance.minkowski_cdist( [ clusters[k] ], points, minkowski_p )

        # create a mask for all the samples belonging to the k-th cluster
        partition_mask = partition[ : ] == k

        # copy the distances of the samples belonging to the k-th cluster into the distances-array
        distances[ partition_mask ] = dist_m[ 0 ][ partition_mask ]

    # return a one-dimensional numpy array of length n containing the distance
    # to the corresponding cluster centroid for each sample
    return distances


def compute_inter_cluster_distances(partition, points, clusters, weights=None, minkowski_p=2):
    """Calculate the pairwise distance between all clusters (this distance is not a metric!)

    Input parameters:
        - partition: The partition of the samples along the clusters, as
          returned from a clustering routine
        - points: A nxm numpy array of the samples to be distributed.
          Rows are samples and columns are features.
        - clusters: A kxm numpy array of the clusters along which the
          samples should be distributed. Rows are clusters and
          columns are features.
        - minkowski_p: The power p of the minkowski metric,
          e.g. 2 for euclidean, 1 for city-block
    Output parameter:
        A kxk numpy array. The i,j-th entry contains the distance of
        cluster i to cluster j. The distance is the mean of the distances
        from all the samples in cluster i to the centroid of cluster j."""

    print 'computing pairwise inter-cluster distances...'

    """###
    ### distance (i,j): mean of distance from all samples of i to their nearest neighbour in cluster j
    ###

    # this will keep the distances to be computed
    distances = numpy.empty( ( clusters.shape[0], clusters.shape[0] ) )

    for k in xrange( clusters.shape[0] ):

        # create a mask for all the samples belonging to the k-th cluster
        k_mask = partition[ : ] == k

        # for each cluster pair (k,l) compute the mean of the distance from the
        # samples of cluster k to their nearest neighbour in cluster j
        for l in xrange( clusters.shape[0] ):

            # create a mask for all the samples belonging to the l-th cluster
            l_mask = partition[ : ] == l

            k_points = points[ k_mask ]
            l_points = points[ l_mask ]

            # calculate the distances of all the samples in cluster k to all the samples in cluster l
            if minkowski_p == 2:
                dist_m = scipy.spatial.distance.cdist( k_points, l_points, 'euclidean' )
            elif minkowski_p == 1:
                dist_m = scipy.spatial.distance.cdist( k_points, l_points, 'cityblock' )
            else:
                dist_m = scipy.spatial.distance.cdist( k_points, l_points, 'minkowski', minkowski_p )

            # determine the nearest neighbours in cluster l
            dist = numpy.min( dist_m, axis=0 )

            # calculate the mean of the distances along the sample-axis, so we get
            # the mean distance to each nearest neighbour in the other cluster
            mean_dist = numpy.mean( dist )

            # copy the mean distance to the cluster centroid of cluster l into the distance-matrix
            distances[ k, l ] = mean_dist

            print '  (%d,%d) done' % ( k, l )"""

    """###
    ### distance (i,j): distance between the cluster centroids
    ###

    # calculate the distances of all the samples to all the cluster centroids
    if minkowski_p == 2:
        distances = scipy.spatial.distance.cdist( clusters, clusters, 'euclidean' )
    elif minkowski_p == 1:
        distances = scipy.spatial.distance.cdist( clusters, clusters, 'cityblock' )
    else:
        distances = scipy.spatial.distance.cdist( clusters, clusters, 'minkowski', minkowski_p )"""

    """###
    ### distance (i,j): ratio of the samples from i whose next nearest centroid is j
    ###

    # this will keep the distances to be computed
    distances = numpy.empty( ( clusters.shape[0], clusters.shape[0] ) )

    # calculate the distances of all the samples to all the cluster centroids
    if minkowski_p == 2:
        dist_m = scipy.spatial.distance.cdist( clusters, points, 'euclidean' )
    elif minkowski_p == 1:
        dist_m = scipy.spatial.distance.cdist( clusters, points, 'cityblock' )
    else:
        dist_m = scipy.spatial.distance.cdist( clusters, points, 'minkowski', minkowski_p )

    for k in xrange( clusters.shape[0] ):

        # create a mask for all the samples belonging to the k-th cluster
        partition_mask = partition[ : ] == k

        # create a mask for the l-th cluster
        cluster_mask = numpy.empty( ( clusters.shape[0], ), dtype=numpy.bool )
        cluster_mask[ : ] = True
        cluster_mask[ k ] = False

        # for each cluster pair (k,l) compute the mean of the distance from the
        # samples of cluster k to the centroid of cluster l
        for l in xrange( clusters.shape[0] ):

            if k == l:

                # set ratio in the distance-matrix
                distances[ k, l ] = 0.0

            else:

                # determine the cluster with the minimum distance for each sample (except for cluster k)
                m = dist_m[ cluster_mask ]
                m = m[ : , partition_mask ]
                min_indices = numpy.argmin( m, axis=0 )

                if l > k:
                    corrected_l = l + 1
                else:
                    corrected_l = l

                # compute the ratio of the samples from k whose next nearest centroid is l
                ratio = numpy.sum( min_indices[:] == corrected_l ) / float( numpy.sum( partition_mask ) )

                # copy the computed ratio to the distance-matrix
                distances[ k, l ] = ratio"""

    """###
    ### distance (i,j): mean of distance from all samples of i to centroid of j
    ###

    # this will keep the distances to be computed
    distances = numpy.empty( ( clusters.shape[0], clusters.shape[0] ) )

    # calculate the distances of all the samples to all the cluster centroids
    if weights == None:
        dist_m = distance.minkowski_cdist( clusters, points, minkowski_p )
    else:
        dist_m = numpy.empty( ( clusters.shape[0], points.shape[0] ) )
        for k in xrange( clusters.shape[0] ):
            dist_m[ k ] = \
                    distance.weighted_minkowski_dist( clusters[k], points, weights[k], minkowski_p )

    for k in xrange( clusters.shape[0] ):

        # create a mask for all the samples belonging to the k-th cluster
        partition_mask = partition[ : ] == k

        # for each cluster pair (k,l) compute the mean of the distance from the
        # samples of cluster k to the centroid of cluster l
        for l in xrange( clusters.shape[0] ):

            # create a mask for the l-th cluster
            cluster_mask = numpy.empty( ( clusters.shape[0], ), dtype=numpy.bool )
            cluster_mask[ : ] = False
            cluster_mask[ l ] = True

            # calculate the mean of the distances along the sample-axis, so we get
            # the mean distance to each cluster centroid
            mean_distance = numpy.mean( dist_m[ cluster_mask, partition_mask ] )

            # copy the mean distance to the cluster centroid of cluster l into the distance-matrix
            distances[ k, l ] = mean_distance

    # return a kxk numpy array. The i,j-th entry contains the distance of
    # cluster i to cluster j. The distance is the mean of the distances
    # from all the samples in cluster i to the centroid of cluster j.
    return distances"""

    # this will keep the distances to be computed
    distances = numpy.empty( ( clusters.shape[0], clusters.shape[0] ) )

    distances[ numpy.identity( distances.shape[0], dtype=bool ) ] = 0.0

    # Compute the Error summed square (ESS) for each cluster
    ESS = numpy.empty( ( clusters.shape[0], ) )
    PMASKS = numpy.empty( ( clusters.shape[0], points.shape[0] ), dtype=bool )
    CLUSTER_SIZES = numpy.empty( ( clusters.shape[0], ), dtype=int )
    for i in xrange( clusters.shape[0] ):
        ess = 0.0
        pmask = partition == i
        PMASKS[ i ] = pmask
        CLUSTER_SIZES[ i ] = numpy.sum( pmask )
        ess = points[ pmask ] - clusters[ i ]
        ess = numpy.sum( ess ** 2, axis=1 )
        ess = numpy.sum( ess )
        ESS[ i ] = ess

    # For each pair of clusters, compute the change in ESS when joining them
    for i in xrange( clusters.shape[0] ):
        for j in xrange( i ):
            pmask_i = PMASKS[ i ]
            pmask_j = PMASKS[ j ]
            pmask = numpy.logical_or( pmask_i, pmask_j )
            centroid = CLUSTER_SIZES[ i ] * clusters[ i ] + CLUSTER_SIZES[ j ] * clusters[ j ]
            centroid = centroid / float( CLUSTER_SIZES[i] + CLUSTER_SIZES[j] )
            ess = points[ pmask ] - centroid
            ess = numpy.sum( ess ** 2, axis=1 )
            ess = numpy.sum( ess )
            dess = ess - ESS[ i ] - ESS[ j ]
            distances[ i, j ] = distances[ j, i ] = dess

    return distances


def __choose_starting_seeds(points, seed_number=100, minkowski_p=2):

    seeds = []
    seed_indices = []

    index = numpy.random.randint( 0,points.shape[0] )
    seeds.append( points[ index ] )
    seed_indices.append( index )

    dist_to_seed = None

    for i in xrange( 1, seed_number ):

        dist_m = distance.minkowski_cdist( [ seeds[-1] ], points, minkowski_p )[ 0 ]

        if dist_to_seed == None:
            dist_to_seed = dist_m
        else:
            dist_to_seed = numpy.min( [ dist_m, dist_to_seed ], axis=0 )
        max_j = numpy.argmax( dist_to_seed )
        seeds.append( points[ max_j ] )
        seed_indices.append( max_j )

    seeds = numpy.array( seeds )
    seed_indices = numpy.array( seed_indices )

    return seeds, seed_indices


def __cluster_hierarchical_seeds(points, seed_number=100, objects_to_cluster=-1, minkowski_p=2):

    seeds = []

    partition = -1 * numpy.ones( points.shape[0], int )

    """seed_mapping = numpy.empty( ( points.shape[0], ), dtype=int )
    seed_mapping[:] = -1
    seed_indices = []

    for i in xrange( seed_number ):
        index = numpy.random.randint( 0, points.shape[0] )
        seed_mapping[ index ] = i
        seed_indices.append( index )
        seeds.append( points[ index ] )
        #partition[ index ] = i

    cluster_mapping = seed_mapping.copy()

    #seeds = numpy.array( seeds )
    seed_indices = numpy.array( seed_indices )"""

    seeds, seed_indices = __choose_starting_seeds( points, seed_number, minkowski_p )

    seed_mapping = numpy.empty( ( points.shape[0], ), dtype=int )
    seed_mapping[:] = -1
    for i in xrange( seed_indices.shape[0] ):
        seed_mapping[ seed_indices[ i ] ] = i

    cluster_mapping = seed_mapping.copy()

    prange = numpy.arange( points.shape[0] )

    dist_m = distance.minkowski_cdist( points, seeds, minkowski_p )
    for seed_i in xrange( seed_number ):
        seed_index = seed_indices[ seed_i ]
        dist_m[ seed_index, seed_i ] = numpy.inf

    nan_mask = numpy.isnan( dist_m )
    prange = numpy.arange( dist_m.shape[1] )
    print 'NaNs from cdist...'
    for row in xrange( dist_m.shape[0] ):
        nan_row_mask = nan_mask[ row ]
        if numpy.any( nan_row_mask ):
            columns = prange[ nan_row_mask ]
            print '  row %d:', columns


    seed_min_dist_m = numpy.min( dist_m, axis=1 )
    seed_min_arg = numpy.argmin( dist_m, axis=1 )

    print 'dist_m[0,:10]:', dist_m[0,:10]
    print 'dist_m[1,:10]:', dist_m[1,:10]
    print 'seed_min_dist_m[:10]', seed_min_dist_m[:10]
    print 'seed_min_arg[:10]', seed_min_arg[:10]

    print 'dist_m.shape:', dist_m.shape

    sorted_dist_arg = numpy.argsort( seed_min_dist_m )

    print 'sorted_dist_arg[:10]:', sorted_dist_arg[:10]
    print 'sorted_dist_m[:10]:', numpy.sort( seed_min_dist_m )[:10]

    cluster_sizes = numpy.ones( ( seeds.shape[0], ), dtype=int )

    #free_mask = partition[:] == -1

    #merged_seed_indices = []

    n = 0
    max_n = int( objects_to_cluster )
    if objects_to_cluster == -1:
        max_n = points.shape[0]

    Z = numpy.empty( ( max_n, 4 ) )

    while n < max_n:

        #min_index = sorted_dist_arg[ free_mask ][ 0 ]
        min_index = sorted_dist_arg[ n ]

        seed_i = seed_min_arg[ min_index ]
        seed_index = seed_indices[ seed_i ]
        mapped_seed_i = cluster_mapping[ seed_index ]

        #print 'n=%d, min_index=%d, seed_i=%d seeds_mapping[min_index]=%d, min_dist=%f' \
        #      % ( n, min_index, seed_i, seeds_mapping[min_index], seed_min_dist_m[min_index] )

        if min_index == seed_index:
            print '%d == %d' % ( min_index, seed_index )

        #partition[ min_index ] = mapped_seed_i
        #free_mask[ min_index ] = False
        #free_mask[ free_mask ][0] = False
        cluster_sizes[ seed_i ] += 1

        #Z[ n, 0 ] = seed_index
        #Z[ n, 1 ] = min_index

        Z[ n, 0 ] = cluster_mapping[ seed_index ]

        cluster_mapping[ seed_index ] = n + points.shape[0]

        mapped_seed_j = cluster_mapping[ min_index ]
        if mapped_seed_j >= 0 and min_index != seed_index:

            Z[ n, 1 ] = cluster_mapping[ min_index ]
            cluster_mapping[ min_index ] = n + points.shape[0]

            seed_j = seed_mapping[ min_index ]
            cluster_sizes[ seed_i ] += cluster_sizes[ seed_j ] - 1
            cluster_sizes[ seed_j ] = cluster_sizes[ seed_i ]

        else:

            Z[ n, 1 ] = min_index

        if not numpy.isinf( seed_min_dist_m[ min_index ] ):
            if seed_min_dist_m[ min_index ] == 0.0:
                print 'd(%d,%d) = 0.0' % ( min_index, seed_index )
            Z[ n, 2 ] = seed_min_dist_m[ min_index ]
        else:
            Z[ n, 2 ] = 0.0
            print 'd(%d,%d) = inf' % ( min_index, seed_index )
        Z[ n, 3 ] = cluster_sizes[ seed_i ]

        n += 1


    #print 'assigned:', numpy.sum( partition[:] >= 0 )

    return Z


def cluster_hierarchical_seeds(points, seed_number=100, objects_to_cluster=-1, minkowski_p=2):

    print 'Running hierarchical clustering with random seeding...'

    # we add a dump-seed later on, so correct the number of seeds
    seed_number = seed_number - 1

    seeds = []

    partition = -1 * numpy.ones( points.shape[0], int )

    seeds_mapping = numpy.empty( ( points.shape[0], ), dtype=int )
    seeds_mapping[:] = -1
    seed_indices = []

    for i in xrange( seed_number ):
        index = numpy.random.randint( 0, points.shape[0] )
        seeds_mapping[ index ] = i
        seed_indices.append( index )
        seeds.append( points[ index ] )
        #partition[ index ] = i

    #seeds = numpy.array( seeds )
    seed_indices = numpy.array( seed_indices )

    prange = numpy.arange( points.shape[0] )

    dist_m = distance.minkowski_cdist( points, seeds, minkowski_p )
    for seed_i in xrange( seed_number ):
        seed_index = seed_indices[ seed_i ]
        dist_m[ seed_index, seed_i ] = numpy.inf

    nan_mask = numpy.isnan( dist_m )
    prange = numpy.arange( dist_m.shape[1] )
    print 'NaNs from cdist...'
    for row in xrange( dist_m.shape[0] ):
        nan_row_mask = nan_mask[ row ]
        if numpy.any( nan_row_mask ):
            columns = prange[ nan_row_mask ]
            print '  row %d:', columns


    seed_min_dist_m = numpy.min( dist_m, axis=1 )
    seed_min_arg = numpy.argmin( dist_m, axis=1 )

    print 'dist_m[0,:10]:', dist_m[0,:10]
    print 'dist_m[1,:10]:', dist_m[1,:10]
    print 'seed_min_dist_m[:10]', seed_min_dist_m[:10]
    print 'seed_min_arg[:10]', seed_min_arg[:10]

    print 'dist_m.shape:', dist_m.shape

    sorted_dist_arg = numpy.argsort( seed_min_dist_m )

    print 'sorted_dist_arg[:10]:', sorted_dist_arg[:10]
    print 'sorted_dist_m[:10]:', numpy.sort( seed_min_dist_m )[:10]

    cluster_sizes = numpy.zeros( ( len( seeds ), ), dtype=int )
    biggest_cluster_size = 0

    #free_mask = partition[:] == -1

    merged_seed_indices = []

    n = 0
    max_n = int( objects_to_cluster )
    if objects_to_cluster == -1:
        max_n = points.shape[0]

    Z = numpy.empty( ( max_n, 4 ) )

    #while biggest_cluster_size < threshold_cluster_size:
    while n < max_n:

        #min_index = sorted_dist_arg[ free_mask ][ 0 ]
        min_index = sorted_dist_arg[ n ]

        seed_i = seed_min_arg[ min_index ]
        seed_index = seed_indices[ seed_i ]

        #print 'n=%d, min_index=%d, seed_i=%d seeds_mapping[min_index]=%d, min_dist=%f' \
        #      % ( n, min_index, seed_i, seeds_mapping[min_index], seed_min_dist_m[min_index] )

        if seeds_mapping[ min_index ] >= 0 and min_index != seed_index:
            # merge
            seed_j = seeds_mapping[ min_index ]
            #if i > j:
            #    tmp = i
            #    i = j
            #    j = tmp
            #partition[ partition[:] == seed_j ] = seed_i
            #seed_min_arg[ seed_min_arg[:] == seed_j ] = seed_i
            #seeds_mapping[ seed_j ] = seed_i
            merged_seed_indices.append( ( seed_i, seed_j ) )

        partition[ min_index ] = seed_i
        #free_mask[ min_index ] = False
        #free_mask[ free_mask ][0] = False
        cluster_sizes[ seed_i ] += 1
        if cluster_sizes[ seed_i ] > biggest_cluster_size:
            biggest_cluster_size = cluster_sizes[ seed_i ]

        Z[ n, 0 ] = seed_index
        Z[ n, 1 ] = min_index
        if not numpy.isinf( seed_min_dist_m[ min_index ] ):
            if seed_min_dist_m[ min_index ] == 0.0:
                print 'd(%d,%d) = 0.0' % ( min_index, seed_index )
            Z[ n, 2 ] = seed_min_dist_m[ min_index ]
        else:
            Z[ n + n_offset, 2 ] = 0.0
            print 'd(%d,%d) = inf' % ( min_index, seed_index )
        Z[ n, 3 ] = cluster_sizes[ seed_i ]

        n += 1


    print 'assigned:', numpy.sum( partition[:] >= 0 )

    for seed_i in xrange( seed_number ):
        seed_index = seed_indices[ seed_i ]
        partition[ seed_index ] = seed_i

    print 'assigned:', numpy.sum( partition[:] >= 0 )
    print 'wrongly assigned:', numpy.sum( partition[:] >= len( seeds ) )
    print 'unassigned:', numpy.sum( partition[:] < 0 )

    remove_seed_indices = []
    seed_remapping = numpy.arange( len( seeds ) )
    for seed_i,seed_j in merged_seed_indices:
        if seed_i == seed_j:
            'merging the same cluster!!!'
            sys.exit(1)
        if seed_i > seed_j:
            seed_tmp = seed_i
            seed_i = seed_j
            seed_j = seed_tmp
        #print 'merging seed cluster %d into seed cluster %d' % ( seed_j, seed_i )
        # now seed_i < seed_j
        partition[ partition[:] == seed_j ] = seed_i
        remove_seed_indices.append( seed_j )

    print 'assigned:', numpy.sum( partition[:] >= 0 )
    print 'wrongly assigned:', numpy.sum( partition[:] >= len( seeds ) )
    print 'unassigned:', numpy.sum( partition[:] < 0 )

    print len(seeds), numpy.max( partition ), numpy.min( partition)

    remove_seed_indices.sort()
    remove_seed_indices.reverse()
    tmp = []
    for seed_i in remove_seed_indices:
        if seed_i not in tmp:
            tmp.append( seed_i )
    remove_seed_indices = tmp
    print remove_seed_indices
    for seed_i in remove_seed_indices:
        del seeds[ seed_i ]
        partition[ partition[:] >= seed_i ] -= 1

    print len(seeds), numpy.max( partition ), numpy.min( partition)

    print 'assigned:', numpy.sum( partition[:] >= 0 )
    print 'wrongly assigned:', numpy.sum( partition[:] >= len( seeds ) )
    print 'unassigned:', numpy.sum( partition[:] < 0 )

    dump_mask = partition[:] < 0
    if numpy.any( dump_mask ):
        partition[ dump_mask ] = len( seeds )
        dump_seed = numpy.empty( ( points.shape[1], ) )
        dump_seed[:] = numpy.nan
        seeds.append( dump_seed )

    seeds = numpy.array( seeds )


    nan_mask = numpy.isnan( Z )
    prange = numpy.arange( Z.shape[1] )
    print 'NaNs from cdist...'
    for row in xrange( Z.shape[0] ):
        nan_row_mask = nan_mask[ row ]
        if numpy.any( nan_row_mask ):
            columns = prange[ nan_row_mask ]
            print '  row %d:', columns


    #for i in xrange( seeds.shape[0] ):
    #    print 'seed cluster %d: %d objects' % ( i, numpy.sum( partition[:] == i ) )

    return partition, seeds, Z


def compute_centroid(points):

    centroid = numpy.mean( points, axis=0 )
    return centroid

# Returns a kmeans-clustering of all the N observations.
# points is a NxM matrix whereas each row is an observation
# k is the number of clusters
# the clustering stops if less than swap_threshold swaps have
# been performed in an iteration
def cluster_kmeans_modified(points, k, minkowski_p=2, swap_threshold = 0, callback=None):

    print 'Running kmeans with feature selection...'

    clusters = []
    clusters.append( points[ numpy.random.randint( 0,points.shape[0] ) ] )

    partition = numpy.zeros( points.shape[0], int )

    dist_to_cluster = None

    for i in xrange(1,k):

        dist_m = distance.minkowski_cdist( [ clusters[-1] ], points, minkowski_p )[ 0 ]

        if dist_to_cluster == None:
            dist_to_cluster = dist_m
        else:
            dist_to_cluster = numpy.min( [ dist_m, dist_to_cluster ], axis=0 )
        max_j = numpy.argmax( dist_to_cluster )
        clusters.append( points[ max_j ] )

    clusters = numpy.array( clusters )

    weights = numpy.ones( clusters.shape )

    #print 'cluster_mean_count:'
    #print cluster_mean_count
    #print 'cluster_mean_count:'
    #print cluster_mean_count

    #print 'cluster_mean_count:'
    #print cluster_mean_count
    #print 'cluster_mean:'
    #print cluster_mean

    dist_to_clusters = numpy.empty( ( clusters.shape[0], points.shape[0] ) )

    #all_stddevs = numpy.std( points, axis=0 )
    #all_stddevs_mask = all_stddevs[:] > 0
    #inv_mask = numpy.invert( all_stddevs_mask )

    iterations = 0
    max_iterations = 150
    swaps = swap_threshold + 1
    while swaps > swap_threshold and iterations < max_iterations:

        if iterations % 1 == 0:
            if callback != None:
                #if not callback( iterations, swaps ):
                #    break
                callback( iterations, swaps )
            sys.stdout.write( '\riteration %d: swaps = %d ... ' % ( iterations, swaps ) )
            sys.stdout.flush()

        swaps = 0

        #dist_to_clusters = distance.minkowski_cdist( clusters, points,  minkowski_p )
        #dist_to_clusters = distance.weighted_minkowski_cdist( points, clusters, weights, minkowski_p )

        for k in xrange( clusters.shape[0] ):
            dist_to_clusters[ k ] = \
                    distance.weighted_minkowski_dist( clusters[k], points, weights[k], minkowski_p )

        """if minkowski_p == 2:
            dist_to_clusters = scipy.spatial.distance.cdist( points, clusters, 'euclidean' )
        elif minkowski_p == 1:
            dist_to_clusters = scipy.spatial.distance.cdist( points, clusters, 'cityblock' )
        else:
            dist_to_clusters = scipy.spatial.distance.cdist( points, clusters, 'minkowski', minkowski_p )"""

        min_index = numpy.argmin( dist_to_clusters, 0 )
        swaps = numpy.sum( min_index != partition )
        partition = min_index


        # update weights

        #print


        # update feature weights for each cluster
        for k in xrange( clusters.shape[0] ):

            #mask = all_stddevs_mask

            p_mask = partition[:] == k
            number_of_points = numpy.sum( p_mask )

            #stddevs = numpy.std( points[ p_mask ], axis=0)

            #print '  cluster %d (%d cells):' % ( k, numpy.sum( partition[:] == k ) )

            if number_of_points > 1:

                medians = numpy.median( points, axis=0 )
                mads = numpy.median( numpy.abs( points - medians ), axis=0 )

                mask = mads[:] > 0

                weights[ k ][ mask ] = mads
            
                min_weight = numpy.min( weights[ k ][ mask ] )
                weights[ k ][ mask ] = weights[ k ][ mask ] - min_weight
            
                max_weight = numpy.max( weights[ k ][ mask ] )
                if max_weight > 0.0:
                    weights[ k ][ mask ] = weights[ k ][ mask ] / max_weight

                weights[ k ][ numpy.invert( mask ) ] = 1.0

                """weights[ k ][ mask ] = 1 - stddevs[ mask ] / all_stddevs[ mask ]

                min_weight = numpy.min( weights[ k ][ mask ] )
                weights[ k ][ mask ] = weights[ k ][ mask ] - min_weight

                #mask = weights[ k ][ mask ] < 0.0
                #weights[ k ][ mask ] = 0.0

                max_weight = numpy.max( weights[ k ][ mask ] )
                if max_weight > 0.0:
                    weights[ k ][ mask ] = weights[ k ][ mask ]/ max_weight

                #print '    ', weights[k][:20]
                #print numpy.max( weights[ k ][ mask ] )"""

            else:

                weights[ k ][ : ] = 1.0

            #weights[ k ] = weights[ k ] ** 2

            #print numpy.min( weights[k][mask])
            #print numpy.max( weights[k][mask])

            #if numpy.sum( inv_mask ) > 0:
            #    for i in xrange( mask.shape[0] ):
            #        print 'i = %d' % i
            #    print 'inv_mask > 0: %d, k=%d' % ( numpy.sum( inv_mask ), k )

            tmp_mask = numpy.logical_or( weights[ k ] > 1.0, weights[ k ] < 0.0 )
            if numpy.sum( tmp_mask ) > 0:
                for i in xrange( tmp_mask.shape[0] ):
                    print 'i = %d' % i
                print 'mask > 1.0: %d, k=%d' % ( numpy.sum( tmp_mask ), k )


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

        #weights = numpy.ones( clusters.shape )

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

        if swaps <= swap_threshold:
            #print 'cluster_mean_count:'
            #print cluster_mean_count
            #print 'cluster_mean:'
            #print cluster_mean
            continue

        iterations += 1

    sys.stdout.write( 'done\n' )
    sys.stdout.flush()

    return partition, clusters, weights



# Returns a kmeans-clustering of all the N observations.
# points is a NxM matrix whereas each row is an observation
# k is the number of clusters
# the clustering stops if less than swap_threshold swaps have
# been performed in an iteration
def cluster_kmeans_modified2(points, k, minkowski_p=2, swap_threshold = 0, callback=None):

    print 'Running kmeans with feature selection...'

    clusters = []
    clusters.append( points[ numpy.random.randint( 0,points.shape[0] ) ] )

    partition = numpy.zeros( points.shape[0], int )

    dist_to_cluster = None


    for i in xrange(1,k):

        #dist_m = distance.weighted_minkowski_cdist( [ clusters[-1] ], points, weights, minkowski_p )
        dist_m = distance.minkowski_cdist( [ clusters[-1] ], points, minkowski_p )[ 0 ]

        if dist_to_cluster == None:
            dist_to_cluster = dist_m
        else:
            dist_to_cluster = numpy.min( [ dist_m, dist_to_cluster ], axis=0 )
        max_j = numpy.argmax( dist_to_cluster )
        clusters.append( points[ max_j ] )

    clusters = numpy.array( clusters )

    #print 'cluster_mean_count:'
    #print cluster_mean_count
    #print 'cluster_mean_count:'
    #print cluster_mean_count

    #print 'cluster_mean_count:'
    #print cluster_mean_count
    #print 'cluster_mean:'
    #print cluster_mean

    dist_to_clusters = numpy.empty( ( clusters.shape[0], points.shape[0] ) )


    weights = numpy.ones( ( points.shape[1], ) )

    medians = numpy.median( points, axis=0 )
    mads = numpy.median( numpy.abs( points - medians ), axis=0 )

    mask = mads[:] > 0

    weights[ mask ] = 1 - stddevs[ mask ] / all_stddevs[ mask ]

    min_weight = numpy.min( weights[ mask ] )
    weights[ mask ] = weights[ k ][ mask ] - min_weight

    #mask = weights[ k ][ mask ] < 0.0
    #weights[ k ][ mask ] = 0.0

    max_weight = numpy.max( weights[ k ][ mask ] )
    if max_weight > 0.0:
        weights[ k ][ mask ] = weights[ k ][ mask ]/ max_weight

    #print '    ', weights[k][:20]
    #print numpy.max( weights[ k ][ mask ] )

    weights[ mask ] = mads

    min_weight = numpy.min( weights[ mask ] )
    weights[ mask ] = weights[ mask ] - min_weight

    max_weight = numpy.max( weights[ mask ] )
    if max_weight > 0.0:
        weights[ mask ] = weights[ mask ] / max_weight

    weights[ numpy.invert( mask ) ] = 1.0

    tmp_mask = numpy.logical_or( weights > 1.0, weights < 0.0 )
    if numpy.sum( tmp_mask ) > 0:
        for i in xrange( tmp_mask.shape[0] ):
            print 'i = %d' % i
        print 'mask > 1.0: %d' % ( numpy.sum( tmp_mask ) )


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

        #dist_to_clusters = distance.minkowski_cdist( clusters, points,  minkowski_p )
        #dist_to_clusters = distance.weighted_minkowski_cdist( points, clusters, weights, minkowski_p )

        dist_to_clusters = distance.weighted_minkowski_cdist( clusters, points, weights, minkowski_p )

        #for k in xrange( clusters.shape[0] ):
        #    dist_to_clusters[ k ] = \
        #            distance.weighted_minkowski_dist( clusters[k], points, weights[k], minkowski_p )

        """if minkowski_p == 2:
            dist_to_clusters = scipy.spatial.distance.cdist( points, clusters, 'euclidean' )
        elif minkowski_p == 1:
            dist_to_clusters = scipy.spatial.distance.cdist( points, clusters, 'cityblock' )
        else:
            dist_to_clusters = scipy.spatial.distance.cdist( points, clusters, 'minkowski', minkowski_p )"""

        min_index = numpy.argmin( dist_to_clusters, 0 )
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

        #weights = numpy.ones( clusters.shape )

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

        if swaps <= swap_threshold:
            #print 'cluster_mean_count:'
            #print cluster_mean_count
            #print 'cluster_mean:'
            #print cluster_mean
            continue

        iterations += 1

    sys.stdout.write( 'done\n' )
    sys.stdout.flush()

    return partition, clusters, weights


# Returns a kmeans-clustering of all the N observations.
# points is a NxM matrix whereas each row is an observation
# k is the number of clusters
# the clustering stops if less than swap_threshold swaps have
# been performed in an iteration
def cluster_kmeans(points, k, minkowski_p=2, swap_threshold = 0, callback=None):

    random_index = numpy.random.randint( 0,points.shape[0] )

    clusters = []
    clusters.append( points[ random_index ] )

    partition = numpy.zeros( points.shape[0], int )

    dist_to_cluster = None

    for i in xrange(1,k):

        dist_m = distance.minkowski_cdist( [ clusters[-1] ], points, minkowski_p )[ 0 ]

        if dist_to_cluster == None:
            dist_to_cluster = dist_m
        else:
            dist_to_cluster = numpy.min( [ dist_m, dist_to_cluster ], axis=0 )
        max_j = numpy.argmax( dist_to_cluster )
        clusters.append( points[ max_j ] )

    clusters = numpy.array( clusters )

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

        dist_to_clusters = distance.minkowski_cdist( points, clusters, minkowski_p )

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

        dist_m = distance.minkowski_cdist( [ clusters[-1] ], points, minkowski_p )[ 0 ]

        if dist_to_cluster == None:
            dist_to_cluster = dist_m
        else:
            dist_to_cluster = numpy.min( [ dist_m, dist_to_cluster ], axis=0 )
        max_j = numpy.argmax( dist_to_cluster )
        clusters.append( points[ max_j ] )

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

        dist_m = distance.minkowski_cdist( [ clusters[-1] ], points, minkowski_p )[ 0 ]

        if dist_to_cluster == None:
            dist_to_cluster = dist_m
        else:
            dist_to_cluster = numpy.min( [ dist_m, dist_to_cluster ], axis=0 )
        max_j = numpy.argmax( dist_to_cluster )
        clusters.append( points[ max_j ] )

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
            sys.stdout.write( '\riteration %d ... ' % ( iterations ) )
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

