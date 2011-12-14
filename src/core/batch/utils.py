# -*- coding: utf-8 -*-

"""
utils.py -- Utility functions for the headless mode.
"""

# This software is distributed under the FreeBSD License.
# See the accompanying file LICENSE for details.
# 
# Copyright 2011 Benjamin Hepp

import numpy
import os
from .. import distance

def compute_edf(observations):
    edf = numpy.copy( observations )
    edf = edf.transpose()
    edf.sort( axis=1 )
    return edf

def compute_edfs(pipeline, masks, points):
    print 'compute_edfs()'

    edfs = []

    for mask in masks:
        p = points[ mask ]
        supports = numpy.copy( p )
        supports = numpy.transpose( supports )
        supports.sort( axis=1 )
        edfs.append( supports )

    return edfs

def get_unique_sorted_array(array, sorted=False):
    if not sorted:
        array = numpy.sort( array )
    mask = numpy.ones( ( array.shape[0], ), dtype=bool )
    v1 = array[0]
    for i in xrange( 1, array.shape[0] ):
        v2 = array[ i ]
        if v1 == v2:
            mask[ i ] = False
            #print i, v1, v2
        v1 = v2
    return array[ mask ]

def compute_edf_distance(support1, support2):

    bin_edges = numpy.empty( ( support1.shape[0] + support2.shape[0] + 2, ) )
    bin_edges[ 0 ] = -numpy.inf
    bin_edges[ -1 ] = numpy.inf
    bin_edges[ 1 : 1 + support1.shape[0] ] = support1
    bin_edges[ 1 + support1.shape[0] : 1 + support1.shape[0] + support2.shape[0] ] = support2
    bin_edges = numpy.sort( bin_edges )
    #print bin_edges.shape
    #print bin_edges
    bin_edges = numpy.unique( bin_edges )
    #bin_edges = get_unique_sorted_array( bin_edges, True )
    #print bin_edges.shape
    #print bin_edges

    #print support1.shape
    #print support2.shape

    bin_counts1_i,bins1 = numpy.histogram( support1, bin_edges )
    bin_counts2_i,bins2 = numpy.histogram( support2, bin_edges )

    bin_counts1 = numpy.float_( bin_counts1_i )
    bin_counts2 = numpy.float_( bin_counts2_i )

    sum_counts1 = numpy.cumsum( bin_counts1 ) / numpy.sum( bin_counts1 )
    sum_counts2 = numpy.cumsum( bin_counts2 ) / numpy.sum( bin_counts2 )

    delta = numpy.abs( sum_counts1 - sum_counts2 )

    dist = numpy.max( delta )

    return dist
    

"""i1 = 0
i2 = 0
supremum = 0.0
while ( i1 < support1.shape[0] ) or ( i2 < support2.shape[0] ):
    if i1 < support1.shape[0]:
        x1 = support1[ i1 ]
    else:
        x1 = None
    if i2 < support2.shape[0]:
        x2 = support2[ i2 ]
    else:
        x2 = None
    if x1 == None or x2 == None:
        i1 += 1
        i2 += 1
    else:
        if x1 < x2:
            i1 += 1
        elif x2 < x1:
            i2 += 1
        else:
            i1 += 1
            i2 += 1
    supremum = abs( i1 - i2 )
    v = abs( i1 - i2 )
    if v > supremum:
        supremum = v
return supremum"""

def compute_edf_weighting(edfs):
    print 'compute_edf_weighting()'
    distMatrix = numpy.empty( ( edfs[0].shape[0], len( edfs ), len( edfs ) ), )

    for k in xrange( edfs[0].shape[0] ):

        for i in xrange( len( edfs ) ):
            supports1 = edfs[ i ]

            distMatrix[ k,i,i ] = 0.0

            for j in xrange( i+1, len( edfs ) ):
                supports2 = edfs[ j ]

                dist = compute_edf_distance( supports1[k], supports2[k] )
                distMatrix[ k,i,j ] = dist
                distMatrix[ k,j,i ] = dist

    dist = numpy.empty( ( distMatrix.shape[0], ) )

    for k in xrange( distMatrix.shape[0] ):
        dist[k] = numpy.mean( distMatrix[k][ numpy.invert( numpy.identity( distMatrix.shape[1], dtype=bool ) ) ] )

    max = numpy.max( dist )
    weighting = max - dist
    weighting = weighting / numpy.sqrt( numpy.sum( weighting ** 2 ) )

    #mask = dist > 0.0
    #weighting = numpy.empty( ( dist.shape[0], ) )
    #weighting[ mask ] = 1.0 / dist[ mask ]
    #weighting[ numpy.invert( mask ) ] = numpy.max( weighting[ mask ] )

    return weighting

def compute_edf_distances(edfs, weights=None):
    print 'compute_edf_distances()'
    distMatrix = numpy.empty( ( len( edfs ), len( edfs ) ), )
    dist = numpy.empty( ( edfs[0].shape[0], ) )
    for i in xrange( len( edfs ) ):
        print '  mask %d' % i
        supports1 = edfs[ i ]
        for j in xrange( len( edfs ) ):
            supports2 = edfs[ j ]
            for k in xrange( supports1.shape[0] ):
                dist[k] = compute_edf_distance( supports1[k], supports2[k] )
            if weights != None:
                wdist = weights * dist
            else:
                wdist = dist
            distMatrix[i,j] = numpy.sqrt( numpy.sum( wdist**2 ) )
    return distMatrix

#def compute_edf_weighting(edfs):
#    pass


def create_path(filename):
    base = os.path.split( filename )[0]
    if not os.path.exists( base ):
        try:
            os.makedirs( base )
        except:
            pass
    if not os.path.isdir( base ):
        raise Exception( 'Not a directory: %s' % base )


def compute_cluster_similarity_matrix(clusters, minkowski_p=2.0):

    dist_m = distance.minkowski_cdist(clusters, clusters, minkowski_p)
    sim_m = numpy.exp(-dist_m)
    return sim_m
    """max_dist = numpy.max( dist )
    if max_dist > 0.0:
        sim = 1.0 - dist / max_dist
    else:
        sim = 1.0 - dist
    return sim"""
    """sim = 1.0 / dist
    finite_mask = numpy.isfinite( sim )
    max_sim = numpy.max( sim[ finite_mask ] )
    sim /= max_sim
    sim[ numpy.invert( finite_mask ) ] = 1.0
    return sim"""
    cluster_dist_m = distance.minkowski_cdist(clusters, clusters, minkowski_p)
    min_cluster_dist = numpy.min(cluster_dist_m[
            numpy.invert(numpy.identity(cluster_dist_m.shape[0], dtype=bool))
    ])
    sim = numpy.exp(-cluster_dist_m/min_cluster_dist)
    return sim

def compute_treatment_similarity_map(distanceMap, profile_metric):

    if profile_metric == 'summed_minmax':
        return 1.0 - distanceMap
    elif profile_metric == 'l2_norm' or profile_metric == 'chi2_norm' or profile_metric == 'l1_norm' or profile_metric == 'chi1_norm' or profile_metric == 'quadratic_chi':
        masked_map = distanceMap[ numpy.isfinite( distanceMap ) ]
        if masked_map.shape[0] == 0:
            return distanceMap
        else:
            max_v = numpy.max( masked_map )
            #print 'max_v:', max_v
            #print 'distanceMap/max_v:', ( distanceMap / max_v )
            if max_v > 0.0:
                return 1.0 - distanceMap / max_v
            else:
                return 1.0 - distanceMap
    else:
        raise Exception( 'No such profile metric: %s' % profile_metric )


def compute_treatment_distance_map(clusterProfiles, profile_metric, profile_threshold=0.0, binSimilarityMatrix=None, normalizationFactor=1.0):

    profileHeatmap = numpy.zeros( ( clusterProfiles.shape[0], clusterProfiles.shape[0] ) )
    profileHeatmap[ numpy.identity( profileHeatmap.shape[0], dtype=bool ) ] = 0.0

    for i in xrange( clusterProfiles.shape[0] ):

        profile1 = clusterProfiles[ i ]
        norm_profile1 = profile1 / float( numpy.sum( profile1 ) )

        if profile_threshold > 0.0:
            max = numpy.max( profile1 )
            threshold_mask = profile1 < max * profile_threshold
            norm_profile1[ threshold_mask ] = 0.0
            norm_profile1 = norm_profile1 / float( numpy.sum( norm_profile1 ) )

        for j in xrange( i ):

            profile2 = clusterProfiles[ j ]
            norm_profile2 = profile2 / float( numpy.sum( profile2 ) )

            if profile_threshold > 0.0:
                max = numpy.max( profile2 )
                threshold_mask = profile2 < max * profile_threshold
                norm_profile2[ threshold_mask ] = 0.0
                norm_profile2 = norm_profile2 / float( numpy.sum( norm_profile2 ) )

            if profile_metric == 'summed_minmax':

                min_match = numpy.sum( numpy.min( [ norm_profile1, norm_profile2 ], axis=0 )**2 )
                max_match = numpy.sum( numpy.max( [ norm_profile1, norm_profile2 ], axis=0 )**2 )

                match = min_match / max_match

                dist = 1.0 - match

            elif profile_metric == 'l2_norm':

                # L2-norm
                dist = numpy.sqrt( numpy.sum( ( norm_profile1 - norm_profile2 ) ** 2 ) )

            elif profile_metric == 'l1_norm':

                # L1-norm
                dist = numpy.sum( numpy.abs( norm_profile1 - norm_profile2 ) )

            elif profile_metric == 'chi1_norm':

                # Chi1
                dist =  numpy.abs( norm_profile1 - norm_profile2 ) / ( norm_profile1 + norm_profile2 )
                dist[ numpy.logical_and( norm_profile1 == 0, norm_profile2 == 0 ) ] = 0.0

                dist = numpy.sum( dist )

            elif profile_metric == 'chi2_norm':

                # chi-square
                dist =  ( norm_profile1 - norm_profile2 ) ** 2 / ( norm_profile1 + norm_profile2 )
                dist[ numpy.logical_and( norm_profile1 == 0, norm_profile2 == 0 ) ] = 0.0

                dist = numpy.sum( dist )

            elif profile_metric == 'quadratic_chi':

                # quadratic-chi
                A = binSimilarityMatrix
                m = normalizationFactor

                P = norm_profile1
                Q = norm_profile2

                """QC = numpy.empty( ( clusterProfiles.shape[0], clusterProfiles.shape[0] ) )
                N = ( P - Q )
                D = numpy.sum( (P+Q)*numpy.transpose(A), axis=1 )
                mask = D == 0.0
                N[ mask ] = 0.0
                D[ mask ] = 1.0
                T = N / D
                B = numpy.matrix( T )
                BB = numpy.dot( numpy.transpose( B ), B )
                dist = numpy.sum( BB * A )"""

                D = numpy.empty((P.shape[0],))
                for k in xrange( P.shape[0] ):
                    D[k] = 0.0
                    for c in xrange(P.shape[0]):
                        D[k] += ( ( P[c] + Q[c] ) * A[c,k] ) ** m
                N = numpy.empty((P.shape[0],))
                for k in xrange( P.shape[0] ):
                    N[k] = P[k] - Q[k]
                zero_mask = D == 0.0
                D[ zero_mask ] = 1.0
                N[ zero_mask ] = 0.0
                M = 0.0
                for k in xrange(N.shape[0]):
                    for l in xrange(N.shape[0]):
                        M += ( N[k] / D[k] ) * ( N[l] / D[l] ) * A[k,l]
                dist = M

            else:
                raise Exception( 'No such profile metric: %s' % profile_metric )

            profileHeatmap[ i, j ] = dist
            profileHeatmap[ j, i ] = dist

    return profileHeatmap


def compute_modified_treatment_similarity_map(profileHeatmap):

    """threshold = batch_utils.compute_profile_heatmap_threshold( profileHeatmap )
    print 'similarity_threshold=%f' % threshold

    #treatmentSimilarityMap = numpy.zeros( ( num_of_treatments/2, num_of_treatments/2 ), dtype=bool )
    #treatmentSimilarityMap[ numpy.identity( treatmentSimilarityMap.shape[0], dtype=bool ) ] = True

    for i in xrange( 0, num_of_treatments, 2 ):
        for j in xrange( i+2, num_of_treatments, 2 ):
            #v1 = profileHeatmap[ fullTreatmentMask ][ i, j ]
            #v2 = profileHeatmap[ fullTreatmentMask ][ i+1, j ]
            #v3 = profileHeatmap[ fullTreatmentMask ][ i, j+1 ]
            #v4 = profileHeatmap[ fullTreatmentMask ][ i+1, j+1 ]
            v_mean = numpy.mean( profileHeatmap[ i:i+2, j:j+2 ] )
            print 'i/2=%d, j/2=%d, v_mean=%f' % ( i, j, v_mean )
            if v_mean <= threshold:
                treatmentSimilarityMap[ i/2, j/2 ] = True
                treatmentSimilarityMap[ j/2, i/2 ] = True"""

    heatmap = profileHeatmap.copy()
    for i in xrange( 0, heatmap.shape[0], 2 ):
        heatmap[ i, i ] = heatmap[ i+1, i+1 ] = heatmap[ i, i+1 ]

    #nan_mask = numpy.isnan( profileHeatmap )
    #for i in xrange( 0, profileHeatmap.shape[0], 2 ):
    #    profileHeatmap[ i, i ] = profileHeatmap[ i, i+1 ]
    #    profileHeatmap[ i+1, i+1 ] = profileHeatmap[ i, i+1 ]

    self_match = heatmap[ numpy.identity( heatmap.shape[0], dtype=bool ) ]
    #self_match = numpy.empty( ( heatmap.shape[0], ) )
    #for i in xrange( 0, heatmap.shape[0], 2 ):
    #    self_match[ i ] = self_match[ i + 1 ] = heatmap[ i, i+1 ]

    #max_self_match = numpy.max( self_match )
    #min_self_match = numpy.min( self_match )

    similarityMap = numpy.empty( ( heatmap.shape[0]/2, heatmap.shape[0]/2 ) )

    for i in xrange( 0, heatmap.shape[0], 2 ):
        for j in xrange( i, heatmap.shape[1], 2 ):
            #v1 = profileHeatmap[ fullTreatmentMask ][ i, j ]
            #v2 = profileHeatmap[ fullTreatmentMask ][ i+1, j ]
            #v3 = profileHeatmap[ fullTreatmentMask ][ i, j+1 ]
            #v4 = profileHeatmap[ fullTreatmentMask ][ i+1, j+1 ]
            v_mean = numpy.mean( heatmap[ i:i+2, j:j+2 ] )
            print 'i/2=%d, j/2=%d, v_mean=%f' % ( i, j, v_mean )
            #similarityMap[ i/2, j/2 ] = similarityMap[ j/2, i/2 ] = ( v_mean - min_self_match ) / max_self_match
            #divider = max( self_match[ i ], self_match[ j ] )
            #similarityMap[ i/2, j/2 ] = ( v_mean / divider )
            #result = v_mean * self_match[ i ] * self_match[ j ]
            min_self_match = min( self_match[i], self_match[j] )
            max_self_match = max( self_match[i], self_match[j] )
            if max_self_match == 0.0:
                min_self_match = max_self_match = 1.0
            result = v_mean * min_self_match / max_self_match
            similarityMap[ i/2, j/2 ] = similarityMap[ j/2, i/2 ] = result
            #similarityMap[ i/2, j/2 ] = ( v_mean / self_match[ i ] ) * self_match[ j ]
            #similarityMap[ j/2, i/2 ] = ( v_mean / self_match[ i ] ) * self_match[ j ]

    #similarityMap = ( profileHeatmap - min_self_match ) / max_self_match
    #similarityMap[ nan_mask ] = 0.0

    return similarityMap


def compute_profile_heatmap_threshold(profileHeatmap):
    max_value = -1.0
    max_index = -1
    for i in xrange( 0, profileHeatmap.shape[0], 2 ):
        if profileHeatmap[ i, i+1 ] > max_value:
            max_value = profileHeatmap[ i, i+1 ]
            max_index = i
    print 'found max_value=%f for max_index=%d' % ( max_value, max_index )
    return 2.0 * max_value


def write_profileHeatmapCSV(title, treatments, heatmap, filename):

    f = open( filename, 'w' )

    f.write( title + '\n' )

    f.write( '\t' )
    for i in xrange( len( treatments ) ):

        tr = treatments[ i ]

        str = '%s' % tr
        if i < len( treatments ) - 1:
            str += '\t'

        f.write( str )

    f.write( '\n' )

    for i in xrange( heatmap.shape[0] ):

        tr = treatments[ i ]

        str = '%s\t' % tr

        f.write( str )

        for j in xrange( heatmap.shape[1] ):

            str = '%f' % heatmap[ i, j ]
            if j < heatmap.shape[1] - 1:
                str += '\t'

            f.write( str )
        f.write( '\n' )

    """if len( treatments ) > 14:

        f.write( '\n' )

        f.write( '\t' )
        for i in xrange( len( treatments ) ):

            if i % 2 != 0:
                continue

            tr = treatments[ i ]

            str = '%s' % tr
            if i < len( treatments ) - 1:
                str += '\t'

            f.write( str )

        f.write( '\n' )

        f.write( 'self match\t' )
        for i in xrange( heatmap.shape[0] ):

            if i % 2 != 0:
                continue

            str = '%f' % heatmap[ i, i+1 ]
            if i < heatmap.shape[0] - 2:
                str += '\t'

            f.write( str )

        f.write( '\n' )

        f.write( 'median to others\t' )
        for i in xrange( heatmap.shape[0] ):

            if i % 2 != 0:
                continue

            mask = numpy.ones( ( heatmap.shape[0], ), dtype=bool )
            mask[ i ] = False
            mask[ i+1 ] = False
            mask2 = numpy.any( numpy.isnan( heatmap[ [i,i+1] ] ), axis=0 )
            mask = numpy.logical_and( mask, numpy.invert( mask2 ) )

            med = numpy.median( heatmap[ [i,i+1] ][ :, mask ] )

            str = '%f' % med
            if i < heatmap.shape[0] - 2:
                str += '\t'

            f.write( str )

        f.write( '\n' )

        f.write( 'mean to others\t' )
        for i in xrange( heatmap.shape[0] ):

            if i % 2 != 0:
                continue

            mask = numpy.ones( ( heatmap.shape[0], ), dtype=bool )
            mask[ i ] = False
            mask[ i+1 ] = False
            mask2 = numpy.any( numpy.isnan( heatmap[ [i,i+1] ] ), axis=0 )
            mask = numpy.logical_and( mask, numpy.invert( mask2 ) )

            med = numpy.mean( heatmap[ [i,i+1] ][ :, mask ] )

            str = '%f' % med
            if i < heatmap.shape[0] - 2:
                str += '\t'

            f.write( str )

        f.write( '\n' )

        f.write( 'min to others\t' )
        for i in xrange( heatmap.shape[0] ):

            if i % 2 != 0:
                continue

            mask = numpy.ones( ( heatmap.shape[0], ), dtype=bool )
            mask[ i ] = False
            mask[ i+1 ] = False
            mask2 = numpy.any( numpy.isnan( heatmap[ [i,i+1] ] ), axis=0 )
            mask = numpy.logical_and( mask, numpy.invert( mask2 ) )

            try:
                min = numpy.min( heatmap[ [i,i+1] ][ :, mask ] )
            except:
                min = numpy.nan

            str = '%f' % min
            if i < heatmap.shape[0] - 2:
                str += '\t'

            f.write( str )

        f.write( '\n' )

        f.write( 'max to others\t' )
        for i in xrange( heatmap.shape[0] ):

            if i % 2 != 0:
                continue

            mask = numpy.ones( ( heatmap.shape[0], ), dtype=bool )
            mask[ i ] = False
            mask[ i+1 ] = False
            mask2 = numpy.any( numpy.isnan( heatmap[ [i,i+1] ] ), axis=0 )
            mask = numpy.logical_and( mask, numpy.invert( mask2 ) )

            try:
                max = numpy.max( heatmap[ [i,i+1] ][ :, mask ] )
            except:
                max = numpy.nan

            str = '%f' % max
            if i < heatmap.shape[0] - 2:
                str += '\t'

            f.write( str )

        f.write( '\n' )

        f.write( 'min to others - self match\t' )
        for i in xrange( heatmap.shape[0] ):

            if i % 2 != 0:
                continue

            self_match = heatmap[ i, i+1 ]

            mask = numpy.ones( ( heatmap.shape[0], ), dtype=bool )
            mask[ i ] = False
            mask[ i+1 ] = False
            mask2 = numpy.any( numpy.isnan( heatmap[ [i,i+1] ] ), axis=0 )
            mask = numpy.logical_and( mask, numpy.invert( mask2 ) )

            try:
                min = numpy.min( heatmap[ [i,i+1] ][ :, mask ] )
            except:
                min = numpy.nan

            str = '%f' % ( min - self_match )
            if i < heatmap.shape[0] - 2:
                str += '\t'

            f.write( str )

        f.write( '\n' )

        f.write( 'self match - max to others\t' )
        for i in xrange( heatmap.shape[0] ):

            if i % 2 != 0:
                continue

            self_match = heatmap[ i, i+1 ]

            mask = numpy.ones( ( heatmap.shape[0], ), dtype=bool )
            mask[ i ] = False
            mask[ i+1 ] = False
            mask2 = numpy.any( numpy.isnan( heatmap[ [i,i+1] ] ), axis=0 )
            mask = numpy.logical_and( mask, numpy.invert( mask2 ) )

            try:
                max = numpy.min( heatmap[ [i,i+1] ][ :, mask ] )
            except:
                max = numpy.nan

            str = '%f' % ( self_match - max )
            if i < heatmap.shape[0] - 2:
                str += '\t'

            f.write( str )

        f.write( '\n' )
    """

    f.close()


def write_similarityMapCSV(title, labels, map, filename):

    f = open( filename, 'w' )

    f.write( title + '\n' )

    f.write( '\t' + '\t'.join( labels ) + '\n' )

    for i in xrange( map.shape[0] ):

        str = '%s\t' % labels[ i ]

        f.write( str )

        for j in xrange( map.shape[1] ):

            str = '%f' % map[ i, j ]
            if j < map.shape[1] - 1:
                str += '\t'

            f.write( str )

        f.write( '\n' )

    f.close()

