import numpy
import os


def create_path(filename):
    base = os.path.split( filename )[0]
    if not os.path.exists( base ):
        try:
            os.makedirs( base )
        except:
            pass
    if not os.path.isdir( base ):
        raise Exception( 'Not a directory: %s' % base )

def compute_treatment_similarity_map(profileHeatmap):

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

        str = '%s' % tr.name
        if i < len( treatments ) - 1:
            str += '\t'

        f.write( str )

    f.write( '\n' )

    for i in xrange( heatmap.shape[0] ):

        tr = treatments[ i ]

        str = '%s\t' % tr.name

        f.write( str )

        for j in xrange( heatmap.shape[1] ):

            str = '%f' % heatmap[ i, j ]
            if j < heatmap.shape[1] - 1:
                str += '\t'

            f.write( str )
        f.write( '\n' )

    if len( treatments ) > 14:

        f.write( '\n' )

        f.write( '\t' )
        for i in xrange( len( treatments ) ):

            if i % 2 != 0:
                continue

            tr = treatments[ i ]

            str = '%s' % tr.name
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

