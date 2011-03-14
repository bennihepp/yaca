import numpy
import subprocess
import sys

import matplotlib.colors
import matplotlib.patches
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.transforms as mtransforms


FILE_VIEWER = 'evince'

def view_file(filename):
    subprocess.Popen( [ FILE_VIEWER, filename ] )


def print_clustering_plot(pipeline, outputFilename, mpl_kwargs={}):

    be = plt.get_backend()
    plt.switch_backend( 'PDF' )

    pp = PdfPages( outputFilename )

    if not mpl_kwargs.has_key( 'facecolor' ): mpl_kwargs[ 'facecolor' ] = 'green'
    if not mpl_kwargs.has_key( 'alpha' ): mpl_kwargs[ 'alpha' ] = 0.75
    if not mpl_kwargs.has_key( 'align' ): mpl_kwargs[ 'align' ] = 'center'

    NUM_OF_ROWS = 1
    NUM_OF_COLS = 1
    LEFT_MARGIN = 0.1
    RIGHT_MARGIN = 0.1
    HORIZONTAL_SPACING = 0.1
    BOTTOM_MARGIN = 0.15
    TOP_MARGIN = 0.1
    VERTICAL_SPACING = 0.15
    WIDTH = 1.0 - LEFT_MARGIN - RIGHT_MARGIN
    HEIGHT = 1.0 - TOP_MARGIN - BOTTOM_MARGIN
    SUBPLOT_WIDTH = ( WIDTH - ( NUM_OF_COLS - 1 ) * HORIZONTAL_SPACING ) / float( NUM_OF_COLS )
    SUBPLOT_HEIGHT = ( HEIGHT - ( NUM_OF_ROWS - 1 ) * VERTICAL_SPACING ) / float( NUM_OF_ROWS )


    labels = []
    cluster_population = numpy.empty( ( pipeline.nonControlClusters.shape[0], ) )
    for i in xrange( cluster_population.shape[0] ):
        pmask = pipeline.nonControlPartition == i
        cluster_population[ i ] = numpy.sum( pmask )
        labels.append( '%d' % i )

    n = 0
    fig = plt.figure()
    header_text = fig.suptitle( 'Clusters' )
    #footer_text = fig.text( 0.5, 0.02, ' Page %d' % m, horizontalalignment='center', verticalalignment='bottom' )
    #if bottom_shift != None:
    #    fig.subplots_adjust( bottom=bottom_shift )

    #axes = plt.subplot( num_of_rows, num_of_cols, n )


    row = int( n / NUM_OF_COLS )
    col = int( n % NUM_OF_COLS )
    left = LEFT_MARGIN + col * ( SUBPLOT_WIDTH + HORIZONTAL_SPACING )
    bottom = BOTTOM_MARGIN + row * ( SUBPLOT_HEIGHT + VERTICAL_SPACING )
    axes = fig.add_axes( [ left, bottom, SUBPLOT_WIDTH, SUBPLOT_HEIGHT ], frameon=True )
    n += 1

    x = numpy.arange( 0, len( labels ) )
    axes.set_xticks( x )
    xlabels = axes.set_xticklabels( labels, rotation='270' )

    #axes.set_title( 'Total cell population' )
    axes.bar( x, cluster_population, **mpl_kwargs )

    axes.set_xlim( -1, len( labels ) )
    axes.set_xlabel( 'cluster', rotation='0' )
    axes.set_ylabel( '# of cells', rotation='270' )
    axes.grid( True )


    pp.savefig( fig )

    pp.close()

    plt.switch_backend( be )


def print_cell_populations_and_penetrance(pipeline, outputFilename, mpl_kwargs={}):

    be = plt.get_backend()
    plt.switch_backend( 'PDF' )

    pp = PdfPages( outputFilename )

    if not mpl_kwargs.has_key( 'facecolor' ): mpl_kwargs[ 'facecolor' ] = 'blue'
    if not mpl_kwargs.has_key( 'alpha' ): mpl_kwargs[ 'alpha' ] = 0.75
    if not mpl_kwargs.has_key( 'align' ): mpl_kwargs[ 'align' ] = 'center'

    NUM_OF_ROWS = 3
    NUM_OF_COLS = 1
    LEFT_MARGIN = 0.1
    RIGHT_MARGIN = 0.1
    HORIZONTAL_SPACING = 0.1
    BOTTOM_MARGIN = 0.15
    TOP_MARGIN = 0.1
    VERTICAL_SPACING = 0.15
    WIDTH = 1.0 - LEFT_MARGIN - RIGHT_MARGIN
    HEIGHT = 1.0 - TOP_MARGIN - BOTTOM_MARGIN
    SUBPLOT_WIDTH = ( WIDTH - ( NUM_OF_COLS - 1 ) * HORIZONTAL_SPACING ) / float( NUM_OF_COLS )
    SUBPLOT_HEIGHT = ( HEIGHT - ( NUM_OF_ROWS - 1 ) * VERTICAL_SPACING ) / float( NUM_OF_ROWS )


    total_population = numpy.empty( ( len( pipeline.validTreatmentIds ) / 2, ) )
    nonControl_population = numpy.empty( ( len( pipeline.validTreatmentIds ) / 2, ) )
    penetrance = numpy.empty( ( len( pipeline.validTreatmentIds ) / 2, ) )
    tr_labels = []

    for i in xrange( total_population.shape[0] ):
        tr1 = pipeline.validTreatments[ 2 * i ]
        tr2 = pipeline.validTreatments[ 2 * i + 1 ]
        total_tr_mask1 = pipeline.pdc.objFeatures[ pipeline.validCellMask ][ :, pipeline.pdc.objTreatmentFeatureId ] == tr1.rowId
        total_tr_mask2 = pipeline.pdc.objFeatures[ pipeline.validCellMask ][ :, pipeline.pdc.objTreatmentFeatureId ] == tr2.rowId
        nonControl_tr_mask1 = pipeline.pdc.objFeatures[ pipeline.nonControlCellMask][ :, pipeline.pdc.objTreatmentFeatureId ] == tr1.rowId
        nonControl_tr_mask2 = pipeline.pdc.objFeatures[ pipeline.nonControlCellMask][ :, pipeline.pdc.objTreatmentFeatureId ] == tr2.rowId
        total_population[ i ] = numpy.sum( total_tr_mask1 ) + numpy.sum( total_tr_mask2 )
        nonControl_population[ i ] = numpy.sum( nonControl_tr_mask1 ) + numpy.sum( nonControl_tr_mask2 )
        tr_labels.append( pipeline.pdc.treatments[ tr1.rowId ].name )
    penetrance = nonControl_population / total_population

    n = 2
    fig = plt.figure()
    header_text = fig.suptitle( 'Cell populations' )
    #footer_text = fig.text( 0.5, 0.02, ' Page %d' % m, horizontalalignment='center', verticalalignment='bottom' )
    #if bottom_shift != None:
    #    fig.subplots_adjust( bottom=bottom_shift )

    #axes = plt.subplot( num_of_rows, num_of_cols, n )


    row = int( n / NUM_OF_COLS )
    col = int( n % NUM_OF_COLS )
    left = LEFT_MARGIN + col * ( SUBPLOT_WIDTH + HORIZONTAL_SPACING )
    bottom = BOTTOM_MARGIN + row * ( SUBPLOT_HEIGHT + VERTICAL_SPACING )
    axes = fig.add_axes( [ left, bottom, SUBPLOT_WIDTH, SUBPLOT_HEIGHT ], frameon=True )
    n -= 1

    x = numpy.arange( 0, len( tr_labels ) )
    axes.set_xticks( x )
    xlabels = axes.set_xticklabels( tr_labels, rotation='270' )

    axes.set_title( 'Total cell population' )
    axes.bar( x, total_population, **mpl_kwargs )

    axes.set_xlim( -1, len( tr_labels ) )
    axes.set_ylabel( '# of cells', rotation='270' )
    axes.grid( True )


    row = int( n / NUM_OF_COLS )
    col = int( n % NUM_OF_COLS )
    left = LEFT_MARGIN + col * ( SUBPLOT_WIDTH + HORIZONTAL_SPACING )
    bottom = BOTTOM_MARGIN + row * ( SUBPLOT_HEIGHT + VERTICAL_SPACING )
    axes = fig.add_axes( [ left, bottom, SUBPLOT_WIDTH, SUBPLOT_HEIGHT ], frameon=True )
    n -= 1

    x = numpy.arange( 0, len( tr_labels ) )
    axes.set_xticks( x )
    xlabels = axes.set_xticklabels( tr_labels, rotation='270' )

    axes.set_title( 'Non-Control cell population' )
    axes.bar( x, nonControl_population, **mpl_kwargs )

    axes.set_xlim( -1, len( tr_labels ) )
    axes.set_ylabel( '# of cells', rotation='270' )
    axes.grid( True )


    row = int( n / NUM_OF_COLS )
    col = int( n % NUM_OF_COLS )
    left = LEFT_MARGIN + col * ( SUBPLOT_WIDTH + HORIZONTAL_SPACING )
    bottom = BOTTOM_MARGIN + row * ( SUBPLOT_HEIGHT + VERTICAL_SPACING )
    axes = fig.add_axes( [ left, bottom, SUBPLOT_WIDTH, SUBPLOT_HEIGHT ], frameon=True )
    n -= 1

    x = numpy.arange( 0, len( tr_labels ) )
    axes.set_xticks( x )
    xlabels = axes.set_xticklabels( tr_labels, rotation='270' )

    axes.set_title( 'Penetrance' )
    axes.bar( x, penetrance, **mpl_kwargs )

    axes.set_xlim( -1, len( tr_labels ) )
    axes.set_ylabel( '# of cells', rotation='270' )
    axes.grid( True )


    pp.savefig( fig )

    pp.close()

    plt.switch_backend( be )


def __print_cluster_profiles_and_heatmap(bottom_shift, pdc, clusterProfiles, outputFilename, normalize=False, profile_threshold=0.0, barplot_kwargs={}, heatmap_kwargs={}):

    be = plt.get_backend()
    plt.switch_backend( 'PDF' )

    pp = PdfPages( outputFilename )

    mpl_kwargs = barplot_kwargs
    if not mpl_kwargs.has_key( 'facecolor' ): mpl_kwargs[ 'facecolor' ] = 'yellow'
    if not mpl_kwargs.has_key( 'alpha' ): mpl_kwargs[ 'alpha' ] = 0.75
    if not mpl_kwargs.has_key( 'align' ): mpl_kwargs[ 'align' ] = 'center'

    NUM_OF_ROWS = 2
    NUM_OF_COLS = 1
    LEFT_MARGIN = 0.1
    RIGHT_MARGIN = 0.05
    HORIZONTAL_SPACING = 0.1
    BOTTOM_MARGIN = 0.1
    TOP_MARGIN = 0.1
    VERTICAL_SPACING = 0.1
    WIDTH = 1.0 - LEFT_MARGIN - RIGHT_MARGIN
    HEIGHT = 1.0 - TOP_MARGIN - BOTTOM_MARGIN
    SUBPLOT_WIDTH = ( WIDTH - ( NUM_OF_COLS - 1 ) * HORIZONTAL_SPACING ) / float( NUM_OF_COLS )
    SUBPLOT_HEIGHT = ( HEIGHT - ( NUM_OF_ROWS - 1 ) * VERTICAL_SPACING ) / float( NUM_OF_ROWS )

    fig = None
    n = NUM_OF_ROWS * NUM_OF_COLS
    m = 1

    nonEmptyProfileIndices = range( clusterProfiles.shape[0] )

    for i in xrange( clusterProfiles.shape[0] ):

        if numpy.all( clusterProfiles[ i ] == 0 ):
            nonEmptyProfileIndices.remove( i )
            continue

        sys.stdout.write( '\r  treatment %s ...' % pdc.treatments[ i ].name )
        sys.stdout.flush()

        if n >= NUM_OF_ROWS * NUM_OF_COLS:
            if fig != None:
                pp.savefig( fig )
            fig = plt.figure()
            header_text = fig.suptitle( 'Cluster profiles' )
            footer_text = fig.text( 0.5, 0.02, ' Page %d' % m, horizontalalignment='center', verticalalignment='bottom' )
            #if bottom_shift != None:
            #    fig.subplots_adjust( bottom=bottom_shift )
            n = 0
            m += 1

        #axes = plt.subplot( num_of_rows, num_of_cols, n )
        row = int( n / NUM_OF_COLS )
        col = int( n % NUM_OF_COLS )
        left = LEFT_MARGIN + col * ( SUBPLOT_WIDTH + HORIZONTAL_SPACING )
        bottom = BOTTOM_MARGIN + row * ( SUBPLOT_HEIGHT + VERTICAL_SPACING )
        axes = fig.add_axes( [ left, bottom, SUBPLOT_WIDTH, SUBPLOT_HEIGHT ], frameon=True )
        n += 1

        profile = clusterProfiles[ i ]

        if normalize:
            #abs_value = numpy.sqrt( numpy.sum( profile ** 2 ) )
            abs_value = float( numpy.sum( profile ) )
            profile = profile / abs_value

        if profile_threshold > 0.0:
            max = numpy.max( profile )
            threshold_mask = profile < max * profile_threshold
            profile[ threshold_mask ] = 0.0
            profile = profile / float( numpy.sum( profile ) )

        x = numpy.arange( 0, profile.shape[0] )
        xlabels = []
        step = int( profile.shape[0] / 40  ) + 1
        for j in xrange( x.shape[0] ):
            if j % step == 0:
                xlabels.append( str( x[ j ] ) )
            else:
                xlabels.append( '' )

        axes.set_title( 'Treatment %s' % pdc.treatments[ i ].name )

        axes.bar( x, profile, **mpl_kwargs )
        axes.set_xticks( x )
        xlabels = axes.set_xticklabels( xlabels, rotation='270' )

        xticks = axes.xaxis.get_major_ticks() + axes.xaxis.get_minor_ticks()
        for tick in xticks:
            tick.tick1On = False
            tick.tick2On = False
            tick.label1On = True
            tick.label2On = False

        axes.set_xlim( -1, profile.shape[0] )

        #axes.set_xlabel( 'cluster ID' )
        axes.set_ylabel( 'population size', rotation='270' )

        #axes.grid( True )

    pp.savefig( fig )

    treatments = []
    for i in xrange( len( pdc.treatments ) ):
        if i in nonEmptyProfileIndices:
            treatments.append( pdc.treatments[ i ] )

    sys.stdout.write( '\n' )
    sys.stdout.flush()

    if bottom_shift == None:

        bboxes = []
        for xlabel in xlabels:
            bbox = xlabel.get_window_extent()
            bboxi = bbox.inverse_transformed( fig.transFigure )
            bboxes.append( bboxi )
        bbox = mtransforms.Bbox.union( bboxes )
        if fig.subplotpars.bottom < bbox.height:
            bottom_shift = 1.1 * bbox.height
        else:
            bottom_shift = 0.0


    LEFT_MARGIN = 0.1
    RIGHT_MARGIN = 0.1
    HORIZONTAL_SPACING = 0.1
    BOTTOM_MARGIN = 0.2
    TOP_MARGIN = 0.1
    WIDTH = 1.0 - LEFT_MARGIN - RIGHT_MARGIN
    HEIGHT = 1.0 - TOP_MARGIN - BOTTOM_MARGIN

    profileHeatmap = numpy.zeros( ( clusterProfiles.shape[0], clusterProfiles.shape[0] ) )
    profileHeatmap[ numpy.identity( profileHeatmap.shape[0], dtype=bool ) ] = numpy.nan

    """cmap = matplotlib.colors.LinearSegmentedColormap(
        'gray_values',
        { 'red' :   [ ( 0.0, 1.0, 1.0 ),
                      ( 1.0, 0.0, 0.0 ) ],
          'green' : [ ( 0.0, 1.0, 1.0 ),
                      ( 1.0, 0.0, 0.0 ) ],
          'blue' :  [ ( 0.0, 1.0, 1.0 ),
                      ( 1.0, 0.0, 0.0 ) ],
        }
    )"""

    fig = plt.figure()
    header_text = fig.suptitle( 'Heatmap of cluster profile distances (Summed min-max)' )
    footer_text = fig.text( 0.5, 0.02, ' Page %d' % m, horizontalalignment='center', verticalalignment='bottom' )

    axes = fig.add_axes( [ LEFT_MARGIN, BOTTOM_MARGIN, WIDTH, HEIGHT ], frameon=True )

    """for i in xrange( clusterProfiles.shape[0] ):

        profile1 = clusterProfiles[ i ]
        norm_profile1 = profile1 / float( numpy.sum( profile1 ) )

        if profile_threshold > 0.0:
            max = numpy.max( profile1 )
            threshold_mask = profile1 < max * profile_threshold
            norm_profile1[ threshold_mask ] = 0.0
            norm_profile1 = norm_profile1 / float( numpy.sum( norm_profile1 ) )

        for j in xrange( 0, i ):

            profile2 = clusterProfiles[ j ]
            norm_profile2 = profile2 / float( numpy.sum( profile2 ) )

            if profile_threshold > 0.0:
                max = numpy.max( profile2 )
                threshold_mask = profile2 < max * profile_threshold
                norm_profile2[ threshold_mask ] = 0.0
                norm_profile2 = norm_profile2 / float( numpy.sum( norm_profile2 ) )

            # L2-norm
            dist = numpy.sqrt( numpy.sum( ( norm_profile1 - norm_profile2 ) ** 2 ) )
            # chi-square
            #dist =  ( norm_profile1 - norm_profile2 ) ** 2 / ( norm_profile1 + norm_profile2 )
            #dist[ numpy.logical_and( norm_profile1 == 0, norm_profile2 == 0 ) ] = 0.0
            #dist = numpy.sum( dist )
            #print '%s <-> %s: %f' % ( self.pipeline.pdc.treatments[ i ].name, self.pipeline.pdc.treatments[ j ].name, dist )

            profileHeatmap[ i, j ] = dist
            profileHeatmap[ j, i ] = dist"""

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

            #match = numpy.sum( norm_profile1 * norm_profile2 )

            #self_match1 = numpy.sum( norm_profile1 ** 2 )
            #self_match2 = numpy.sum( norm_profile1 ** 2 )

            #match = match / ( 0.5 * ( self_match1 + self_match2 ) )

            #match = numpy.sum( numpy.min( [ norm_profile1, norm_profile2 ], axis=0 ) )

            # L2-norm
            #dist = numpy.sqrt( numpy.sum( ( norm_profile1 - norm_profile2 ) ** 2 ) )
            # chi-square
            #dist =  ( norm_profile1 - norm_profile2 ) ** 2 / ( norm_profile1 + norm_profile2 )
            #dist[ numpy.logical_and( norm_profile1 == 0, norm_profile2 == 0 ) ] = 1.0
            #dist = 0.5 * numpy.sum( dist )
            #print '%s <-> %s: %f' % ( self.pipeline.pdc.treatments[ i ].name, self.pipeline.pdc.treatments[ j ].name, dist )

            #min_match = numpy.sum( numpy.min( [ norm_profile1, norm_profile2 ], axis=0 ) )
            #max_match = numpy.sum( numpy.max( [ norm_profile1, norm_profile2 ], axis=0 ) )

            # summed minmax-metric
            min_match = numpy.sum( numpy.min( [ norm_profile1, norm_profile2 ], axis=0 )**2 )
            max_match = numpy.sum( numpy.max( [ norm_profile1, norm_profile2 ], axis=0 )**2 )

            match = min_match / max_match

            # L1-norm
            #dist2 = numpy.sum( numpy.abs( norm_profile1 - norm_profile2 ) )

            profileHeatmap[ i, j ] = match # Summed min-max metric, lower triangular matrix
            profileHeatmap[ j, i ] = match # Summed min-max metric, upper triangular matrix
            #profileHeatmap[ j, i ] = dist2 # L^1-norm, upper triangular matrix

    profileHeatmap = profileHeatmap[ nonEmptyProfileIndices ][ :, nonEmptyProfileIndices ]

    #aximg = axes.imshow( profileHeatmap, cmap=cmap, interpolation='nearest' )

    #x = numpy.arange( 0.5, profileHeatmap.shape[1] - 1.5 )
    #axes.set_xticks( x, minor=True )
    x = numpy.arange( profileHeatmap.shape[1] )
    axes.set_xticks( x, minor=False )
    axes.set_xlim( -0.5, profileHeatmap.shape[1] - 0.5)

    #y = numpy.arange( 0.5, profileHeatmap.shape[0] - 1.5 )
    #axes.set_yticks( y, minor=True )
    y = numpy.arange( profileHeatmap.shape[0] )
    axes.set_yticks( y, minor=False )
    axes.set_ylim( -0.5, profileHeatmap.shape[0] - 0.5)

    labels = []
    for tr in treatments:
        labels.append( tr.name )
    #for i in xrange( clusterProfiles.shape[0] ):
    #    if i in nonEmptyProfileIndices:
    #        labels.append( pdc.treatments[ i ].name )

    axes.set_xticklabels( labels, rotation='270' )
    axes.set_yticklabels( labels[ ::-1 ], rotation='0' )

    #heatmap = profileHeatmap.copy()
    #for i in xrange( 0, heatmap.shape[0], 2 ):
    #    heatmap[ i, i ] = heatmap[ i+1, i+1 ] = heatmap[ i, i+1 ]

    xticks = axes.xaxis.get_major_ticks() + axes.xaxis.get_minor_ticks()
    for tick in xticks:
        tick.tick1On = False
        tick.tick2On = False
        tick.label1On = True
        tick.label2On = False

    yticks = axes.yaxis.get_major_ticks() + axes.yaxis.get_minor_ticks()
    for tick in yticks:
        tick.tick1On = False
        tick.tick2On = False
        tick.label1On = True
        tick.label2On = False

    #color_factory = lambda v: ( 1-v, 1-v, 1-v )
    #draw_sum_minmax_heatmap( axes, profileHeatmap, ( 0, 0 ), 6, color_factory )
    draw_sum_minmax_heatmap( axes, 1.0 - profileHeatmap, ( 0, 0 ), 6 )

    """max_value = numpy.max( profileHeatmap[ numpy.invert( numpy.isnan( profileHeatmap ) ) ] )
    for i in xrange( profileHeatmap.shape[0] ):
        for j in xrange( i, profileHeatmap.shape[0] ):
            if not numpy.isnan( profileHeatmap[ i, j ] ):
                x = ( j + 0.5 ) / float( profileHeatmap.shape[0] )
                y = ( i + 0.5 ) / float( profileHeatmap.shape[1] )
                string = '%.2f' % ( profileHeatmap[ i, j ] )
                if ( profileHeatmap[ i, j ] / max_value ) > 0.5:
                    textColor = 'white'
                else:
                    textColor = 'black'
                axes.text( x, y, string, fontsize=8,
                           horizontalalignment = 'center',
                           verticalalignment = 'center',
                           transform = axes.transAxes,
                           color = textColor
                )"""

    mpl_kwargs = heatmap_kwargs
    if not mpl_kwargs.has_key( 'color' ): mpl_kwargs[ 'color' ] = 'white'
    if not mpl_kwargs.has_key( 'linestyle' ): mpl_kwargs[ 'linestyle' ] = '-'
    if not mpl_kwargs.has_key( 'linewidth' ): mpl_kwargs[ 'linewidth' ] = 2
    if not mpl_kwargs.has_key( 'which' ): mpl_kwargs[ 'which' ] = 'minor'
    #axes.grid( True )

    #fig.colorbar( aximg )

    pp.savefig( fig )

    sum_of_min_profileHeatmap = profileHeatmap


    profileHeatmap = numpy.zeros( ( clusterProfiles.shape[0], clusterProfiles.shape[0] ) )
    profileHeatmap[ numpy.identity( profileHeatmap.shape[0], dtype=bool ) ] = numpy.nan

    """cmap = matplotlib.colors.LinearSegmentedColormap(
        'gray_values',
        { 'red' :   [ ( 0.0, 0.0, 0.0 ),
                      ( 1.0, 1.0, 1.0 ) ],
          'green' : [ ( 0.0, 0.0, 0.0 ),
                      ( 1.0, 1.0, 1.0 ) ],
          'blue' :  [ ( 0.0, 0.0, 0.0 ),
                      ( 1.0, 1.0, 1.0 ) ],
        }
    )"""

    fig = plt.figure()
    header_text = fig.suptitle( 'Heatmap of cluster profile distances (L^2 [lower] and Chi^2 [upper])' )
    footer_text = fig.text( 0.5, 0.02, ' Page %d' % m, horizontalalignment='center', verticalalignment='bottom' )

    axes = fig.add_axes( [ LEFT_MARGIN, BOTTOM_MARGIN, WIDTH, HEIGHT ], frameon=True )

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

            # L2-norm
            dist1 = numpy.sqrt( numpy.sum( ( norm_profile1 - norm_profile2 ) ** 2 ) )
            # chi-square
            dist2 =  ( norm_profile1 - norm_profile2 ) ** 2 / ( norm_profile1 + norm_profile2 )
            dist2[ numpy.logical_and( norm_profile1 == 0, norm_profile2 == 0 ) ] = 0.0
            dist2 = numpy.sum( dist2 )
            #print '%s <-> %s: %f' % ( self.pipeline.pdc.treatments[ i ].name, self.pipeline.pdc.treatments[ j ].name, dist )

            profileHeatmap[ i, j ] = dist1 # L^2-norm, lower triangular matrix
            profileHeatmap[ j, i ] = dist2 # Chi^2-norm, upper triangular matrix


    profileHeatmap = profileHeatmap[ nonEmptyProfileIndices ][ :, nonEmptyProfileIndices ]

    #aximg = axes.imshow( profileHeatmap, cmap=cmap, interpolation='nearest' )

    #x = numpy.arange( 0.5, profileHeatmap.shape[1] - 1.5 )
    #axes.set_xticks( x, minor=True )
    x = numpy.arange( profileHeatmap.shape[1] )
    axes.set_xticks( x, minor=False )
    axes.set_xlim( -0.5, profileHeatmap.shape[1] - 0.5)

    #y = numpy.arange( 0.5, profileHeatmap.shape[0] - 1.5 )
    #axes.set_yticks( y, minor=True )
    y = numpy.arange( profileHeatmap.shape[0] )
    axes.set_yticks( y, minor=False )
    axes.set_ylim( -0.5, profileHeatmap.shape[0] - 0.5)

    labels = []
    for tr in treatments:
        labels.append( tr.name )
    #for i in xrange( clusterProfiles.shape[0] ):
    #    if i in nonEmptyProfileIndices:
    #        labels.append( pdc.treatments[ i ].name )

    axes.set_xticklabels( labels, rotation='270' )
    axes.set_yticklabels( labels[ ::-1 ], rotation='0' )

    #heatmap = profileHeatmap.copy()
    #for i in xrange( 0, heatmap.shape[0], 2 ):
    #    heatmap[ i, i ] = heatmap[ i+1, i+1 ] = heatmap[ i, i+1 ]

    xticks = axes.xaxis.get_major_ticks() + axes.xaxis.get_minor_ticks()
    for tick in xticks:
        tick.tick1On = False
        tick.tick2On = False
        tick.label1On = True
        tick.label2On = False

    yticks = axes.yaxis.get_major_ticks() + axes.yaxis.get_minor_ticks()
    for tick in yticks:
        tick.tick1On = False
        tick.tick2On = False
        tick.label1On = True
        tick.label2On = False

    draw_diagonal_heatmap( axes, profileHeatmap, ( 0, 0 ), 6 )
    draw_upper_heatmap( axes, profileHeatmap, ( 0, 0 ), 6 )
    draw_lower_heatmap( axes, profileHeatmap, ( 0, 0 ), 6 )

    """max_value = numpy.max( profileHeatmap[ numpy.invert( numpy.isnan( profileHeatmap ) ) ] )
    for i in xrange( profileHeatmap.shape[0] ):
        for j in xrange( i, profileHeatmap.shape[0] ):
            if not numpy.isnan( profileHeatmap[ i, j ] ):
                x = ( j + 0.5 ) / float( profileHeatmap.shape[0] )
                y = ( i + 0.5 ) / float( profileHeatmap.shape[1] )
                string = '%.2f' % ( profileHeatmap[ i, j ] )
                if ( profileHeatmap[ i, j ] / max_value ) < 0.5:
                    textColor = 'white'
                else:
                    textColor = 'black'
                axes.text( x, y, string, fontsize=8,
                           horizontalalignment = 'center',
                           verticalalignment = 'center',
                           transform = axes.transAxes,
                           color = textColor
                )"""

    mpl_kwargs = heatmap_kwargs
    if not mpl_kwargs.has_key( 'color' ): mpl_kwargs[ 'color' ] = 'white'
    if not mpl_kwargs.has_key( 'linestyle' ): mpl_kwargs[ 'linestyle' ] = '-'
    if not mpl_kwargs.has_key( 'linewidth' ): mpl_kwargs[ 'linewidth' ] = 2
    if not mpl_kwargs.has_key( 'which' ): mpl_kwargs[ 'which' ] = 'minor'
    #axes.grid( True )

    #fig.colorbar( aximg )

    pp.savefig( fig )

    l2_norm_profileHeatmap = profileHeatmap


    pp.close()

    plt.switch_backend( be )

    return bottom_shift, sum_of_min_profileHeatmap, l2_norm_profileHeatmap, treatments

def print_cluster_profiles_and_heatmap(pdc, clusterProfiles, outputFilename, normalize=False, profile_threshold=0.0, barplot_kwargs={}, heatmap_kwargs={}):

    bottom_shift = None
    bottom_shift, sum_of_min_profileHeatmap, l2_norm_profileHeatmap, treatments = \
        __print_cluster_profiles_and_heatmap( bottom_shift, pdc, clusterProfiles, outputFilename, normalize, profile_threshold, barplot_kwargs, heatmap_kwargs )
    __print_cluster_profiles_and_heatmap( bottom_shift, pdc, clusterProfiles, outputFilename, normalize, profile_threshold, barplot_kwargs, heatmap_kwargs )

    return sum_of_min_profileHeatmap, l2_norm_profileHeatmap, treatments


def draw_diagonal_heatmap(axes, map, xy, fontsize=12, color_factory=None):

    trans1 = mtransforms.Affine2D.identity()
    trans1 = trans1.scale( 1. / ( map.shape[0] ), 1. / ( map.shape[1] ) ).scale( 1.0, -1.0 ).translate( 0.0, 1.0 )
    trans = mtransforms.composite_transform_factory( trans1, axes.transAxes )

    if color_factory == None:
        color_factory = lambda v: ( v, v, v )
    max_value = numpy.max( map[ numpy.isfinite( map ) ] )
    if max_value == 0.0 or not numpy.isfinite( max_value ):
        max_value = 1.0
    print 'max_value:', max_value
    for i in xrange( map.shape[0] ):
        if numpy.isfinite( map[ i, i ] ):
            string = '%.2f' % ( map[ i, i ] )
            v = map[ i, i ] / max_value
            c = color_factory( v )
            c_mean = numpy.mean( c )
            fillColor = c
            frameColor = 'black'
            if c_mean > 0.5:
                textColor = 'black'
            else:
                textColor = 'white'
            fillZorder = 0
            frameZorder = 1
            textZorder = 3
            nx = i + xy[0]
            ny = i + xy[1]
            rect_xy = ( nx, ny )
            tx = nx + 0.5
            ty = ny + 0.5
            rect_d = 1.0
            fs = fontsize

            axes.text( tx, ty, string, fontsize=fs,
                       horizontalalignment = 'center',
                       verticalalignment = 'center',
                       transform = trans,
                       color = textColor, zorder=textZorder
            )

            rec = matplotlib.patches.Rectangle( rect_xy, rect_d, rect_d, color=fillColor, zorder=fillZorder, transform = trans )
            axes.add_artist( rec )
            rec = matplotlib.patches.Rectangle( rect_xy, rect_d, rect_d, color=frameColor, fill=False, zorder=frameZorder, transform = trans )
            axes.add_artist( rec )

def draw_upper_heatmap(axes, map, xy, fontsize=12, color_factory=None):

    trans1 = mtransforms.Affine2D.identity()
    trans1 = trans1.scale( 1. / ( map.shape[0] ), 1. / ( map.shape[1] ) ).scale( 1.0, -1.0 ).translate( 0.0, 1.0 )
    trans = mtransforms.composite_transform_factory( trans1, axes.transAxes )

    if color_factory == None:
        color_factory = lambda v: ( v, v, v )
    max_value = numpy.max( map[ numpy.isfinite( map ) ] )
    if max_value == 0.0 or not numpy.isfinite( max_value ):
        max_value = 1.0
    print 'max_value:', max_value
    for i in xrange( map.shape[0] ):
        for j in xrange( i, map.shape[0] ):
            if numpy.isfinite( map[ i, j ] ):
                string = '%.2f' % ( map[ i, j ] )
                v = map[ i, j ] / max_value
                c = color_factory( v )
                c_mean = numpy.mean( c )
                fillColor = c
                frameColor = 'black'
                if c_mean > 0.5:
                    textColor = 'black'
                else:
                    textColor = 'white'
                fillZorder = 0
                frameZorder = 1
                textZorder = 3
                nx = j + xy[0]
                ny = i + xy[1]
                rect_xy = ( nx, ny )
                tx = nx + 0.5
                ty = ny + 0.5
                rect_d = 1.0
                fs = fontsize

                axes.text( tx, ty, string, fontsize=fs,
                           horizontalalignment = 'center',
                           verticalalignment = 'center',
                           transform = trans,
                           color = textColor, zorder=textZorder
                )

                rec = matplotlib.patches.Rectangle( rect_xy, rect_d, rect_d, color=fillColor, zorder=fillZorder, transform = trans )
                axes.add_artist( rec )
                rec = matplotlib.patches.Rectangle( rect_xy, rect_d, rect_d, color=frameColor, fill=False, zorder=frameZorder, transform = trans )
                axes.add_artist( rec )

def draw_lower_heatmap(axes, map, xy, fontsize=12, color_factory=None):
    trans1 = mtransforms.Affine2D.identity()
    trans1 = trans1.scale( 1. / ( map.shape[0] ), 1. / ( map.shape[1] ) ).scale( 1.0, -1.0 ).translate( 0.0, 1.0 )
    trans = mtransforms.composite_transform_factory( trans1, axes.transAxes )

    if color_factory == None:
        color_factory = lambda v: ( v, v, v )
    max_value = numpy.max( map[ numpy.isfinite( map ) ] )
    if max_value == 0.0 or not numpy.isfinite( max_value ):
        max_value = 1.0
    print 'max_value:', max_value
    for i in xrange( map.shape[0] ):
        for j in xrange( i ):
            if numpy.isfinite( map[ i, j ] ):
                string = '%.2f' % ( map[ i, j ] )
                v = map[ i, j ] / max_value
                c = color_factory( v )
                c_mean = numpy.mean( c )
                fillColor = c
                frameColor = 'black'
                if c_mean > 0.5:
                    textColor = 'black'
                else:
                    textColor = 'white'
                fillZorder = 0
                frameZorder = 1
                textZorder = 3
                nx = j + xy[0]
                ny = i + xy[1]
                rect_xy = ( nx, ny )
                tx = nx + 0.5
                ty = ny + 0.5
                rect_d = 1.0
                fs = fontsize

                axes.text( tx, ty, string, fontsize=fs,
                           horizontalalignment = 'center',
                           verticalalignment = 'center',
                           transform = trans,
                           color = textColor, zorder=textZorder
                )

                rec = matplotlib.patches.Rectangle( rect_xy, rect_d, rect_d, color=fillColor, zorder=fillZorder, transform = trans )
                axes.add_artist( rec )
                rec = matplotlib.patches.Rectangle( rect_xy, rect_d, rect_d, color=frameColor, fill=False, zorder=frameZorder, transform = trans )
                axes.add_artist( rec )

def draw_sum_minmax_heatmap(axes, map, xy, fontsize=12, color_factory=None):
    #axes.set_frame_on( False )

    trans1 = mtransforms.Affine2D.identity()
    trans1 = trans1.scale( 1. / ( map.shape[0] ), 1. / ( map.shape[1] ) ).scale( 1.0, -1.0 ).translate( 0.0, 1.0 )
    trans = mtransforms.composite_transform_factory( trans1, axes.transAxes )

    print 'original transformation:'
    print axes.transAxes.transform( ( 0, 0 ) )
    print axes.transAxes.transform( ( 0.5, 0.5 ) )
    print axes.transAxes.transform( ( 0, 1 ) )
    print axes.transAxes.transform( ( 1, 0 ) )
    print 'transformation:'
    print trans1.transform( ( 0, 0 ) )
    print trans1.transform( ( ( map.shape[0])/2.0, ( map.shape[0])/2.0 ) )
    print trans1.transform( ( 0, ( map.shape[0]) ) )
    print trans1.transform( ( map.shape[0], 0 ) )
    print 'composite transformation:'
    print trans.transform( ( 0, 0 ) )
    print trans.transform( ( ( map.shape[0])/2.0, ( map.shape[0])/2.0 ) )
    print trans.transform( ( 0, ( map.shape[0]) ) )
    print trans.transform( ( map.shape[0], 0 ) )

    print 'draw_sum_minmax_heatmaps'
    if color_factory == None:
        color_factory = lambda v: ( v, v, v )
    max_value = numpy.max( map[ numpy.isfinite( map ) ] )
    if max_value == 0.0 or not numpy.isfinite( max_value ):
        max_value = 1.0
    print 'max_value:', max_value
    for i in xrange( map.shape[0] ):
        #for j in xrange( map.shape[0]-1-i, -1, -1 ):
        for j in xrange( 0, i ):
            if numpy.isfinite( map[ i, j ] ):
                string = '%.2f' % ( map[ i, j ] )
                v = map[ i, j ] / max_value
                c = color_factory( v )
                c_mean = numpy.mean( c )
                fillColor = c
                frameColor = 'black'
                if c_mean > 0.5:
                    textColor = 'black'
                else:
                    textColor = 'white'
                fillZorder = 0
                frameZorder = 1
                textZorder = 3
                if ( j % 2 == 0 ) and ( i == j + 1 ):
                    nx = j + xy[0]
                    ny = i + xy[1] - 1.0
                    rect_xy = ( nx, ny )
                    tx = nx + 1
                    ty = ny + 1
                    rect_d = 2.0
                    fs = 2 * fontsize
                    frameZorder = 2
                elif ( i % 2 == 0 ) and ( j == i + 1 ):
                    continue
                else:
                    nx = j + xy[0]
                    ny = i + xy[1]
                    rect_xy = ( nx, ny )
                    tx = nx + 0.5
                    ty = ny + 0.5
                    rect_d = 1.0
                    fs = fontsize

                axes.text( tx, ty, string, fontsize=fs,
                           horizontalalignment = 'center',
                           verticalalignment = 'center',
                           transform = trans,
                           color = textColor, zorder=textZorder
                )

                rec = matplotlib.patches.Rectangle( rect_xy, rect_d, rect_d, color=fillColor, zorder=fillZorder, transform = trans )
                axes.add_artist( rec )
                rec = matplotlib.patches.Rectangle( rect_xy, rect_d, rect_d, color=frameColor, fill=False, zorder=frameZorder, transform = trans )
                axes.add_artist( rec )

def draw_treatment_similarity_map(axes, map, xy, fontsize=12, color_factory=None):
    #axes.set_frame_on( False )

    trans1 = mtransforms.Affine2D.identity()
    trans1 = trans1.scale( 1. / ( map.shape[0] ), 1. / ( map.shape[1] ) ).scale( 1.0, -1.0 ).translate( 0.0, 1.0 )
    trans = mtransforms.composite_transform_factory( trans1, axes.transAxes )

    if color_factory == None:
        color_factory = lambda v: ( v, v, v )
    max_value = numpy.max( map[ numpy.isfinite( map ) ] )
    if max_value == 0.0 or not numpy.isfinite( max_value ):
        max_value = 1.0
    for i in xrange( map.shape[0] ):
        #for j in xrange( map.shape[0]-1-i, -1, -1 ):
        for j in xrange( 0, i+1 ):
            if numpy.isfinite( map[ i, j ] ):
                string = '%.2f' % ( map[ i, j ] )
                v = map[ i, j ] / max_value
                c = color_factory( v )
                c_mean = numpy.mean( c )
                fillColor = c
                frameColor = 'black'
                if c_mean > 0.5:
                    textColor = 'black'
                else:
                    textColor = 'white'
                fillZorder = 0
                frameZorder = 1
                textZorder = 3
                nx = j + xy[0]
                ny = i + xy[1]
                rect_xy = ( nx, ny )
                tx = nx + 0.5
                ty = ny + 0.5
                rect_d = 1.0
                fs = fontsize

                axes.text( tx, ty, string, fontsize=fs,
                           horizontalalignment = 'center',
                           verticalalignment = 'center',
                           transform = trans,
                           color = textColor, zorder=textZorder
                )

                rec = matplotlib.patches.Rectangle( rect_xy, rect_d, rect_d, color=fillColor, zorder=fillZorder, transform = trans )
                axes.add_artist( rec )
                rec = matplotlib.patches.Rectangle( rect_xy, rect_d, rect_d, color=frameColor, fill=False, zorder=frameZorder, transform = trans )
                axes.add_artist( rec )


def print_treatment_similarity_map(treatmentSimilarityMap, labels, outputFilename, title='Similarity map of treatments', mpl_kwargs={}):

    be = plt.get_backend()
    plt.switch_backend( 'PDF' )

    pp = PdfPages( outputFilename )

    fig = plt.figure()

    LEFT_MARGIN = 0.1
    RIGHT_MARGIN = 0.1
    HORIZONTAL_SPACING = 0.1
    BOTTOM_MARGIN = 0.2
    TOP_MARGIN = 0.1
    WIDTH = 1.0 - LEFT_MARGIN - RIGHT_MARGIN
    HEIGHT = 1.0 - TOP_MARGIN - BOTTOM_MARGIN

    """cmap = matplotlib.colors.LinearSegmentedColormap(
        'gray_values',
        { 'red' :   [ ( 0.0, 1.0, 1.0 ),
                      ( 1.0, 0.0, 0.0 ) ],
          'green' : [ ( 0.0, 1.0, 1.0 ),
                      ( 1.0, 0.0, 0.0 ) ],
          'blue' :  [ ( 0.0, 1.0, 1.0 ),
                      ( 1.0, 0.0, 0.0 ) ],
        }
    )"""

    fig = plt.figure()
    header_text = fig.suptitle( title )
    #footer_text = fig.text( 0.5, 0.02, ' Page %d' % m, horizontalalignment='center', verticalalignment='bottom' )

    axes = fig.add_axes( [ LEFT_MARGIN, BOTTOM_MARGIN, WIDTH, HEIGHT ], frameon=True )

    #max_value = numpy.max( treatmentSimilarityMap )
    #treatmentSimilarityMap = treatmentSimilarityMap / max_value

    #aximg = axes.imshow( treatmentSimilarityMap, vmin=0.0, vmax=max_value, cmap=cmap, interpolation='nearest' )

    #x = numpy.arange( 0.5, profileHeatmap.shape[1] - 1.5 )
    #axes.set_xticks( x, minor=True )
    x = numpy.arange( treatmentSimilarityMap.shape[1] )
    axes.set_xticks( x, minor=False )
    axes.set_xlim( -0.5, treatmentSimilarityMap.shape[1] - 0.5)

    #y = numpy.arange( 0.5, profileHeatmap.shape[0] - 1.5 )
    #axes.set_yticks( y, minor=True )
    y = numpy.arange( treatmentSimilarityMap.shape[0] )
    axes.set_yticks( y, minor=False )
    axes.set_ylim( -0.5, treatmentSimilarityMap.shape[0] - 0.5)

    """labels = []
    for i in xrange( clusterProfiles.shape[0] ):
        if i in nonEmptyProfileIndices:
            labels.append( pdc.treatments[ i ].name )"""

    axes.set_xticklabels( labels, rotation='270' )
    axes.set_yticklabels( labels[ ::-1 ], rotation='0' )

    xticks = axes.xaxis.get_major_ticks() + axes.xaxis.get_minor_ticks()
    for tick in xticks:
        tick.tick1On = False
        tick.tick2On = False
        tick.label1On = True
        tick.label2On = False

    yticks = axes.yaxis.get_major_ticks() + axes.yaxis.get_minor_ticks()
    for tick in yticks:
        tick.tick1On = False
        tick.tick2On = False
        tick.label1On = True
        tick.label2On = False

    #print treatmentSimilarityMap
    #draw_lower_heatmap( axes, treatmentSimilarityMap, ( -0.5, -0.5 ), 12 )
    #color_factory = lambda v: ( 1-v, 1-v, 1-v )
    #draw_treatment_similarity_map( axes, treatmentSimilarityMap, ( 0, 0 ), 12, color_factory )
    draw_treatment_similarity_map( axes, treatmentSimilarityMap, ( 0, 0 ), 12 )

    """for i in xrange( treatmentSimilarityMap.shape[0] ):
        for j in xrange( i, treatmentSimilarityMap.shape[0] ):
            x = ( j + 0.5 ) / float( treatmentSimilarityMap.shape[0] )
            y = ( i + 0.5 ) / float( treatmentSimilarityMap.shape[1] )
            string = '%.2f' % ( treatmentSimilarityMap[ i, j ] )
            if ( treatmentSimilarityMap[ i, j ] / max_value ) > 0.5:
                textColor = 'white'
            else:
                textColor = 'black'
            axes.text( x, y, string,
                       horizontalalignment = 'center',
                       verticalalignment = 'center',
                       transform = axes.transAxes,
                       color = textColor
            )
            v = treatmentSimilarityMap[ i, j ] / max_value
            #if i != j:
            #    x = i
            #    y = j
            #    r = 0.95 * 0.5 * v
            #    ell = matplotlib.patches.Circle( ( x, y ), radius=r, color=( 0.7,0.7,0.7 ) )
            #    axes.add_artist( ell )
            x = j - 0.5
            y = i - 0.5
            d = 1.0
            c = ( 1 - v, 1 - v, 1 - v )
            rec = matplotlib.patches.Rectangle( ( x, y ), d, d, color=c )
            axes.add_artist( rec )
            c = ( 0,0,0 )
            rec = matplotlib.patches.Rectangle( ( x, y ), d, d, color=c, fill=False )
            axes.add_artist( rec )"""

    if not mpl_kwargs.has_key( 'color' ): mpl_kwargs[ 'color' ] = 'white'
    if not mpl_kwargs.has_key( 'linestyle' ): mpl_kwargs[ 'linestyle' ] = '-'
    if not mpl_kwargs.has_key( 'linewidth' ): mpl_kwargs[ 'linewidth' ] = 2
    if not mpl_kwargs.has_key( 'which' ): mpl_kwargs[ 'which' ] = 'minor'
    #axes.grid( True )

    #fig.colorbar( aximg )

    pp.savefig( fig )

    pp.close()

    plt.switch_backend( be )

def print_analyse_plot(random_mean, random_std, replicate_mean, replicate_std, labels, outputFilename, title='Analysation plot', xlabel=None, ylabel=None, mpl_kwargs={}):


    be = plt.get_backend()
    plt.switch_backend( 'PDF' )

    pp = PdfPages( outputFilename )

    if mpl_kwargs.has_key( 'facecolor' ): del mpl_kwargs[ 'facecolor' ]
    if not mpl_kwargs.has_key( 'alpha' ): mpl_kwargs[ 'alpha' ] = 0.75
    if not mpl_kwargs.has_key( 'align' ): mpl_kwargs[ 'align' ] = 'center'

    NUM_OF_ROWS = 1
    NUM_OF_COLS = 1
    LEFT_MARGIN = 0.1
    RIGHT_MARGIN = 0.1
    HORIZONTAL_SPACING = 0.1
    BOTTOM_MARGIN = 0.1
    TOP_MARGIN = 0.1
    VERTICAL_SPACING = 0.1
    WIDTH = 1.0 - LEFT_MARGIN - RIGHT_MARGIN
    HEIGHT = 1.0 - TOP_MARGIN - BOTTOM_MARGIN
    SUBPLOT_WIDTH = ( WIDTH - ( NUM_OF_COLS - 1 ) * HORIZONTAL_SPACING ) / float( NUM_OF_COLS )
    SUBPLOT_HEIGHT = ( HEIGHT - ( NUM_OF_ROWS - 1 ) * VERTICAL_SPACING ) / float( NUM_OF_ROWS )

    n = 0

    fig = plt.figure()
    header_text = fig.suptitle( title )
    #footer_text = fig.text( 0.5, 0.02, ' Page %d' % m, horizontalalignment='center', verticalalignment='bottom' )
    #if bottom_shift != None:
    #    fig.subplots_adjust( bottom=bottom_shift )

    #axes = plt.subplot( num_of_rows, num_of_cols, n )


    row = int( n / NUM_OF_COLS )
    col = int( n % NUM_OF_COLS )
    left = LEFT_MARGIN + col * ( SUBPLOT_WIDTH + HORIZONTAL_SPACING )
    bottom = BOTTOM_MARGIN + row * ( SUBPLOT_HEIGHT + VERTICAL_SPACING )
    axes = fig.add_axes( [ left, bottom, SUBPLOT_WIDTH, SUBPLOT_HEIGHT ], frameon=True )
    n += 1

    x = numpy.arange( 0, 2 * len( labels ), 2 )
    #axes.set_xticks( x )
    #axes.set_xticklabels( xlabels, rotation='270', visible=False )

    axes.set_xticks( x )
    axes.set_xticklabels( labels, rotation='270' )

    xticks = axes.xaxis.get_major_ticks()
    for tick in xticks:
        tick.tick1On = False
        tick.tick2On = False
        tick.label1On = True
        tick.label2On = False


    p1 = axes.bar( x-0.4, random_mean, yerr=random_std, width=0.8, facecolor='r', **mpl_kwargs )
    p2 = axes.bar( x+0.4, replicate_mean, yerr=replicate_std, width=0.8, facecolor='y', **mpl_kwargs )

    axes.set_xlim( -2, 2 * len( labels ) )
    if xlabel != None:
        axes.set_xlabel( xlabel, rotation='0' )
    if ylabel == None:
        ylabel = 'similarity'
    axes.set_ylabel( ylabel, rotation='270' )
    axes.grid( True )

    leg = axes.legend( ( p1[0], p2[0] ), ('Random', 'Replicate'), fancybox=True )
    leg.get_frame().set_alpha( 0.25 )


    pp.savefig( fig )

    pp.close()

    plt.switch_backend( be )
