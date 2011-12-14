# -*- coding: utf-8 -*-

"""
print_engine.py -- PDF printing engine and printing functions.
"""

# This software is distributed under the FreeBSD License.
# See the accompanying file LICENSE for details.
# 
# Copyright 2011 Benjamin Hepp

import numpy
import subprocess
import sys
import os

import matplotlib.colors
import matplotlib.patches
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.transforms as mtransforms

import hcluster as hc
import dendrogram

class PdfDocument(object):
    def __init__(self, filename, pdf_backend='PDF'):
        self.__pp = PdfPages( filename )
        self.__fig = None
        self.__n = 0
        self.__rows = 1
        self.__cols = 1
        self.__m = 1
        self.__title = None
        self.__page_number = False
        self.__be = []
        pdf_backend = 'cairo.pdf'
        self.__pdf_backend = pdf_backend
    def __begin_pdf(self):
        self.__be.append( plt.get_backend() )
        plt.switch_backend(self.__pdf_backend)
        #print 'switched to backend: %s' % plt.get_backend()
    def __end_pdf(self):
        if len( self.__be ) > 0:
            plt.switch_backend( self.__be.pop() )
        else:
            raise Exception( 'A call to __end_pdf() must be preceded by a call to __begin_pdf()' )
    def get_pages(self):
        return self.__n
    def next_page(self, rows=None, cols=None, **kwargs):
        self.__begin_pdf()
        try:
            if rows != None and cols != None:
                self.__rows = rows
                self.__cols = cols
            self.__title = kwargs.get( 'title', self.__title )
            self.__page_number = kwargs.get( 'page_number', self.__page_number )
            self.__left_margin = kwargs.get( 'left_margin', 0.1 )
            self.__right_margin = kwargs.get( 'right_margin', 0.1 )
            self.__horizontal_spacing = kwargs.get( 'horizontal_spacing', 0.1 )
            self.__vertical_spacing = kwargs.get( 'vertical_spacing', 0.15 )
            self.__top_margin = kwargs.get( 'top_margin', 0.1 )
            self.__bottom_margin = kwargs.get( 'bottom_margin', 0.15 )
            self.__width = 1.0 - self.__left_margin - self.__right_margin
            self.__height = 1.0 - self.__top_margin - self.__bottom_margin
            self.__subplot_width = ( self.__width - ( self.__cols - 1 ) * self.__horizontal_spacing ) / float( self.__cols )
            self.__subplot_height = ( self.__height - ( self.__rows - 1 ) * self.__vertical_spacing ) / float( self.__rows )
            if self.__fig != None:
                self.__pp.savefig( self.__fig )
            self.__fig = plt.figure()
            self.__n += 1
            self.__m = 0
            if self.__title != None:
                self.__fig.suptitle( self.__title )
            if self.__page_number:
                self.__fig.text( 0.5, 0.02, ' Page %d' % self.__n, horizontalalignment='center', verticalalignment='bottom' )
        finally:
            self.__end_pdf()
        return self.__fig
    def next_plot(self, wrap=True, **kwargs):
        draw_frame = kwargs.get( 'draw_frame', True )
        if self.__m >= self.__cols * self.__rows:
            self.next_page()
        self.__begin_pdf()
        try:
            row = int( self.__m / self.__cols )
            col = int( self.__m % self.__cols )
            left = self.__left_margin + col * ( self.__subplot_width + self.__horizontal_spacing )
            top = 1.0 - ( self.__top_margin + row * ( self.__subplot_height + self.__vertical_spacing ) )
            axes = self.__fig.add_axes( [ left, top - self.__subplot_height, self.__subplot_width, self.__subplot_height ], frameon=draw_frame )
            self.__m += 1
        finally:
            self.__end_pdf()
        return axes
    def begin_next_page(self, rows=None, cols=None, **kwargs):
        fig = self.next_page( rows, cols, **kwargs )
        self.__begin_pdf()
        return fig
    def begin_next_plot(self, wrap=True, **kwargs):
        axes = self.next_plot( wrap, **kwargs )
        self.__begin_pdf()
        return axes
    def get_current_figure(self):
        return self.__fig
    def begin(self):
        self.__begin_pdf()
    def end(self):
        self.__end_pdf()
    def close(self):
        self.save()
        while len( self.__be ) > 0:
            self.end()
    def save(self):
        self.__begin_pdf()
        try:
            if self.__fig != None:
                self.__pp.savefig( self.__fig )
            self.__pp.close()
        finally:
            self.__end_pdf()



FILE_VIEWER = 'evince'

def view_file(filename):
    subprocess.Popen( [ FILE_VIEWER, filename ] )


def print_dendrogram(outputFilename, Z, labels=None, p=30, truncate_mode=None, color_threshold=None,
           get_leaves=True, orientation='top',
           count_sort=False, distance_sort=False, show_leaf_counts=True,
           no_plot=False, no_labels=False, color_list=None,
           leaf_font_size=None, leaf_rotation=45, leaf_label_func=None,
           no_leaves=False, show_contracted=False,
           link_color_func=None):

    pdfDocument = PdfDocument( outputFilename )

    title = 'Dendrogram of similarity map clustering'

    draw_dendrogram( pdfDocument, Z, labels, title, p, truncate_mode, color_threshold,
           get_leaves, orientation,
           count_sort, distance_sort, show_leaf_counts,
           no_plot, no_labels, color_list,
           leaf_font_size, leaf_rotation, leaf_label_func,
           no_leaves, show_contracted,
           link_color_func
    )

    pdfDocument.close()


def draw_dendrogram(pdfDocument, Z, labels=None, title=None, p=30, truncate_mode=None, color_threshold=None,
           get_leaves=True, orientation='top',
           count_sort=False, distance_sort=False, show_leaf_counts=True,
           no_plot=False, no_labels=False, color_list=None,
           leaf_font_size=None, leaf_rotation=45, leaf_label_func=None,
           no_leaves=False, show_contracted=False,
           link_color_func=None,
           xlabel=None, ylabel=None):

    if type( pdfDocument ) == PdfDocument:
        axes = pdfDocument.begin_next_plot()
    else:
        axes = pdfDocument

    if xlabel != None:
        axes.set_xlabel( xlabel, rotation='0' )
    if ylabel != None:
        axes.set_ylabel( ylabel, rotation='270' )

    if title != None:
        axes.set_title( title )
    dendrogram.dendrogram( axes, Z, p, truncate_mode, color_threshold,
           get_leaves, orientation, labels,
           count_sort, distance_sort, show_leaf_counts,
           no_plot, no_labels, color_list,
           leaf_font_size, leaf_rotation, leaf_label_func,
           no_leaves, show_contracted,
           link_color_func
    )

    if type( pdfDocument ) == PdfDocument:
        pdfDocument.end()



def print_cell_selection_plot(pipeline, outputFilename, mpl_kwargs={ 'align' : 'center' }):

    median_mahal_dist = numpy.empty( ( len( pipeline.validTreatmentIds ), ) )
    mad_mahal_dist = numpy.empty( ( len( pipeline.validTreatmentIds ), ) )
    median_mahal_cutoff_dist = numpy.empty( ( len( pipeline.validTreatmentIds ), ) )
    mad_mahal_cutoff_dist = numpy.empty( ( len( pipeline.validTreatmentIds ), ) )
    tr_labels = []

    for i in xrange( len( pipeline.validTreatmentIds ) ):
        tr = pipeline.validTreatments[ i ]
        total_mask = pipeline.get_cell_mask( 'valid', trId=tr.index )
        nonControl_mask = pipeline.get_cell_mask( 'valid', 'nonControl', trId=tr.index )
        median_mahal_dist[ i ] = numpy.median( pipeline.cell_selection_stats[ 'mahal_dist' ][ total_mask ] )
        mad_mahal_dist[ i ] = numpy.median( numpy.abs( pipeline.cell_selection_stats[ 'mahal_dist' ][ total_mask ] - median_mahal_dist[ i ] ) )
        median_mahal_cutoff_dist[ i ] = numpy.median( pipeline.cell_selection_stats[ 'mahal_dist' ][ nonControl_mask ] )
        mad_mahal_cutoff_dist[ i ] = numpy.median( numpy.abs( pipeline.cell_selection_stats[ 'mahal_dist' ][ nonControl_mask ] - median_mahal_cutoff_dist[ i ] ) )
        tr_labels.append( tr.name )

    import cPickle
    f = open(outputFilename, 'w')
    p = cPickle.Pickler(f)
    p.dump({
        'labels' : tr_labels,
        'total' : median_mahal_dist,
        'total_error' : mad_mahal_dist,
        'nonControl' : median_mahal_cutoff_dist,
        'nonControl_error' : mad_mahal_cutoff_dist
    })
    f.close()
    return

    print 'population_treatments:', tr_labels

    ctrl_median_mahal_dist = pipeline.cell_selection_stats[ 'ctrl_median_mahal_dist' ]
    ctrl_mad_mahal_dist = pipeline.cell_selection_stats[ 'ctrl_mad_mahal_dist' ]
    cutoff_mahal_dist_first = pipeline.cell_selection_stats[ 'cutoff_mahal_dist_first' ]
    cutoff_mahal_dist = pipeline.cell_selection_stats[ 'cutoff_mahal_dist' ]

    import StringIO
    str = StringIO.StringIO()
    str.write( 'ctrl median mahal dist\n%.2f\n' % ctrl_median_mahal_dist )
    str.write( 'ctrl mad mahal dist\n%.2f\n' % ctrl_mad_mahal_dist )
    str.write( 'cutoff mahal dist first\n%.2f\n' % cutoff_mahal_dist_first )
    str.write( 'cutoff mahal dist\n%.2f\n' % cutoff_mahal_dist )
    str.write( '\n' )
    str.write( 'treatments\n' )
    for i in xrange( len( tr_labels ) ):
        str.write( '%s\t' % tr_labels[ i ] )
    str.write( '\n\nmedian mahal dist\n' )
    for i in xrange( median_mahal_dist.shape[0] ):
        str.write( '%.2f\t' % median_mahal_dist[ i ] )
    str.write( '\n\nmad mahal dist\n' )
    for i in xrange( mad_mahal_dist.shape[0] ):
        str.write( '%.2f\t' % mad_mahal_dist[ i ] )
    str.write( '\n\nmedian mahal cutoff dist\n' )
    for i in xrange( median_mahal_cutoff_dist.shape[0] ):
        str.write( '%.2f\t' % median_mahal_cutoff_dist[ i ] )
    str.write( '\n\nmad mahal cutoff dist\n' )
    for i in xrange( mad_mahal_cutoff_dist.shape[0] ):
        str.write( '%.2f\t' % mad_mahal_cutoff_dist[ i ] )
    str.write( '\n' )

    f = open( os.path.splitext( outputFilename )[ 0 ] + '.xls', 'w' )
    f.write( str.getvalue() )
    f.close()

    str.close()

    error_kw = { 'ecolor' : 'green' }

    pdfDocument = PdfDocument(outputFilename)

    axes = pdfDocument.begin_next_plot()

    x = numpy.arange(0, len(tr_labels))
    axes.set_xticks(x)
    #xlabels = axes.set_xticklabels(tr_labels, rotation='270')
    xlabels = axes.set_xticklabels(tr_labels, rotation='270', fontsize=16)

    axes.set_title('Median Mahalanobis distance of valid cell objects')
    axes.bar(x, median_mahal_dist, yerr=mad_mahal_dist, error_kw=error_kw, **mpl_kwargs)

    axes.set_xlim(-1, len(tr_labels))
    axes.set_ylabel('Mahalanobis distance', rotation='270')
    axes.grid(True)

    axes = pdfDocument.next_plot()

    #if bottom_shift != None:
    #    fig.subplots_adjust( bottom=bottom_shift )

    x = numpy.arange( 0, len( tr_labels ) )
    axes.set_xticks( x )
    #xlabels = axes.set_xticklabels( tr_labels, rotation='270' )
    xlabels = axes.set_xticklabels(tr_labels, rotation='270', fontsize=16)

    axes.set_title( 'Median Mahalanobis distance of non-control-like cell objects' )
    axes.bar( x, median_mahal_cutoff_dist, yerr=mad_mahal_cutoff_dist, error_kw=error_kw, **mpl_kwargs )

    axes.set_xlim( -1, len( tr_labels ) )
    axes.set_ylabel( 'Mahalanobis distance', rotation='270' )
    axes.grid( True )

    pdfDocument.end()

    pdfDocument.close()


def print_clustering_plot(pipeline, outputFilename, mpl_kwargs={ 'align' : 'center', 'color' : 'green' }):

    labels = []
    cluster_population = numpy.empty( ( pipeline.nonControlClusters.shape[0], ) )
    for i in xrange( cluster_population.shape[0] ):
        pmask = pipeline.nonControlPartition == i
        cluster_population[ i ] = numpy.sum( pmask )
        labels.append( '%d' % i )

    import StringIO
    str = StringIO.StringIO()
    str.write( 'labels\n' )
    for i in xrange( len( labels ) ):
        str.write( '%s\t' % labels[ i ] )
    str.write( '\n\ncluster population\n' )
    for i in xrange( cluster_population.shape[0] ):
        str.write( '%.2f\t' % cluster_population[ i ] )
    str.write( '\n' )

    f = open( os.path.splitext( outputFilename )[ 0 ] + '.xls', 'w' )
    f.write( str.getvalue() )
    f.close()

    str.close()


    pdfDocument = PdfDocument( outputFilename )

    axes = pdfDocument.begin_next_plot()

    x = numpy.arange( 0, len( labels ) )
    axes.set_xticks( x )
    xlabels = axes.set_xticklabels( labels, rotation='270' )

    #axes.set_title( 'Total cell population' )
    axes.bar( x, cluster_population, **mpl_kwargs )

    axes.set_xlim( -1, len( labels ) )
    axes.set_xlabel( 'cluster', rotation='0' )
    axes.set_ylabel( '# of cells', rotation='270' )
    axes.grid( True )


    pdfDocument.end()

    pdfDocument.close()


def print_cell_populations_and_penetrance(pipeline, outputFilename, mpl_kwargs={ 'align' : 'center', 'color' : 'red' }):

    total_population = numpy.empty( ( len( pipeline.validTreatmentIds ), ) )
    nonControl_population = numpy.empty( ( len( pipeline.validTreatmentIds ), ) )
    penetrance = numpy.empty( ( len( pipeline.validTreatmentIds ), ) )
    tr_labels = []

    for i in xrange( total_population.shape[0] ):
        tr = pipeline.validTreatments[ i ]
        total_population[ i ] = numpy.sum( pipeline.get_cell_mask( 'valid', trId=tr.index ) )
        nonControl_population[ i ] = numpy.sum( pipeline.get_cell_mask( 'valid', 'nonControl', trId=tr.index ) )
        tr_labels.append( tr.name )
    penetrance = nonControl_population / total_population

    import cPickle
    f = open(outputFilename, 'w')
    p = cPickle.Pickler(f)
    p.dump({'labels' : tr_labels, 'total' : total_population, 'nonControl' : nonControl_population})
    f.close()
    return

    print 'population_treatments:', tr_labels

    import StringIO
    str = StringIO.StringIO()
    str.write( 'treatments\n' )
    for i in xrange( len( tr_labels ) ):
        str.write( '%s\t' % tr_labels[ i ] )
    str.write( '\n\ntotal population\n' )
    for i in xrange( total_population.shape[0] ):
        str.write( '%.2f\t' % total_population[ i ] )
    str.write( '\n\nnon-control population\n' )
    for i in xrange( total_population.shape[0] ):
        str.write( '%.2f\t' % total_population[ i ] )
    str.write( '\n\npenetrance\n' )
    for i in xrange( penetrance.shape[0] ):
        str.write( '%.2f\t' % penetrance[ i ] )
    str.write( '\n' )

    f = open( os.path.splitext( outputFilename )[ 0 ] + '.xls', 'w' )
    f.write( str.getvalue() )
    f.close()

    str.close()


    pdfDocument = PdfDocument( outputFilename )

    axes = pdfDocument.begin_next_plot()

    #if bottom_shift != None:
    #    fig.subplots_adjust( bottom=bottom_shift )

    x = numpy.arange( 0, len( tr_labels ) )
    axes.set_xticks( x )
    #xlabels = axes.set_xticklabels( tr_labels, rotation='270' )
    xlabels = axes.set_xticklabels(tr_labels, rotation='270', fontsize=16)

    axes.set_title( 'Population of valid cell objects' )
    axes.bar( x, total_population, **mpl_kwargs )

    axes.set_xlim( -1, len( tr_labels ) )
    axes.set_ylabel( '# of cells', rotation='270' )
    axes.grid( True )

    axes = pdfDocument.next_plot()

    x = numpy.arange( 0, len( tr_labels ) )
    axes.set_xticks( x )
    #xlabels = axes.set_xticklabels( tr_labels, rotation='270' )
    xlabels = axes.set_xticklabels(tr_labels, rotation='270', fontsize=16)

    axes.set_title( 'Population of non-control-like cell objects' )
    axes.bar( x, nonControl_population, **mpl_kwargs )

    axes.set_xlim( -1, len( tr_labels ) )
    axes.set_ylabel( '# of cells', rotation='270' )
    axes.grid( True )

    axes = pdfDocument.next_plot()

    x = numpy.arange( 0, len( tr_labels ) )
    axes.set_xticks( x )
    #xlabels = axes.set_xticklabels( tr_labels, rotation='270' )
    xlabels = axes.set_xticklabels(tr_labels, rotation='270', fontsize=16)

    axes.set_title( 'Apparent penetrance' )
    axes.bar( x, 100.0 * penetrance, **mpl_kwargs )

    axes.set_xlim( -1, len( tr_labels ) )
    axes.set_ylabel( 'penetrance in %', rotation='270' )
    axes.grid( True )


    pdfDocument.end()

    pdfDocument.close()


def print_feature_weights(feature_weights, outputFilename, title=None, barplot_kwargs={}):

    pdfDocument = PdfDocument( outputFilename )

    mpl_kwargs = barplot_kwargs
    if not mpl_kwargs.has_key( 'facecolor' ): mpl_kwargs[ 'facecolor' ] = 'yellow'
    if not mpl_kwargs.has_key( 'alpha' ): mpl_kwargs[ 'alpha' ] = 0.75
    if not mpl_kwargs.has_key( 'align' ): mpl_kwargs[ 'align' ] = 'center'

    pdfDocument.next_page( 1, 1, title=title )

    draw_feature_weights( pdfDocument, feature_weights, mpl_kwargs=barplot_kwargs )

    pdfDocument.close()

def draw_feature_weights(pdfDocument, feature_weights, mpl_kwargs={}):

    axes = pdfDocument.begin_next_plot()
    try:

        x = numpy.arange( 0, feature_weights.shape[0] )
        xlabels = []
        step = int( feature_weights.shape[0] / 40  ) + 1
        for j in xrange( x.shape[0] ):
            if j % step == 0:
                xlabels.append( str( x[ j ] ) )
            else:
                xlabels.append( '' )

        #axes.set_title( labels[ i ] )

        axes.bar( x, feature_weights, **mpl_kwargs )
        axes.set_xticks( x )
        xlabels = axes.set_xticklabels( xlabels, rotation='270' )

        xticks = axes.xaxis.get_major_ticks() + axes.xaxis.get_minor_ticks()
        for tick in xticks:
            tick.tick1On = False
            tick.tick2On = False
            tick.label1On = True
            tick.label2On = False

        axes.set_xlim( -1, feature_weights.shape[0] )

        #axes.set_xlabel( 'cluster ID' )
        axes.set_ylabel( 'weight', rotation='270' )

        #axes.grid( True )

    finally:
        pdfDocument.end()


def print_cluster_profiles_and_heatmap(labels, clusterProfiles, heatmap, outputFilename, normalize=False, profile_threshold=0.0, hcluster_method='average', barplot_kwargs={}, heatmap_kwargs={ 'lower' : True }, binSimilarityMatrix=None, xlabel=None, ylabel=None, random_split_dist=None, replicate_split_dist=None, replicate_distance_threshold=-1.0, cluster_mask=None, map_fontsize=6, map_label_fontsize=8, cluster_label_fontsize=8):

    import cPickle
    profiles = numpy.empty(clusterProfiles.shape, dtype=float)
    for i, profile in enumerate(clusterProfiles):
        abs_value = float(numpy.sum(profile))
        profiles[i] = profile / abs_value
    filename = '.'.join(outputFilename.split('.')[:-1]) + '.pic'
    f = open(filename, 'w')
    p = cPickle.Pickler(f)
    p.dump({
        'labels' : labels,
        'profiles' : profiles
    })
    f.close()

    pdfDocument = PdfDocument( outputFilename )

    mpl_kwargs = barplot_kwargs
    if not mpl_kwargs.has_key( 'facecolor' ): mpl_kwargs[ 'facecolor' ] = 'yellow'
    if not mpl_kwargs.has_key( 'alpha' ): mpl_kwargs[ 'alpha' ] = 0.75
    if not mpl_kwargs.has_key( 'align' ): mpl_kwargs[ 'align' ] = 'center'

    #title = 'Cluster profiles'
    #pdfDocument.next_page( 2, 1, title=title, page_number=True )
    pdfDocument.next_page( 2, 1 )

    draw_cluster_profiles( pdfDocument, labels, clusterProfiles, normalize=normalize, profile_threshold=profile_threshold, mpl_kwargs=barplot_kwargs )

    float_precision = 3

    if binSimilarityMatrix != None:

        pdfDocument.next_page( 1, 1, title='Cross-bin similarity matrix' )

        axes = pdfDocument.begin_next_plot()

        color_factory = lambda v: ( v, v, v )
        binLabels = []
        for i in xrange( binSimilarityMatrix.shape[0] ):
            binLabels.append( str( i ) )
        draw_lower_heatmap( axes, binSimilarityMatrix, ( 0, 0 ), binLabels, map_fontsize, color_factory, label_fontsize=map_label_fontsize, float_precision=float_precision )
        draw_upper_heatmap( axes, binSimilarityMatrix, ( 0, 0 ), binLabels, map_fontsize, color_factory, label_fontsize=map_label_fontsize, float_precision=float_precision )
        draw_diagonal_heatmap( axes, binSimilarityMatrix, ( 0, 0 ), binLabels, map_fontsize, color_factory, label_fontsize=map_label_fontsize, float_precision=float_precision )

    #pdfDocument.next_page( 1, 1, title='Heatmap of cluster profiles similarities', page_number=True )
    pdfDocument.next_page( 1, 1, title='Treatment distance matrix' )

    axes = pdfDocument.begin_next_plot()

    fig = pdfDocument.get_current_figure()
    fig.subplots_adjust( left=fig.subplotpars.left + 0.1, bottom=fig.subplotpars.bottom + 0.1 )

    #color_factory = lambda v: ( 1-v, 1-v, 1-v )
    color_factory = lambda v: ( v, v, v )
    #draw_treatment_similarity_map( pdfDocument, heatmap, ( 0, 0 ), labels, 6, color_factory, xlabel=xlabel, ylabel=ylabel )
    if heatmap_kwargs.get( 'lower', False ):
        draw_lower_heatmap( axes, heatmap, ( 0, 0 ), labels, map_fontsize, color_factory, xlabel=xlabel, ylabel=ylabel, label_fontsize=map_label_fontsize, float_precision=float_precision )
    if heatmap_kwargs.get( 'upper', False ):
        draw_upper_heatmap( axes, heatmap, ( 0, 0 ), labels, map_fontsize, color_factory, xlabel=xlabel, ylabel=ylabel, label_fontsize=map_label_fontsize, float_precision=float_precision )
    if heatmap_kwargs.get( 'diagonal', False ):
        draw_diagonal_heatmap( axes, heatmap, ( 0, 0 ), labels, map_fontsize, color_factory, xlabel=xlabel, ylabel=ylabel, label_fontsize=map_label_fontsize, float_precision=float_precision )

    """mpl_kwargs = heatmap_kwargs
    if not mpl_kwargs.has_key( 'color' ): mpl_kwargs[ 'color' ] = 'white'
    if not mpl_kwargs.has_key( 'linestyle' ): mpl_kwargs[ 'linestyle' ] = '-'
    if not mpl_kwargs.has_key( 'linewidth' ): mpl_kwargs[ 'linewidth' ] = 2
    if not mpl_kwargs.has_key( 'which' ): mpl_kwargs[ 'which' ] = 'minor'"""

    if random_split_dist != None:

        #pdfDocument.next_page( 1, 1, title='Heatmap of cluster profiles similarities', page_number=True )
        pdfDocument.next_page( 1, 1, title='Random split distances' )

        axes = pdfDocument.begin_next_plot()
        try:
            left = numpy.arange( random_split_dist.shape[0] )
            height = random_split_dist
            axes.bar( left, height, color='green', align='center' )
            axes.set_xlabel( xlabel )
            axes.set_xlim( -1, random_split_dist.shape[0] )
            axes.set_xticks( left )
            axes.set_xticklabels( labels, rotation='270', fontsize=map_label_fontsize )
            xticks = axes.xaxis.get_major_ticks() + axes.xaxis.get_minor_ticks()
            for tick in xticks:
                tick.tick1On = False
                tick.tick2On = False
                tick.label1On = True
                tick.label2On = False
            axes.set_ylabel( 'distance' )
            axes.set_ylim( 0.0, numpy.max( height ) * 1.1 )
            print 'height:', height
            print 'min_height:', numpy.min(height), 'max_height:', numpy.max(height)
        finally:
            pdfDocument.end()


    #pdfDocument.next_page( 1, 1, title='Clustering of treatments by similarity', page_number=True )
    pdfDocument.next_page( 1, 1, title=None )

    #distanceMap = 1.0 - heatmap
    #print distanceMap.shape
    try:
        distanceMap = heatmap.copy()
        distanceMap[ numpy.identity( distanceMap.shape[0], dtype=bool ) ] = 0.0
        cdm = hc.squareform( distanceMap )

        if type( hcluster_method ) != list:
            hcluster_method = [ hcluster_method ]

        if cdm.shape[0] > 0:
            for method in hcluster_method:
                Z = hc.linkage( cdm, method )
                draw_dendrogram( pdfDocument, Z, labels=labels, leaf_rotation=270, xlabel=xlabel, ylabel='similarity distance', title='%s clustering' % method, leaf_font_size=cluster_label_fontsize )
    except:
        pass


    if replicate_split_dist != None:

        #pdfDocument.next_page( 1, 1, title='Heatmap of cluster profiles similarities', page_number=True )
        pdfDocument.next_page( 1, 1, title='Replicate split distances' )

        axes = pdfDocument.begin_next_plot()
        try:
            left = numpy.arange( replicate_split_dist.shape[0] )
            height = replicate_split_dist
            axes.bar( left, height, color='green', align='center' )
            axes.set_xlabel( xlabel )
            axes.set_xlim( -1, replicate_split_dist.shape[0] )
            axes.set_xticks( left )
            axes.set_xticklabels( labels, rotation='270', fontsize=map_label_fontsize )
            xticks = axes.xaxis.get_major_ticks() + axes.xaxis.get_minor_ticks()
            for tick in xticks:
                tick.tick1On = False
                tick.tick2On = False
                tick.label1On = True
                tick.label2On = False
            axes.set_ylabel( 'distance' )
            axes.set_ylim( 0.0, numpy.max( height ) * 1.1 )
            import matplotlib.lines
            line = matplotlib.lines.Line2D( axes.get_xlim(), [ replicate_distance_threshold, replicate_distance_threshold ], linestyle='-', color='red' )
            axes.add_line( line )
            print 'height:', height
            print 'min_height:', numpy.min(height), 'max_height:', numpy.max(height)
        finally:
            pdfDocument.end()

        print "cluster_mask:", numpy.sum( cluster_mask ), cluster_mask.shape
        print cluster_mask

        pdfDocument.next_page( 1, 1, title=None )


        try:
            #distanceMap = 1.0 - heatmap
            #print distanceMap.shape
            distanceMap = heatmap.copy()
            distanceMap[ numpy.identity( distanceMap.shape[0], dtype=bool ) ] = 0.0
            if cluster_mask != None:
                distanceMap = distanceMap[ cluster_mask ][ : , cluster_mask ]
                new_labels = []
                for b,label in zip( cluster_mask, labels ):
                    if b:
                        new_labels.append( label )
                labels = new_labels
            cdm = hc.squareform( distanceMap )

            if type( hcluster_method ) != list:
                hcluster_method = [ hcluster_method ]

            if cdm.shape[0] > 0:
                for method in hcluster_method:
                    Z = hc.linkage( cdm, method )
                    draw_dendrogram( pdfDocument, Z, labels=labels, leaf_rotation=270, xlabel=xlabel, ylabel='similarity distance', title='%s clustering' % method, leaf_font_size=cluster_label_fontsize )
        except:
            pass


    pdfDocument.close()


def draw_cluster_profiles(pdfDocument, labels, clusterProfiles, normalize=False, profile_threshold=0.0, mpl_kwargs={}):

    for i in xrange( clusterProfiles.shape[0] ):

        #sys.stdout.write( '\r  treatment %s ...' % treatmentLabels[ i ] )
        #sys.stdout.flush()

        axes = pdfDocument.begin_next_plot()
        try:

            profile = clusterProfiles[ i ]

            if normalize:
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

            axes.set_title( labels[ i ] )

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

            axes.set_xlabel( 'cluster index' )
            axes.set_ylabel( 'population size', rotation='270' )

            #axes.grid( True )

        finally:
            pdfDocument.end()


def draw_diagonal_heatmap(pdfDocument, map, xy, labels, fontsize=12, color_factory=None, xlabel=None, ylabel=None, label_fontsize=10, float_precision=2):

    if type( pdfDocument ) == PdfDocument:
        axes = pdfDocument.begin_next_plot()
    else:
        axes = pdfDocument

    if xlabel != None:
        axes.set_xlabel( xlabel, rotation='0' )
    if ylabel != None:
        axes.set_ylabel( ylabel, rotation='270' )

    x = numpy.arange( map.shape[1] )
    axes.set_xticks( x, minor=False )
    axes.set_xlim( -0.5, map.shape[1] - 0.5)

    y = numpy.arange( map.shape[0] )
    axes.set_yticks( y, minor=False )
    axes.set_ylim( -0.5, map.shape[0] - 0.5)

    axes.set_xticklabels( labels, rotation='270', fontsize=label_fontsize )
    axes.set_yticklabels( labels[ ::-1 ], rotation='0', fontsize=label_fontsize )

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

    trans1 = mtransforms.Affine2D.identity()
    trans1 = trans1.scale( 1. / ( map.shape[0] ), 1. / ( map.shape[1] ) ).scale( 1.0, -1.0 ).translate( 0.0, 1.0 )
    trans = mtransforms.composite_transform_factory( trans1, axes.transAxes )

    masked_map = map[ numpy.isfinite( map ) ]
    if masked_map.shape[0] == 0:
        return

    if color_factory == None:
        color_factory = lambda v: ( v, v, v )
    min_value = numpy.min( map[ numpy.isfinite( map ) ] )
    max_value = numpy.max( map[ numpy.isfinite( map ) ] )
    if max_value == 0.0 or not numpy.isfinite( max_value ):
        max_value = 1.0
    if not numpy.isfinite( min_value ):
        min_value = 0.0
    print 'min_value:', min_value, 'max_value:', max_value
    for i in xrange( map.shape[0] ):
        if numpy.isfinite( map[ i, i ] ):
            string = ( '%%.%df' % float_precision ) % ( map[ i, i ] )
            v = ( map[ i, i ] - min_value ) / ( max_value - min_value )
            if not numpy.isfinite( v ):
                v = 0.0
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

    if type( pdfDocument ) == PdfDocument:
        pdfDocument.end()


def draw_upper_heatmap(pdfDocument, map, xy, labels, fontsize=12, color_factory=None, xlabel=None, ylabel=None, label_fontsize=10, float_precision=2):

    if type( pdfDocument ) == PdfDocument:
        axes = pdfDocument.begin_next_plot()
    else:
        axes = pdfDocument

    if xlabel != None:
        axes.set_xlabel( xlabel, rotation='0' )
    if ylabel != None:
        axes.set_ylabel( ylabel, rotation='270' )

    x = numpy.arange( map.shape[1] )
    axes.set_xticks( x, minor=False )
    axes.set_xlim( -0.5, map.shape[1] - 0.5)

    y = numpy.arange( map.shape[0] )
    axes.set_yticks( y, minor=False )
    axes.set_ylim( -0.5, map.shape[0] - 0.5)

    axes.set_xticklabels( labels, rotation='270', fontsize=label_fontsize )
    axes.set_yticklabels( labels[ ::-1 ], rotation='0', fontsize=label_fontsize )

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

    trans1 = mtransforms.Affine2D.identity()
    trans1 = trans1.scale( 1. / ( map.shape[0] ), 1. / ( map.shape[1] ) ).scale( 1.0, -1.0 ).translate( 0.0, 1.0 )
    trans = mtransforms.composite_transform_factory( trans1, axes.transAxes )

    masked_map = map[ numpy.isfinite( map ) ]
    if masked_map.shape[0] == 0:
        return

    if color_factory == None:
        color_factory = lambda v: ( v, v, v )
    min_value = numpy.min( map[ numpy.isfinite( map ) ] )
    max_value = numpy.max( map[ numpy.isfinite( map ) ] )
    if max_value == 0.0 or not numpy.isfinite( max_value ):
        max_value = 1.0
    if not numpy.isfinite( min_value ):
        min_value = 0.0
    print 'min_value:', min_value, 'max_value:', max_value
    for i in xrange( map.shape[0] ):
        for j in xrange( i, map.shape[0] ):
            if numpy.isfinite( map[ i, j ] ):
                string = ( '%%.%df' % float_precision ) % ( map[ i, j ] )
                v = ( map[ i, j ] - min_value ) / ( max_value - min_value )
                if not numpy.isfinite( v ):
                    v = 0.0
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

    if type( pdfDocument ) == PdfDocument:
        pdfDocument.end()


def draw_lower_heatmap(pdfDocument, map, xy, labels, fontsize=12, color_factory=None, xlabel=None, ylabel=None, label_fontsize=10, float_precision=2):

    if type( pdfDocument ) == PdfDocument:
        axes = pdfDocument.begin_next_plot()
    else:
        axes = pdfDocument

    if xlabel != None:
        axes.set_xlabel( xlabel, rotation='0' )
    if ylabel != None:
        axes.set_ylabel( ylabel, rotation='270' )

    x = numpy.arange( map.shape[1] )
    axes.set_xticks( x, minor=False )
    axes.set_xlim( -0.5, map.shape[1] - 0.5)

    y = numpy.arange( map.shape[0] )
    axes.set_yticks( y, minor=False )
    axes.set_ylim( -0.5, map.shape[0] - 0.5)

    axes.set_xticklabels( labels, rotation='270', fontsize=label_fontsize )
    axes.set_yticklabels( labels[ ::-1 ], rotation='0', fontsize=label_fontsize )

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

    trans1 = mtransforms.Affine2D.identity()
    trans1 = trans1.scale( 1. / ( map.shape[0] ), 1. / ( map.shape[1] ) ).scale( 1.0, -1.0 ).translate( 0.0, 1.0 )
    trans = mtransforms.composite_transform_factory( trans1, axes.transAxes )

    masked_map = map[ numpy.isfinite( map ) ]
    if masked_map.shape[0] == 0:
        return

    if color_factory == None:
        color_factory = lambda v: ( v, v, v )
    min_value = numpy.min( map[ numpy.isfinite( map ) ] )
    max_value = numpy.max( map[ numpy.isfinite( map ) ] )
    if max_value == 0.0 or not numpy.isfinite( max_value ):
        max_value = 1.0
    if not numpy.isfinite( min_value ):
        min_value = 0.0
    print 'min_value:', min_value, 'max_value:', max_value
    for i in xrange( map.shape[0] ):
        for j in xrange( i ):
            if numpy.isfinite( map[ i, j ] ):
                string = ( '%%.%df' % float_precision ) % ( map[ i, j ] )
                v = ( map[ i, j ] - min_value ) / ( max_value - min_value )
                if not numpy.isfinite( v ):
                    v = 0.0
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

    if type( pdfDocument ) == PdfDocument:
        pdfDocument.end()


def draw_treatment_similarity_map(pdfDocument, map, xy, labels, fontsize=12, color_factory=None, xlabel=None, ylabel=None, label_fontsize=10, float_precision=2):

    if type( pdfDocument ) == PdfDocument:
        axes = pdfDocument.begin_next_plot()
    else:
        axes = pdfDocument

    if xlabel != None:
        axes.set_xlabel( xlabel, rotation='0' )
    if ylabel != None:
        axes.set_ylabel( ylabel, rotation='270' )

    x = numpy.arange( map.shape[1] )
    axes.set_xticks( x, minor=False )
    axes.set_xlim( -0.5, map.shape[1] - 0.5)

    y = numpy.arange( map.shape[0] )
    axes.set_yticks( y, minor=False )
    axes.set_ylim( -0.5, map.shape[0] - 0.5)

    axes.set_xticklabels( labels, rotation='270', fontsize=label_fontsize )
    axes.set_yticklabels( labels[ ::-1 ], rotation='0', fontsize=label_fontsize )

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

    trans1 = mtransforms.Affine2D.identity()
    trans1 = trans1.scale( 1. / ( map.shape[0] ), 1. / ( map.shape[1] ) ).scale( 1.0, -1.0 ).translate( 0.0, 1.0 )
    trans = mtransforms.composite_transform_factory( trans1, axes.transAxes )

    #print 'original transformation:'
    #print axes.transAxes.transform( ( 0, 0 ) )
    #print axes.transAxes.transform( ( 0.5, 0.5 ) )
    #print axes.transAxes.transform( ( 0, 1 ) )
    #print axes.transAxes.transform( ( 1, 0 ) )
    #print 'transformation:'
    #print trans1.transform( ( 0, 0 ) )
    #print trans1.transform( ( ( map.shape[0])/2.0, ( map.shape[0])/2.0 ) )
    #print trans1.transform( ( 0, ( map.shape[0]) ) )
    #print trans1.transform( ( map.shape[0], 0 ) )
    #print 'composite transformation:'
    #print trans.transform( ( 0, 0 ) )
    #print trans.transform( ( ( map.shape[0])/2.0, ( map.shape[0])/2.0 ) )
    #print trans.transform( ( 0, ( map.shape[0]) ) )
    #print trans.transform( ( map.shape[0], 0 ) )

    #print 'draw_treatment_similarity_map'

    if color_factory == None:
        color_factory = lambda v: ( v, v, v )
    masked_map = map[ numpy.isfinite( map ) ]
    if masked_map.shape[0] == 0:
        return

    max_value = numpy.max( map[ numpy.isfinite( map ) ] )
    if max_value == 0.0 or not numpy.isfinite( max_value ):
        max_value = 1.0
    print 'max_value:', max_value
    for i in xrange( map.shape[0] ):
        #for j in xrange( map.shape[0]-1-i, -1, -1 ):
        for j in xrange( 0, i ):
            if numpy.isfinite( map[ i, j ] ):
                string = ( '%%.%df' % float_precision ) % ( map[ i, j ] )
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

    if type( pdfDocument ) == PdfDocument:
        pdfDocument.end()


def draw_modified_treatment_similarity_map(pdfDocument, map, xy, labels, fontsize=12, color_factory=None):

    if type( pdfDocument ) == PdfDocument:
        axes = pdfDocument.begin_next_plot()
    else:
        axes = pdfDocument

    x = numpy.arange( map.shape[1] )
    axes.set_xticks( x, minor=False )
    axes.set_xlim( -0.5, map.shape[1] - 0.5)

    y = numpy.arange( map.shape[0] )
    axes.set_yticks( y, minor=False )
    axes.set_ylim( -0.5, map.shape[0] - 0.5)

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

    trans1 = mtransforms.Affine2D.identity()
    trans1 = trans1.scale( 1. / ( map.shape[0] ), 1. / ( map.shape[1] ) ).scale( 1.0, -1.0 ).translate( 0.0, 1.0 )
    trans = mtransforms.composite_transform_factory( trans1, axes.transAxes )

    if color_factory == None:
        color_factory = lambda v: ( v, v, v )
    masked_map = map[ numpy.isfinite( map ) ]
    if masked_map.shape[0] == 0:
        return

    max_value = numpy.max( masked_map )
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

    if type( pdfDocument ) == PdfDocument:
        pdfDocument.end()


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

    print treatmentSimilarityMap
    #draw_lower_heatmap( axes, treatmentSimilarityMap, ( -0.5, -0.5 ), 12 )
    #draw_treatment_similarity_map( axes, treatmentSimilarityMap, ( 0, 0 ), 12, color_factory )
    color_factory = lambda v: ( 1-v, 1-v, 1-v )
    draw_modified_treatment_similarity_map( axes, treatmentSimilarityMap, ( 0, 0 ), labels, 12, color_factory )

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


    import StringIO
    str = StringIO.StringIO()
    str.write( 'treatments\n' )
    for i in xrange( len( labels ) ):
        str.write( '%s\t' % labels[ i ] )
    str.write( '\n\nrandom mean\n' )
    for i in xrange( random_mean.shape[0] ):
        str.write( '%.2f\t' % random_mean[ i ] )
    str.write( '\n\nrandom std\n' )
    for i in xrange( random_std.shape[0] ):
        str.write( '%.2f\t' % random_std[ i ] )
    str.write( '\n\nreplicate mean\n' )
    for i in xrange( replicate_mean.shape[0] ):
        str.write( '%.2f\t' % replicate_mean[ i ] )
    str.write( '\n\nreplicate std\n' )
    for i in xrange( replicate_std.shape[0] ):
        str.write( '%.2f\t' % replicate_std[ i ] )
    str.write( '\n' )

    f = open( os.path.splitext( outputFilename )[ 0 ] + '.xls', 'w' )
    f.write( str.getvalue() )
    f.close()

    str.close()


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

def print_map_quality_plot(map_qualities, labels, outputFilename, title='Map quality plot', xlabel=None, ylabel=None, mpl_kwargs={}):

    pdfDocument = PdfDocument( outputFilename )

    axes = pdfDocument.begin_next_plot()

    if mpl_kwargs.has_key( 'facecolor' ): del mpl_kwargs[ 'facecolor' ]
    if not mpl_kwargs.has_key( 'alpha' ): mpl_kwargs[ 'alpha' ] = 0.75
    #if not mpl_kwargs.has_key( 'align' ): mpl_kwargs[ 'align' ] = 'center'

    x = numpy.arange( len( labels ) )
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


    p1 = axes.bar( x, map_qualities, align='center', color='r', **mpl_kwargs )

    axes.set_xlim( -1, len( labels ) )
    if xlabel != None:
        axes.set_xlabel( xlabel, rotation='0' )
    if ylabel == None:
        ylabel = 'similarity'
    axes.set_ylabel( ylabel, rotation='270' )
    axes.grid( True )

    #leg = axes.legend( ( p1[0], p2[0] ), ('Random', 'Replicate'), fancybox=True )
    #leg.get_frame().set_alpha( 0.25 )


    pdfDocument.end()

    pdfDocument.close()
