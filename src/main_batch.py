# -*- coding: utf-8 -*-

"""
main_batch.py -- Runs the batch mode processing.
"""

# This software is distributed under the FreeBSD License.
# See the accompanying file LICENSE for details.
# 
# Copyright 2011 Benjamin Hepp

import sys, os
import time
import numpy

import hcluster as hc

config_file = None

pdf_map_fontsize=6
pdf_map_label_fontsize=8
pdf_cluster_label_fontsize=10
create_random_splits = False
num_of_random_splits = 3
size_of_random_splits = 0.5
similarity_map_clustering_method = ['average']
clustering_method_list = [ 'k-means' ]
clustering_index_list = [ 0 ]
clustering_param1_list = [ 100 ]
clustering_param2_list = [ -1 ]
clustering_param3_list = [ 2 ]
clustering_param4_list = [ 20 ]
clustering_exp_factor_list = [ -1 ]
normalizationFactor = 1.0
clustering_file_template = 'clustering_%(method)s_(%(index)d)_(%(param1)d,%(param2)d,%(param3)d,%(param4)d),%(exp_factor).1f.pic'
profile_file_template = 'profiles_%(method)s_(%(index)d)_(%(param1)d,%(param2)d,%(param3)d,%(param4)d),%(exp_factor).1f.pdf'
similarity_map_file_template = 'similarity_map_%(method)s_(%(index)d)_(%(param1)d,%(param2)d,%(param3)d,%(param4)d),%(exp_factor).1f.pdf'
combined_map_file_template = 'combined_map_%(method)s_(%(index)d)_(%(param1)d,%(param2)d,%(param3)d,%(param4)d),%(exp_factor).1f.pdf'
map_quality_random_file_template = 'map_quality_random_%(method)s_(%(index)d)_(%(param1)d,%(param2)d,%(param3)d,%(param4)d),%(exp_factor).1f.pdf'
map_quality_replicate_file_template = 'map_quality_replicate_%(method)s_(%(index)d)_(%(param1)d,%(param2)d,%(param3)d,%(param4)d),%(exp_factor).1f.pdf'
map_quality_results_file_template = 'map_quality_results_%(method)s_(%(index)d)_(%(param2)d,%(param3)d,%(param4)d),%(exp_factor).1f.pdf'
analyse_plot_file_template = 'analyse_plot_%(method)s_(%(index)d)_(%(param2)d,%(param3)d,%(param4)d),%(exp_factor).1f.pdf'
analyse2_plot_file_template = 'analyse2_plot_%(method)s_(%(index)d)_(%(param1)d,%(param2)d,%(param3)d,%(param4)d),%(exp_factor).1f.pdf'
analyse3_plot_file_template = 'analyse3_plot_%(method)s_(%(index)d)_(%(param2)d,%(param3)d,%(param4)d),%(exp_factor).1f.pdf'
analyse_map_quality_plot_file_template = 'analyse_map_quality_plot_%(method)s_(%(index)d)_(%(param2)d,%(param3)d,%(param4)d),%(exp_factor).1f.pdf'
only_run_clustering = False
combine_maps = False
only_combine_maps = False
do_map_quality = False
only_combine_maps = False
analyse_map_quality = False
only_analyse_map_quality = False
clustering_file = None
clustering_files = []
pipeline_state_file = 'pipeline_states/pipeline_state.pis'
population_plot_file = 'population_plots/population_plot.pdf'
selection_plot_file = 'selection_plots/selection_plot.pdf'
print_population_plot = False
print_selection_plot = False
analyse_maps = False
only_analyse_maps = False
analyse2_maps = False
only_analyse2_maps = False
analyse3_maps = False
only_analyse3_maps = False
penetrance_threshold = 0.0
replicate_distance_threshold = -1.0
#profile_metric = 'summed_minmax'
control_filter_mode = 'MEDIAN_xMAD'
job_state_filename_template = 'states/state_job_%(job-id)s.txt'

skip_next = 0


get_groups_function_code = """
def get_groups_function(pl):
    groups = []
    for tr in pl.pdc.treatments:
        tr_mask = pl.get_treatment_cell_mask( tr.index )
        groups.append( ( tr.name, tr_mask ) )
    return groups
"""
get_control_groups_function_code = """
def get_control_groups_function(pl):
    control_groups = []
    for tr in pl.pdc.treatments:
        tr_mask = pl.get_treatment_cell_mask( tr.index )
        control_mask = pl.get_control_treatment_cell_mask()
        tr_control_mask = pl.mask_and( tr_mask, control_mask )
        if numpy.all( tr_control_mask == tr_mask ):
            for repl in pl.pdc.replicates:
                repl_mask = pl.get_replicate_cell_mask( repl.index )
                mask = pl.mask_and( tr_control_mask, repl_mask )
                if numpy.sum( mask ) > 0:
                    name = tr.name + ( '_%d' % repl.index )
                    control_groups.append( ( name, mask ) )
    return control_groups
"""


def print_help():

    sys.stderr.write( """Usage: python %s [options]
Necessary options:
  --config-file <filename>      Configuration file for batch processing.
""" % sys.argv[ 0 ] )


def make_clustering_param_list(x):
    if type( x ) == str:
        l1 = []
        x = x.split()
        for y in x:
            for z in y.split( ',' ):
                z = z.strip()
                if z:
                    l1.append( z )
        l2 = []
        for y in l1:
            for z in y.split( ';' ):
                z = z.strip()
                if z:
                    l2.append( z )
        x = l2
    elif type( x ) == float or type( x ) == int:
        return [ x ]
    l = []
    for y in x:
        try:
            l.append( int( y ) )
        except:
            try:
                l.append( float( y ) )
            except:
                l.append( y )
    return l

if len( sys.argv ) > 1:
    for i in xrange( 1, len( sys.argv ) ):

        arg = sys.argv[ i ]
        if i < len( sys.argv ) - 1:
            next_arg = sys.argv[ i+1 ]
        else:
            next_arg = None

        if skip_next > 0:
            skip_next -= 1
            continue

        if arg == '--clustering-method':
            clustering_method_list = next_arg
            skip_next = 1
        elif arg == '--clustering-index':
            clustering_index_list = make_clustering_param_list( next_arg )
            skip_next = 1
        elif arg == '--clustering-param1':
            clustering_param1_list = make_clustering_param_list( next_arg )
            skip_next = 1
        elif arg == '--clustering-param2':
            clustering_param2_list = make_clustering_param_list( next_arg )
            skip_next = 1
        elif arg == '--clustering-param3':
            clustering_param3_list = make_clustering_param_list( next_arg )
            skip_next = 1
        elif arg == '--clustering-param4':
            clustering_param4_list = make_clustering_param_list( next_arg )
            skip_next = 1
        elif arg == '--clustering-exp-factor':
            clustering_exp_factor_list = make_clustering_param_list( next_arg )
            skip_next = 1
        elif arg == '--config-file':
            config_file = next_arg
            skip_next = 1
        elif arg == '--help':
            print_help()
            sys.exit( 0 )
        elif arg == '--job-id':
            try:
                job_id = str( int( next_arg ) )
            except:
                job_id = 'invalid'
            job_state_filename = job_state_filename_template % { 'job-id' : job_id }
            skip_next = 1
        elif arg != '':
            sys.stderr.write( 'Unknown option: %s\n' % arg )
            print_help()
            sys.exit( -1 )

if config_file == None:
    print 'YACA batch configuration file needs to be specified...'
    print_help()
    sys.exit( 1 )



from PyQt4.QtCore import *


from . core import pipeline
from . core import importer
from . core import headless_cluster_configuration
from . core.batch import utils as batch_utils

from . gui import print_engine

from . core import parameter_utils as utils


import yaml

file = None
try:
    file = open( config_file, 'r' )
    yaml_container = yaml.load( file )
finally:
    if file:
        file.close()


try:
    general_config = yaml_container[ 'general_config' ]
    path_prefix = general_config[ 'path_prefix' ]
    if 'get_groups_function_code' in general_config and general_config[ 'get_groups_function_code' ] != None:
        get_groups_function_code = general_config[ 'get_groups_function_code' ]
    exec get_groups_function_code
    if 'get_control_groups_function_code' in general_config and general_config[ 'get_control_groups_function_code' ] != None:
        get_control_groups_function_code = general_config[ 'get_control_groups_function_code' ]
    exec get_control_groups_function_code
    if 'pdf_map_fontsize' in general_config: pdf_map_fontsize = int( general_config[ 'pdf_map_fontsize' ] )
    if 'pdf_map_label_fontsize' in general_config: pdf_map_label_fontsize = int( general_config[ 'pdf_map_label_fontsize' ] )
    if 'pdf_cluster_label_fontsize' in general_config: pdf_cluster_label_fontsize = int( general_config[ 'pdf_cluster_label_fontsize' ] )
    if 'penetrance_threshold' in general_config: penetrance_threshold = general_config[ 'penetrance_threshold' ]
    if 'replicate_distance_threshold' in general_config: replicate_distance_threshold = general_config[ 'replicate_distance_threshold' ]
    if 'similarity_map_clustering_method' in general_config: similarity_map_clustering_method = general_config[ 'similarity_map_clustering_method' ]
    if 'only_analyse_maps' in general_config: only_analyse_maps = general_config[ 'only_analyse_maps' ]
    if 'analyse_maps' in general_config: analyse_maps = general_config[ 'analyse_maps' ]
    if 'only_analyse2_maps' in general_config: only_analyse2_maps = general_config[ 'only_analyse2_maps' ]
    if 'analyse2_maps' in general_config: analyse2_maps = general_config[ 'analyse2_maps' ]
    if 'only_analyse3_maps' in general_config: only_analyse3_maps = general_config[ 'only_analyse3_maps' ]
    if 'analyse3_maps' in general_config: analyse3_maps = general_config[ 'analyse3_maps' ]
    if not ( analyse_maps and only_analyse_maps ) and not ( analyse2_maps and only_analyse2_maps ) and not ( analyse3_maps and only_analyse3_maps ) and not ( do_map_quality and only_map_quality ) and not ( analyse_map_quality and only_analyse_map_quality ):
        project_file = general_config[ 'project_file' ]
    if 'create_random_splits' in general_config: create_random_splits = general_config[ 'create_random_splits' ]
    if 'num_of_random_splits' in general_config: num_of_random_splits = int( general_config[ 'num_of_random_splits' ] )
    if 'size_of_random_splits' in general_config: size_of_random_splits = float( general_config[ 'size_of_random_splits' ] )
    #treatments_to_use = general_config[ 'treatments_to_use' ]
    if 'control_filter_mode' in general_config: control_filter_mode = general_config[ 'control_filter_mode' ]
    if 'random_split_file' in general_config: random_split_file = general_config[ 'random_split_file' ]
    if 'profile_file_template' in general_config: profile_file_template = general_config[ 'profile_file_template' ]
    if 'random_profile_file_template' in general_config: random_profile_file_template = general_config[ 'random_profile_file_template' ]
    if 'replicate_profile_file_template' in general_config: replicate_profile_file_template = general_config[ 'replicate_profile_file_template' ]
    if 'control_profile_file_template' in general_config: control_profile_file_template = general_config[ 'control_profile_file_template' ]
    if 'similarity_map_file_template' in general_config: similarity_map_file_template = general_config[ 'similarity_map_file_template' ]
    if 'combined_map_file_template' in general_config: combined_map_file_template = general_config[ 'combined_map_file_template' ]
    if 'map_quality_file_template' in general_config: map_quality_file_template = general_config[ 'map_quality_file_template' ]
    if 'profile_files' in general_config: profile_files = general_config[ 'profile_files' ]
    if 'map_quality_random_file_template' in general_config: map_quality_random_file_template = general_config[ 'map_quality_random_file_template' ]
    if 'map_quality_replicate_file_template' in general_config: map_quality_replicate_file_template = general_config[ 'map_quality_replicate_file_template' ]
    if 'map_quality_results_file_template' in general_config: map_quality_results_file_template = general_config[ 'map_quality_results_file_template' ]
    if 'analyse_map_quality_plot_file_template' in general_config: analyse_map_quality_plot_file_template = general_config[ 'analyse_map_quality_plot_file_template' ]
    if 'analyse_plot_file_template' in general_config: analyse_plot_file_template = general_config[ 'analyse_plot_file_template' ]
    if 'analyse2_plot_file_template' in general_config: analyse2_plot_file_template = general_config[ 'analyse2_plot_file_template' ]
    if 'analyse3_plot_file_template' in general_config: analyse3_plot_file_template = general_config[ 'analyse3_plot_file_template' ]
    if 'only_run_clustering' in general_config: only_run_clustering = general_config[ 'only_run_clustering' ]
    if 'only_combine_maps' in general_config: only_combine_maps = general_config[ 'only_combine_maps' ]
    if 'combine_maps' in general_config: combine_maps = general_config[ 'combine_maps' ]
    if 'only_map_quality' in general_config: only_map_quality = general_config[ 'only_map_quality' ]
    if 'do_map_quality' in general_config: do_map_quality = str( general_config[ 'do_map_quality' ] ).lower() == 'true'
    if do_map_quality: only_map_quality = True
    if 'analyse_map_quality' in general_config: analyse_map_quality = general_config[ 'analyse_map_quality' ]
    if 'only_analyse_map_quality' in general_config: only_analyse_map_quality = general_config[ 'only_analyse_map_quality' ]
    if combine_maps and only_combine_maps:
        do_map_quality = False
        analyse_maps = False
        analyse2_maps = False
        analyse3_maps = False
        analyse_map_quality = False
        similarity_map_files = general_config[ 'similarity_map_files' ]
    else:
        similarity_map_files = []
    if do_map_quality and only_map_quality:
        combine_maps = False
        analyse_maps = False
        analyse2_maps = False
        analyse3_maps = False
        analyse_map_quality = False
        #random_similarity_map_dict2 = general_config[ 'random_similarity_map_dict' ]
        #replicate_similarity_map_dict2 = general_config[ 'replicate_similarity_map_dict' ]
    else:
        random_similarity_map_dict2 = {}
        replicate_similarity_map_dict2 = {}
    if analyse_maps and only_analyse_maps:
        combine_maps = False
        do_map_quality = False
        analyse2_maps = False
        analyse3_maps = False
        analyse_map_quality = False
        random_combined_map_files = general_config[ 'random_combined_map_files' ]
        replicate_combined_map_files = general_config[ 'replicate_combined_map_files' ]
    else:
        random_combined_map_files = []
        replicate_combined_map_files = []
    if analyse2_maps and only_analyse2_maps:
        combine_maps = False
        do_map_quality = False
        analyse_maps = False
        analyse3_maps = False
        analyse_map_quality = False
        random_similarity_map_files = general_config[ 'random_similarity_map_files' ]
        replicate_similarity_map_files = general_config[ 'replicate_similarity_map_files' ]
    else:
        random_similarity_map_files = []
        replicate_similarity_map_files = []
    if analyse3_maps and only_analyse3_maps:
        combine_maps = False
        do_map_quality = False
        analyse_maps = False
        analyse2_maps = False
        analyse_map_quality = False
        random_similarity_map_dict = general_config[ 'random_similarity_map_dict' ]
        replicate_similarity_map_dict = general_config[ 'replicate_similarity_map_dict' ]
    else:
        random_similarity_map_dict = {}
        replicate_similarity_map_dict = {}
    if analyse_map_quality and only_analyse_map_quality:
        combine_maps = False
        do_map_quality = False
        analyse_maps = False
        analyse2_maps = False
        analyse3_maps = False
        map_quality_files = general_config[ 'map_quality_files' ]
    else:
        map_quality_files = []
    if 'clustering_file' in general_config: clustering_file = general_config[ 'clustering_file' ]
    if 'print_population_plot' in general_config: print_population_plot = general_config[ 'print_population_plot' ]
    if 'pipeline_state_file_template' in general_config: pipeline_state_file = general_config[ 'pipeline_state_file_template' ]
    if print_population_plot:
        if 'population_plot_file_template' in general_config: population_plot_file = general_config[ 'population_plot_file_template' ]
    if 'print_selection_plot' in general_config: print_selection_plot = general_config[ 'print_selection_plot' ]
    if print_selection_plot:
        if 'selection_plot_file_template' in general_config: selection_plot_file = general_config[ 'selection_plot_file_template' ]
except:
    print 'Invalid YACA batch configuration file'
    raise

try:
    clustering_config = yaml_container[ 'clustering_config' ]
    if 'clustering_files' in clustering_config: clustering_files = clustering_config[ 'clustering_files' ]
    if 'method' in clustering_config: clustering_method_list = make_clustering_param_list( clustering_config[ 'method' ] )
    if 'index' in clustering_config: clustering_index_list = make_clustering_param_list( clustering_config[ 'index' ] )
    if 'param1' in clustering_config: clustering_param1_list = make_clustering_param_list( clustering_config[ 'param1' ] )
    if 'param2' in clustering_config: clustering_param2_list = make_clustering_param_list( clustering_config[ 'param2' ] )
    if 'param3' in clustering_config: clustering_param3_list = make_clustering_param_list( clustering_config[ 'param3' ] )
    if 'param4' in clustering_config: clustering_param4_list = make_clustering_param_list( clustering_config[ 'param4' ] )
    if 'profile_metric' in clustering_config: clustering_profile_metric_list = make_clustering_param_list( clustering_config[ 'profile_metric' ] )
    if 'normalizationFactor' in clustering_config: normalizationFactor = float( clustering_config[ 'normalizationFactor' ] )
    if 'exp_factor' in clustering_config: clustering_exp_factor_list = make_clustering_param_list( clustering_config[ 'exp_factor' ] )
    if 'file_template' in clustering_config: clustering_file_template = clustering_config[ 'file_template' ]
except:
    #raise Exception( 'Invalid YACA batch configuration file' )
    print 'Invalid YACA batch configuration file'
    raise

clustering_param_set = []
def recursive_make_clustering_param_set(clustering_param_lists, clustering_param_names, clustering_param_values, clustering_param_set):
    if len( clustering_param_names ) > 0:
        param_name = clustering_param_names[ 0 ]
        param_list = clustering_param_lists[ param_name ]
        for p in param_list:
            recursive_make_clustering_param_set(
                                clustering_param_lists,
                                clustering_param_names[ 1: ],
                                clustering_param_values + [ ( param_name, p ) ],
                                clustering_param_set
            )
    else:
        clustering_param_dict = {}
        for n,p in clustering_param_values:
            clustering_param_dict[ n ] = p
        clustering_param_set.append( clustering_param_dict )
clustering_param_lists = {
            'method' : clustering_method_list,
            'index' : clustering_index_list,
            'param1' : clustering_param1_list,
            'param2' : clustering_param2_list,
            'param3' : clustering_param3_list,
            'param4' : clustering_param4_list,
            'exp_factor' : clustering_exp_factor_list,
            'profile_metric' : clustering_profile_metric_list
}
recursive_make_clustering_param_set(
                clustering_param_lists,
                clustering_param_lists.keys(),
                [],
                clustering_param_set
)

#for d in clustering_param_set:
#    print d


print 'Starting batch processing...'


treatment_similarity_maps = {}


if not ( combine_maps and only_combine_maps ) and not ( do_map_quality and only_map_quality ) and not ( analyse_maps and only_analyse_maps ) and not ( analyse2_maps and only_analyse2_maps ) and not ( analyse3_maps and only_analyse3_maps ):


    print 'Running pipeline for project file %s' % ( project_file )

    headlessClusterConfiguration = headless_cluster_configuration.HeadlessClusterConfiguration()

    utils.load_module_configuration( project_file )


    modules = utils.list_modules()
    for module in modules:

        if not utils.all_parameters_set( module ):
            print 'Not all required parameters for module %s have been set' % module
            sys.exit( 1 )

        elif not utils.all_requirements_met( module ):
            print 'Not all requirements for module %s have been fulfilled' % module
            sys.exit( 1 )


    pdc = importer.Importer().get_pdc()
    clusterConfiguration = headlessClusterConfiguration.clusterConfiguration


    pl = pipeline.Pipeline( pdc, clusterConfiguration )


    def callback_pipeline_update_progress(progress):

        sys.stdout.write( '\r  progress: %d %%...' % progress )
        sys.stdout.flush()


    pl.connect( pl, SIGNAL('updateProgress'), callback_pipeline_update_progress )
    #pl.connect( pl, SIGNAL('finished()'), callback_pipeline_finished )

    print '  Running quality control...'
    pl.start_quality_control()

    pl.wait_safe()

    pl.disconnect( pl, SIGNAL('updateProgress'), callback_pipeline_update_progress )

    #sys.stdout.write( '\n' )
    #sys.stdout.flush()

    #sys.stdout.write( '\n' )
    #sys.stdout.flush()

    #pl.disconnect( pl, SIGNAL('finished()'), callback_pipeline_finished )


    """if print_population_plot:
        print '  Printing population plots...'
        filename = os.path.join( path_prefix, population_plot_file )
        batch_utils.create_path( filename )
        print_engine.print_cell_populations_and_penetrance( pl, filename )

    if print_selection_plot:
        print '  Printing selection plots...'
        filename = os.path.join( path_prefix, selection_plot_file )
        batch_utils.create_path( filename )
        print_engine.print_cell_selection_plot( pl, filename )"""

    if num_of_random_splits < 1:
        create_random_splits = False

    if create_random_splits:
        print ' Creating random splits...'
        masks = []
        for i in xrange( num_of_random_splits ):
            print 'creating random split # %d' % i
            objIds = numpy.arange( pdc.objFeatures.shape[0] )
            numpy.random.shuffle( objIds )
            size_of_mask = int( size_of_random_splits * objIds.shape[0] + 0.5 )
            objIds = objIds[ : size_of_mask ]
            mask = numpy.zeros( pdc.objFeatures.shape[0], dtype=bool )
            #mask2 = numpy.zeros( pdc.objFeatures.shape[0], dtype=bool )
            mask[ objIds ] = True
            masks.append( mask )

            pic_filename = os.path.join( path_prefix, random_split_file )
            batch_utils.create_path( pic_filename )
            f = open( pic_filename, 'w' )
            import cPickle
            p = cPickle.Pickler( f )
            d = p.dump( masks )
            f.close()


    for i in xrange( len( clustering_param_set ) ):

        clustering_param_dict = clustering_param_set[ i ]

        clustering_exp_factor = clustering_param_dict[ 'exp_factor' ]
        clustering_param3 = clustering_param_dict[ 'param3' ]
        profile_metric = clustering_param_dict[ 'profile_metric' ]

        template_format_dict = clustering_param_dict.copy()

        keys = clustering_param_dict.keys()
        keys.sort()
        keys.remove( 'param1' )
        values = []
        for k in keys:
            values.append( clustering_param_dict[ k ] )
        values = tuple( values )
        if not values in treatment_similarity_maps:
            treatment_similarity_maps[ values ] = ( template_format_dict, [] )

        """print '  Creating directories...'
        for template in [ clustering_file_template, profile_pdf_file_template, profile_xls_file_template, similarity_map_file_template, combined_map_file_template ]:
            path = os.path.join( path_prefix, template % template_format_dict )
            base = os.path.split( path )[0]
            print '  %s' % base
            if not os.path.exists( base ):
                try:
                    os.makedirs( base )
                except:
                    pass
            if not os.path.isdir( base ):
                raise Exception( 'Not a directory: %s' % base )"""


        if control_filter_mode == 'MEDIAN_xMAD':
            filter_mode = pipeline.analyse.FILTER_MODE_MEDIAN_xMAD
        elif control_filter_mode == 'xMEDIAN':
            filter_mode = pipeline.analyse.FILTER_MODE_xMEDIAN
        elif control_filter_mode == 'MEDIAN_xMAD_AND_LIMIT':
            filter_mode = pipeline.analyse.FILTER_MODE_MEDIAN_xMAD_AND_LIMIT
        elif control_filter_mode == 'xMEDIAN_AND_LIMIT':
            filter_mode = pipeline.analyse.FILTER_MODE_xMEDIAN_AND_LIMIT
        else:
            raise Exception( 'Unknown control filter mode: %s' % control_filter_mode )

        print '  Running pre filtering with control filter mode "%s"...' % control_filter_mode

        # determine the IDs of the features to be used for pre-filtering
        featureNames = pl.clusterConfiguration[ clustering_param_dict[ 'index' ] ][ 1 ]

        pl.connect( pl, SIGNAL('updateProgress'), callback_pipeline_update_progress )

        pl.start_pre_filtering( filter_mode, featureNames )

        pl.wait_safe()

        pl.disconnect( pl, SIGNAL('updateProgress'), callback_pipeline_update_progress )


        control_penetrance = 0.0
        tr_mask = pl.mask_and( pl.get_valid_cell_mask(), pl.get_control_treatment_cell_mask() )
        nonCtrl_tr_mask = pl.mask_and( pl.get_non_control_cell_mask(), pl.get_control_treatment_cell_mask() )
        control_treatment_penetrance = numpy.sum( nonCtrl_tr_mask ) / float( numpy.sum( tr_mask ) )
        penetrance_threshold = 1.00 * control_treatment_penetrance
        print 'penetrance_threshold:', penetrance_threshold

        penetrance_cell_mask = numpy.zeros( pl.controlCellMask.shape, dtype=bool )
        print 'penetrance filtering of treatments..'
        for tr in pdc.treatments:
            tr_mask = pl.mask_and( pl.get_valid_cell_mask(), pl.get_treatment_cell_mask( tr.index ) )
            nonCtrl_tr_mask = pl.mask_and( pl.get_non_control_cell_mask(), pl.get_treatment_cell_mask( tr.index ) )
            penetrance = numpy.sum( nonCtrl_tr_mask ) / float( numpy.sum( tr_mask ) )
            if penetrance > penetrance_threshold:
                print 'treatment %s shows a phenotype!!!' % tr.name
                penetrance_cell_mask[ pl.get_treatment_cell_mask( tr.index ) ] = True
        penetrance_cell_mask = pl.mask_and( penetrance_cell_mask, pl.get_valid_cell_mask() )


        if i < len( clustering_files ) and template_format_dict[ 'method' ] == 'k-means':

            print '  Loading clustering file for param1=%(param1)d, param2=%(param2)d, param3=%(param3)d, param4=%(param4)d, exp_factor=%(exp_factor).1f...' \
                    % template_format_dict

            pl.load_clusters( clustering_files[ i ], clustering_param_dict[ 'exp_factor' ] )

        elif clustering_file and template_format_dict[ 'method' ] == 'k-means':

            print '  Loading clustering file for param1=%(param1)d, param2=%(param2)d, param3=%(param3)d, param4=%(param4)d, exp_factor=%(exp_factor).1f...' \
                    % template_format_dict
            filename = os.path.join( path_prefix, clustering_file )

            pl.load_clusters( filename, clustering_param_dict[ 'exp_factor' ] )

        else:

            if clustering_param_dict[ 'param1' ] == clustering_param1_list[ 0 ] and \
               clustering_param_dict[ 'param2' ] == clustering_param2_list[ 0 ] and \
               clustering_param_dict[ 'param3' ] == clustering_param3_list[ 0 ] and \
               clustering_param_dict[ 'param4' ] == clustering_param4_list[ 0 ] and \
               clustering_param_dict[ 'method' ] == clustering_method_list[ 0 ] and \
               clustering_param_dict[ 'exp_factor' ] == clustering_exp_factor_list[ 0 ] and \
               clustering_param_dict[ 'profile_metric' ] == clustering_profile_metric_list[ 0 ]:

                #if print_population_plot:
                print '  Printing population plots...'
                filename = os.path.join( path_prefix, population_plot_file % template_format_dict )
                batch_utils.create_path( filename )
                print_engine.print_cell_populations_and_penetrance( pl, filename )

                #if print_selection_plot:
                print '  Printing selection plots...'
                filename = os.path.join( path_prefix, selection_plot_file % template_format_dict )
                batch_utils.create_path( filename )
                print_engine.print_cell_selection_plot( pl, filename )


            pl.connect( pl, SIGNAL('updateProgress'), callback_pipeline_update_progress )
            #pl.connect( pl, SIGNAL('finished()'), callback_pipeline_finished )

            print '  Running %(method)s clustering with index=%(index)d, param1=%(param1)d, param2=%(param2)d, param3=%(param3)d, param4=%(param4)d, exp_factor=%(exp_factor).1f...' \
                    % template_format_dict
            #clustering_kwargs = clustering_param_dict.copy()
            #del clustering_kwargs[ 'profile_metric' ]
            #pl.start_clustering( **clustering_kwargs )

            #pl.wait_safe()
            #pl.disconnect( pl, SIGNAL('updateProgress'), callback_pipeline_update_progress )

            oldControlCellMask = pl.controlCellMask
            oldNonControlCellMask = pl.nonControlCellMask
            pl.controlCellMask = numpy.zeros( pl.controlCellMask.shape, dtype=bool )
            pl.nonControlCellMask = penetrance_cell_mask

            clustering_kwargs = clustering_param_dict.copy()
            del clustering_kwargs[ 'profile_metric' ]
            pl.run_clustering( **clustering_kwargs )

            filename = os.path.join( path_prefix, clustering_file_template % template_format_dict )
            batch_utils.create_path( filename )
            clustering_files.append( filename )

            pl.controlCellMask = oldControlCellMask
            pl.nonControlCellMask = oldNonControlCellMask

            if template_format_dict[ 'method' ] not in [ 'ward', 'average', 'single', 'complete' ]:

                print '  Saving clustering file...'

                pl.save_clusters( filename )

                print '  Printing clustering plot...'
                print_engine.print_clustering_plot( pl, filename + '.pdf' )

            if clustering_param_dict[ 'param1' ] == clustering_param1_list[ 0 ] and \
               clustering_param_dict[ 'param2' ] == clustering_param2_list[ 0 ] and \
               clustering_param_dict[ 'param3' ] == clustering_param3_list[ 0 ] and \
               clustering_param_dict[ 'param4' ] == clustering_param4_list[ 0 ] and \
               clustering_param_dict[ 'method' ] == clustering_method_list[ 0 ] and \
               clustering_param_dict[ 'exp_factor' ] == clustering_exp_factor_list[ 0 ] and \
               clustering_param_dict[ 'profile_metric' ] == clustering_profile_metric_list[ 0 ]:

                print '  Saving pipeline state...'
                filename = os.path.join( path_prefix, pipeline_state_file % template_format_dict )
                batch_utils.create_path( filename )
                pl.save_state( filename )


        if not only_run_clustering:


            #random_split_match = numpy.empty( ( len( pdc.treatments ), ) )
            #replicate_split_match = numpy.empty( ( len( pdc.treatments ), ) )


            """random_splits = []
            random_split_labels = []

            for i in xrange( num_of_random_splits ):

                f = open( os.path.join( path_prefix, random_split_file_template % { 'split_index' : i } ), 'r' )
                import cPickle
                up = cPickle.Unpickler( f )
                d = up.load()
                f.close()

                tr_ids = d[ 'tr_ids' ]

                masks = []
                labels = []
                for tr in d[ 'treatments' ]:
                    tr_mask = numpy.logical_and( tr_ids == tr.index, pl.get_non_control_cell_mask() )
                    masks.append( tr_mask )
                    labels.append( tr.name )

                random_splits.append( masks )
                random_split_labels.append( labels )

            random_split_profiles = []

            for i in xrange( num_of_random_splits ):

                masks = random_splits[ i ]
                cluster_profiles = pl.compute_cluster_profiles(
                                            masks,
                                            pl.nonControlNormFeatures,
                                            pl.nonControlClusters,
                                            clustering_exp_factor,
                                            clustering_param3
                )
                random_split_profiles.append( cluster_profiles )

            new_random_split_profiles = []
            multi_cluster_profiles = numpy.empty( ( num_of_random_splits, random_split_profiles[0].shape[0], random_split_profiles[0].shape[1] ) )

            for i in xrange( num_of_random_splits ):

                clusterProfiles = random_split_profiles[ i ]
                labels = random_split_labels[ i ]

                nonEmptyProfileIndices = range( clusterProfiles.shape[0] )
                print 'nonEmptyProfilesIndices:', nonEmptyProfileIndices

                for i in xrange( clusterProfiles.shape[0] ):

                    if numpy.all( clusterProfiles[ i ] == 0 ):
                        nonEmptyProfileIndices.remove( i )

                treatmentLabels = []
                for j in xrange( len( labels ) ):
                    if j in nonEmptyProfileIndices:
                        treatmentLabels.append( labels[ j ] )

                clusterProfiles = clusterProfiles[ nonEmptyProfileIndices ]
                multi_cluster_profiles[ i ] = clusterProfiles
                new_random_split_profiles.append( clusterProfiles )

            for i in xrange( multi_cluster_profiles.shape[1] ):

                print 'multi_cluster_profiles.shape[:,%d,:]:' % i, multi_cluster_profiles[:,i,:].shape

                clusterProfiles = multi_cluster_profiles[:,i,:]

                distanceHeatmap = batch_utils.compute_treatment_distance_map( clusterProfiles, profile_metric )
                similarityHeatmap = batch_utils.compute_treatment_similarity_map( distanceHeatmap, profile_metric )
                #profileHeatmap = batch_utils.compute_treatment_similarity_map( clusterProfiles )

                labels = []
                for j in xrange( num_of_random_splits ):
                    labels.append( 'random split %d' % j )

                print 'distanceHeatmap:', distanceHeatmap
                print 'similarityHeatmap:', similarityHeatmap

                print 'Printing cluster profiles for treatment %s...' % random_split_labels[ 0 ][ i ]

                pdf_filename = os.path.join( path_prefix, random_profile_pdf_file_template % template_format_dict )
                batch_utils.create_path( pdf_filename)
                #sum_minmax_profileHeatmap, l2_chi2_norm_profileHeatmap, treatments = print_engine.print_cluster_profiles_and_heatmap( pl.pdc, clusterProfiles, pdf_filename, True, 0.0 )
                print_engine.print_cluster_profiles_and_heatmap( labels, clusterProfiles, similarityHeatmap, pdf_filename, True, 0.0, similarity_map_clustering_method )

                print 'Writing profile heatmap xls file...'

                template_format_dict[ 'random_split' ] = i
                template_format_dict[ 'tr' ] = random_split_labels[ 0 ][ i ]
                xls_title = 'Profile heatmap for random split %(random_split)d of treatment %(tr)s with method=%(method)s, index=%(index)d, param1=%(param1)d, param2=%(param2)d, param3=%(param3)d, param4=%(param4)d, exp_factor=%(exp_factor).1f' \
                            % template_format_dict
                xls_filename = os.path.join( path_prefix, random_profile_xls_file_template % template_format_dict )
                batch_utils.create_path( xls_filename )
                batch_utils.write_profileHeatmapCSV( xls_title, labels, similarityHeatmap, xls_filename )

                random_split_match[ i ] = numpy.mean( similarityHeatmap[ numpy.identity( similarityHeatmap.shape[0], dtype=bool ) ] )"""


            random_split_dist = []
            random_split_dist_dict = {}
            #random_split_match = numpy.empty( ( len( pdc.treatments ), ) )
       
            if num_of_random_splits > 0:
                pic_filename = os.path.join( path_prefix, random_split_file )
                f = open( pic_filename, 'r' )
                import cPickle
                up = cPickle.Unpickler( f )
                random_splits = up.load()
                f.close()

            """for tr in pdc.treatments:

                print 'processing random splits of treatment %s' % tr.name

                #tr_mask = pdc.objFeatures[ :, pdc.objTreatmentFeatureId ] == tr.index
                tr_mask = pl.get_treatment_cell_mask( tr.index )
                tr_mask = tr_mask[ pl.get_non_control_cell_mask() ]

                masks = []
                labels = []
                for i in xrange( len( random_splits ) ):
                    random_split = random_splits[ i ]
                    random_split = random_split[ pl.get_non_control_cell_mask() ]
                    mask = pl.mask_and( tr_mask, random_split )
                    if numpy.sum( mask ) > 0:
                        masks.append( mask )
                        labels.append( '%d' % i )"""

            groups = get_groups_function( pl )
            control_groups = get_control_groups_function( pl )

            print 'control groups:'
            for group_name, group_mask in control_groups:
                print '  %s' % group_name
            print 'groups:'
            for group_name, group_mask in groups:
                print '  %s' % group_name
            print 'treatments:'
            for tr in pl.pdc.treatments:
                print '  %s' % tr.name
            print 'replicates:'
            for repl in pl.pdc.replicates:
                print '  %s' % repl.name
            print 'wells:'
            for well in pl.pdc.wells:
                print '  %s' % well.name

            random_feature_weights = []

            if num_of_random_splits > 0:

                for group_name, group_mask in groups:

                    print 'processing random splits of group %s' % group_name

                    if numpy.sum( pl.mask_and( group_mask, pl.get_non_control_cell_mask() ) ) < 2:
                        continue
                    # compute penetrance
                    penetrance = numpy.sum( pl.mask_and( group_mask, pl.get_non_control_cell_mask() ) ) / float( numpy.sum( pl.mask_and( group_mask, pl.get_valid_cell_mask() ) ) )
                    if penetrance < penetrance_threshold:
                        continue
                    group_mask = group_mask[ pl.get_non_control_cell_mask() ]

                    masks = []
                    labels = []
                    for i in xrange( len( random_splits ) ):
                        random_split = random_splits[ i ]
                        random_split = random_split[ pl.get_non_control_cell_mask() ]
                        mask = pl.mask_and( group_mask, random_split )
                        if numpy.sum( mask ) > 0:
                            masks.append( mask )
                            labels.append( '%d' % i )

                    if len( masks ) < 2:
                        print 'skipping random splits of group %s' % group_name
                        continue

                    if profile_metric == 'kolmo_smirnov':

                        clusterProfiles = pl.compute_cluster_profiles( masks, pl.nonControlNormFeatures, pl.nonControlClusters, clustering_exp_factor, clustering_param3 )

                        edfs = batch_utils.compute_edfs( pl, masks, pl.nonControlNormFeatures )

                        distanceHeatmap = batch_utils.compute_edf_distances( edfs )
                        random_feature_weights.append( batch_utils.compute_edf_weighting( edfs ) )

                        print 'distanceHeatmap:', distanceHeatmap

                        template_format_dict[ 'group' ] = group_name

                        if random_profile_file_template:

                            print 'Printing feature weights of group %s...' % group_name

                            pdf_filename = os.path.join( path_prefix, random_profile_file_template % template_format_dict ) + '.pdf'
                            batch_utils.create_path( pdf_filename)
                            #sum_minmax_profileHeatmap, l2_chi2_norm_profileHeatmap, treatments = print_engine.print_cluster_profiles_and_heatmap( pl.pdc, clusterProfiles, pdf_filename, True, 0.0 )
                            #print_engine.print_cluster_profiles_and_heatmap( labels, clusterProfiles, similarityHeatmap, pdf_filename, True, 0.0, similarity_map_clustering_method, xlabel='Random split #', ylabel='Random split #' )
                            print_engine.print_cluster_profiles_and_heatmap( labels, clusterProfiles, distanceHeatmap, pdf_filename, False, 0.0, similarity_map_clustering_method, xlabel='Random split #', ylabel='Random split #', map_fontsize=pdf_map_fontsize, map_label_fontsize=pdf_map_label_fontsize, cluster_label_fontsize=pdf_cluster_label_fontsize )

                            pdf_filename = os.path.join( path_prefix, random_profile_file_template % template_format_dict ) + '_weights.pdf'
                            batch_utils.create_path( pdf_filename)
                            print_engine.print_feature_weights( random_feature_weights[ -1 ], pdf_filename, title=group_name )

                            print 'done'

                            pic_filename = os.path.join( path_prefix, random_profile_file_template % template_format_dict ) + '.pic'
                            batch_utils.create_path( pic_filename )
                            f = open( pic_filename, 'w' )
                            import cPickle
                            p = cPickle.Pickler( f )
                            p.dump( { 'labels' : labels, 'masks' : masks, 'clusterProfiles' : clusterProfiles, 'distanceHeatmap' : distanceHeatmap } )
                            f.close()

                    else:

                        clusterProfiles = pl.compute_cluster_profiles( masks, pl.nonControlNormFeatures, pl.nonControlClusters, clustering_exp_factor, clustering_param3 )

                        binSimilarityMatrix = batch_utils.compute_cluster_similarity_matrix( pl.nonControlClusters )
                        distanceHeatmap = batch_utils.compute_treatment_distance_map( clusterProfiles, profile_metric, 0.0, binSimilarityMatrix, normalizationFactor )
                        similarityHeatmap = batch_utils.compute_treatment_similarity_map( distanceHeatmap, profile_metric )

                        print 'distanceHeatmap:', distanceHeatmap
                        print 'similarityHeatmap:', similarityHeatmap

                        template_format_dict[ 'group' ] = group_name

                        if random_profile_file_template:

                            print 'Printing cluster profiles for random splits of group %s...' % group_name

                            pdf_filename = os.path.join( path_prefix, random_profile_file_template % template_format_dict ) + '.pdf'
                            print 'pdf_filename:', pdf_filename
                            batch_utils.create_path( pdf_filename)
                            #sum_minmax_profileHeatmap, l2_chi2_norm_profileHeatmap, treatments = print_engine.print_cluster_profiles_and_heatmap( pl.pdc, clusterProfiles, pdf_filename, True, 0.0 )
                            #print_engine.print_cluster_profiles_and_heatmap( labels, clusterProfiles, similarityHeatmap, pdf_filename, True, 0.0, similarity_map_clustering_method, xlabel='Random split #', ylabel='Random split #' )
                            print_engine.print_cluster_profiles_and_heatmap( labels, clusterProfiles, distanceHeatmap, pdf_filename, True, 0.0, similarity_map_clustering_method, binSimilarityMatrix=binSimilarityMatrix, xlabel='Random split #', ylabel='Random split #', map_fontsize=pdf_map_fontsize, map_label_fontsize=pdf_map_label_fontsize, cluster_label_fontsize=pdf_cluster_label_fontsize )

                            print 'done'

                            print 'Writing profile heatmap xls file...'

                            xls_title = 'Profile distance heatmap for random splits of treatment %(group)s with method=%(method)s, index=%(index)d, param1=%(param1)d, param2=%(param2)d, param3=%(param3)d, param4=%(param4)d, exp_factor=%(exp_factor).1f' \
                                        % template_format_dict
                            xls_filename = os.path.join( path_prefix, random_profile_file_template % template_format_dict ) + '.xls'
                            batch_utils.create_path( xls_filename )
                            batch_utils.write_profileHeatmapCSV( xls_title, labels, distanceHeatmap, xls_filename )

                            pic_filename = os.path.join( path_prefix, random_profile_file_template % template_format_dict ) + '.pic'
                            batch_utils.create_path( pic_filename )
                            f = open( pic_filename, 'w' )
                            import cPickle
                            p = cPickle.Pickler( f )
                            p.dump( { 'labels' : labels, 'masks' : masks, 'clusterProfiles' : clusterProfiles, 'similarityHeatmap' : similarityHeatmap, 'distanceHeatmap' : distanceHeatmap } )
                            f.close()

                    dist = numpy.mean( distanceHeatmap[ numpy.invert( numpy.identity( distanceHeatmap.shape[0], dtype=bool ) ) ] )
                    random_split_dist.append( dist )
                    random_split_dist_dict[ group_name ] = dist

                    #random_split_dist.append( numpy.mean( distanceHeatmap[ numpy.invert( numpy.identity( distanceHeatmap.shape[0], dtype=bool ) ) ] ) )
                    #random_split_match[ tr.index ] = numpy.mean( similarityHeatmap[ numpy.identity( similarityHeatmap.shape[0], dtype=bool ) ] )

            random_split_dist = numpy.array( random_split_dist )

            #replicate_split_match = numpy.empty( ( len( pdc.treatments ), ) )

            replicate_split_dist = []

            for repl in pdc.replicates:
                print 'replicate %d: %s' % ( repl.index, repl.name )

            """for tr in pdc.treatments:

                print 'processing replicate splits of treatment %s' % tr.name

                labels = []
                masks = []
                for repl in pdc.replicates:
                    repl_mask = pl.get_replicate_cell_mask( repl.index )
                    tr_mask = pl.get_treatment_cell_mask( tr.index )
                    mask = pl.mask_and( repl_mask, tr_mask  )
                    mask = mask[ pl.get_non_control_cell_mask() ]
                    if numpy.sum( mask ) > 0:
                        masks.append( mask )
                        #labels.append( repl.name )
                        labels.append( repl.index )
                    #else:
                    #    print 'replicate %s of treatment %s is empty!' % ( repl.name, tr.name )"""

            replicate_feature_weights = []

            replicate_split_dist_dict = {}

            for group_name, group_mask in groups:

                print 'processing replicate splits of group %s' % group_name

                if numpy.sum( pl.mask_and( group_mask, pl.get_non_control_cell_mask() ) ) < 2:
                    continue
                # compute penetrance
                penetrance = numpy.sum( pl.mask_and( group_mask, pl.get_non_control_cell_mask() ) ) / float( numpy.sum( pl.mask_and( group_mask, pl.get_valid_cell_mask() ) ) )
                if penetrance < penetrance_threshold:
                    continue
                group_mask = group_mask[ pl.get_non_control_cell_mask() ]

                masks = []
                labels = []
                for repl in pdc.replicates:
                    repl_mask = pl.get_replicate_cell_mask( repl.index )
                    repl_mask = repl_mask[ pl.get_non_control_cell_mask() ]
                    mask = pl.mask_and( repl_mask, group_mask  )
                    if numpy.sum( mask ) > 0:
                        masks.append( mask )
                        labels.append( repl.name )
                        #labels.append( repl.index )

                print len(pdc.replicates)
                print len(masks)

                if len( masks ) < 2:
                    print 'skipping replicate splits of group %s' % group_name
                    continue

                if profile_metric == 'kolmo_smirnov':

                    clusterProfiles = pl.compute_cluster_profiles( masks, pl.nonControlNormFeatures, pl.nonControlClusters, clustering_exp_factor, clustering_param3 )

                    edfs = batch_utils.compute_edfs( pl, masks, pl.nonControlNormFeatures )

                    distanceHeatmap = batch_utils.compute_edf_distances( edfs )
                    replicate_feature_weights.append( batch_utils.compute_edf_weighting( edfs ) )

                    #feature_weights = batch_utils.compute_edf_weighting( edfs )

                    print 'distanceHeatmap:', distanceHeatmap

                    template_format_dict[ 'group' ] = group_name

                    if replicate_profile_file_template:

                        print 'Printing feature weights of group %s...' % group_name

                        pdf_filename = os.path.join( path_prefix, replicate_profile_file_template % template_format_dict ) + '.pdf'
                        batch_utils.create_path( pdf_filename)
                        #sum_minmax_profileHeatmap, l2_chi2_norm_profileHeatmap, treatments = print_engine.print_cluster_profiles_and_heatmap( pl.pdc, clusterProfiles, pdf_filename, True, 0.0 )
                        #print_engine.print_cluster_profiles_and_heatmap( labels, clusterProfiles, similarityHeatmap, pdf_filename, True, 0.0, similarity_map_clustering_method, xlabel='Random split #', ylabel='Random split #' )
                        print_engine.print_cluster_profiles_and_heatmap( labels, clusterProfiles, distanceHeatmap, pdf_filename, False, 0.0, similarity_map_clustering_method, xlabel='Random split #', ylabel='Random split #', map_fontsize=pdf_map_fontsize, map_label_fontsize=pdf_map_label_fontsize, cluster_label_fontsize=pdf_cluster_label_fontsize )

                        pdf_filename = os.path.join( path_prefix, replicate_profile_file_template % template_format_dict ) + '_weights.pdf'
                        batch_utils.create_path( pdf_filename)
                        print_engine.print_feature_weights( replicate_feature_weights[ -1 ], pdf_filename, title=group_name )

                        print 'done'

                        pic_filename = os.path.join( path_prefix, replicate_profile_file_template % template_format_dict ) + '.pic'
                        batch_utils.create_path( pic_filename )
                        f = open( pic_filename, 'w' )
                        import cPickle
                        p = cPickle.Pickler( f )
                        p.dump( { 'labels' : labels, 'masks' : masks, 'clusterProfiles' : clusterProfiles, 'distanceHeatmap' : distanceHeatmap } )
                        f.close()

                else:

                    clusterProfiles = pl.compute_cluster_profiles( masks, pl.nonControlNormFeatures, pl.nonControlClusters, clustering_exp_factor, clustering_param3 )

                    binSimilarityMatrix = batch_utils.compute_cluster_similarity_matrix( pl.nonControlClusters )
                    distanceHeatmap = batch_utils.compute_treatment_distance_map( clusterProfiles, profile_metric, 0.0, binSimilarityMatrix, normalizationFactor )
                    similarityHeatmap = batch_utils.compute_treatment_similarity_map( distanceHeatmap, profile_metric )

                    print 'distanceHeatmap:', distanceHeatmap
                    print 'similarityHeatmap:', similarityHeatmap

                    template_format_dict[ 'group' ] = group_name

                    if replicate_profile_file_template:

                        print 'Printing cluster profiles for replicate splits of treatment %s...' % tr.name

                        heatmap_kwargs = { 'lower' : True, 'diagonal' : True }
                        pdf_filename = os.path.join( path_prefix, replicate_profile_file_template % template_format_dict ) + '.pdf'
                        batch_utils.create_path( pdf_filename)
                        #sum_minmax_profileHeatmap, l2_chi2_norm_profileHeatmap, treatments = print_engine.print_cluster_profiles_and_heatmap( pl.pdc, clusterProfiles, pdf_filename, True, 0.0 )
                        #print_engine.print_cluster_profiles_and_heatmap( labels, clusterProfiles, similarityHeatmap, pdf_filename, True, 0.0, similarity_map_clustering_method, xlabel='Random split #', ylabel='Random split #' )
                        print_engine.print_cluster_profiles_and_heatmap( labels, clusterProfiles, distanceHeatmap, pdf_filename, True, 0.0, similarity_map_clustering_method, binSimilarityMatrix=binSimilarityMatrix, xlabel='Replicate split #', ylabel='Replicate split #', heatmap_kwargs=heatmap_kwargs, map_fontsize=pdf_map_fontsize, map_label_fontsize=pdf_map_label_fontsize, cluster_label_fontsize=pdf_cluster_label_fontsize )

                        print 'Writing profile heatmap xls file...'

                        xls_title = 'Profile distance heatmap for replicates of treatment %(group)s with method=%(method)s, index=%(index)d, param1=%(param1)d, param2=%(param2)d, param3=%(param3)d, param4=%(param4)d, exp_factor=%(exp_factor).1f' \
                                    % template_format_dict
                        xls_filename = os.path.join( path_prefix, replicate_profile_file_template % template_format_dict ) + '.xls'
                        batch_utils.create_path( xls_filename )
                        batch_utils.write_profileHeatmapCSV( xls_title, labels, distanceHeatmap, xls_filename )

                        pic_filename = os.path.join( path_prefix, replicate_profile_file_template % template_format_dict ) + '.pic'
                        batch_utils.create_path( pic_filename )
                        f = open( pic_filename, 'w' )
                        import cPickle
                        p = cPickle.Pickler( f )
                        p.dump( { 'labels' : labels, 'masks' : masks, 'clusterProfiles' : clusterProfiles, 'similarityHeatmap' : similarityHeatmap, 'distanceHeatmap' : distanceHeatmap } )
                        f.close()

                dist = numpy.mean( distanceHeatmap[ numpy.invert( numpy.identity( distanceHeatmap.shape[0], dtype=bool ) ) ] )
                replicate_split_dist.append( dist )
                replicate_split_dist_dict[ group_name ] = dist

                #replicate_split_match[ tr.index ] = numpy.mean( similarityHeatmap[ numpy.identity( similarityHeatmap.shape[0], dtype=bool ) ] )

            replicate_split_dist = numpy.array( replicate_split_dist )

            control_split_dist = None

            print 'processing comparison of control groups'

            masks = []
            labels = []

            for group_name, group_mask in control_groups:

                if numpy.sum( pl.mask_and( group_mask, pl.get_non_control_cell_mask() ) ) < 2:
                    continue
                # compute penetrance
                penetrance = numpy.sum( pl.mask_and( group_mask, pl.get_non_control_cell_mask() ) ) / float( numpy.sum( pl.mask_and( group_mask, pl.get_valid_cell_mask() ) ) )
                if penetrance < penetrance_threshold:
                    continue
                group_mask = group_mask[ pl.get_non_control_cell_mask() ]

                if numpy.sum( group_mask ) > 0:
                    masks.append( group_mask )
                    labels.append( group_name )
                    #labels.append( repl.index )

            print len(masks)

            if len( masks ) > 1:

                if profile_metric == 'kolmo_smirnov':

                    """clusterProfiles = pl.compute_cluster_profiles( masks, pl.nonControlNormFeatures, pl.nonControlClusters, clustering_exp_factor, clustering_param3 )

                    edfs = batch_utils.compute_edfs( pl, masks, pl.nonControlNormFeatures )

                    distanceHeatmap = batch_utils.compute_edf_distances( edfs )
                    replicate_feature_weights.append( batch_utils.compute_edf_weighting( edfs ) )

                    #feature_weights = batch_utils.compute_edf_weighting( edfs )

                    print 'distanceHeatmap:', distanceHeatmap

                    template_format_dict[ 'group' ] = group_name

                    if replicate_profile_file_template:

                        print 'Printing feature weights of group %s...' % group_name

                        pdf_filename = os.path.join( path_prefix, replicate_profile_file_template % template_format_dict ) + '.pdf'
                        batch_utils.create_path( pdf_filename)
                        #sum_minmax_profileHeatmap, l2_chi2_norm_profileHeatmap, treatments = print_engine.print_cluster_profiles_and_heatmap( pl.pdc, clusterProfiles, pdf_filename, True, 0.0 )
                        #print_engine.print_cluster_profiles_and_heatmap( labels, clusterProfiles, similarityHeatmap, pdf_filename, True, 0.0, similarity_map_clustering_method, xlabel='Random split #', ylabel='Random split #' )
                        print_engine.print_cluster_profiles_and_heatmap( labels, clusterProfiles, distanceHeatmap, pdf_filename, False, 0.0, similarity_map_clustering_method, xlabel='Random split #', ylabel='Random split #', map_fontsize=pdf_map_fontsize, map_label_fontsize=pdf_map_label_fontsize, cluster_label_fontsize=pdf_cluster_label_fontsize )

                        pdf_filename = os.path.join( path_prefix, replicate_profile_file_template % template_format_dict ) + '_weights.pdf'
                        batch_utils.create_path( pdf_filename)
                        print_engine.print_feature_weights( replicate_feature_weights[ -1 ], pdf_filename, title=group_name )

                        print 'done'

                        pic_filename = os.path.join( path_prefix, replicate_profile_file_template % template_format_dict ) + '.pic'
                        batch_utils.create_path( pic_filename )
                        f = open( pic_filename, 'w' )
                        import cPickle
                        p = cPickle.Pickler( f )
                        p.dump( { 'labels' : labels, 'masks' : masks, 'clusterProfiles' : clusterProfiles, 'distanceHeatmap' : distanceHeatmap } )
                        f.close()"""

                    raise Exception( 'not implemented yet' )

                else:

                    clusterProfiles = pl.compute_cluster_profiles( masks, pl.nonControlNormFeatures, pl.nonControlClusters, clustering_exp_factor, clustering_param3 )

                    binSimilarityMatrix = batch_utils.compute_cluster_similarity_matrix( pl.nonControlClusters )
                    distanceHeatmap = batch_utils.compute_treatment_distance_map( clusterProfiles, profile_metric, 0.0, binSimilarityMatrix, normalizationFactor )
                    similarityHeatmap = batch_utils.compute_treatment_similarity_map( distanceHeatmap, profile_metric )

                    print 'distanceHeatmap:', distanceHeatmap
                    print 'similarityHeatmap:', similarityHeatmap

                    if control_profile_file_template:

                        print 'Printing cluster profiles of control comparison'

                        heatmap_kwargs = { 'lower' : True, 'diagonal' : True }
                        pdf_filename = os.path.join( path_prefix, control_profile_file_template % template_format_dict ) + '.pdf'
                        batch_utils.create_path( pdf_filename)
                        #sum_minmax_profileHeatmap, l2_chi2_norm_profileHeatmap, treatments = print_engine.print_cluster_profiles_and_heatmap( pl.pdc, clusterProfiles, pdf_filename, True, 0.0 )
                        #print_engine.print_cluster_profiles_and_heatmap( labels, clusterProfiles, similarityHeatmap, pdf_filename, True, 0.0, similarity_map_clustering_method, xlabel='Random split #', ylabel='Random split #' )
                        print_engine.print_cluster_profiles_and_heatmap( labels, clusterProfiles, distanceHeatmap, pdf_filename, True, 0.0, similarity_map_clustering_method, binSimilarityMatrix=binSimilarityMatrix, xlabel='Control group', ylabel='Control group', heatmap_kwargs=heatmap_kwargs, map_fontsize=pdf_map_fontsize, map_label_fontsize=pdf_map_label_fontsize, cluster_label_fontsize=pdf_cluster_label_fontsize )

                        print 'Writing profile heatmap xls file...'

                        xls_title = 'Profile distance heatmap for control groups with method=%(method)s, index=%(index)d, param1=%(param1)d, param2=%(param2)d, param3=%(param3)d, param4=%(param4)d, exp_factor=%(exp_factor).1f' \
                                    % template_format_dict
                        xls_filename = os.path.join( path_prefix, control_profile_file_template % template_format_dict ) + '.xls'
                        batch_utils.create_path( xls_filename )
                        batch_utils.write_profileHeatmapCSV( xls_title, labels, distanceHeatmap, xls_filename )

                        pic_filename = os.path.join( path_prefix, control_profile_file_template % template_format_dict ) + '.pic'
                        batch_utils.create_path( pic_filename )
                        f = open( pic_filename, 'w' )
                        import cPickle
                        p = cPickle.Pickler( f )
                        p.dump( { 'labels' : labels, 'masks' : masks, 'clusterProfiles' : clusterProfiles, 'similarityHeatmap' : similarityHeatmap, 'distanceHeatmap' : distanceHeatmap } )
                        f.close()

                control_split_dist = numpy.mean( distanceHeatmap[ numpy.invert( numpy.identity( distanceHeatmap.shape[0], dtype=bool ) ) ] )

                #replicate_split_match[ tr.index ] = numpy.mean( similarityHeatmap[ numpy.identity( similarityHeatmap.shape[0], dtype=bool ) ] )

            else:
                print 'skipping comparison of control groups'

            print 'replicate split groups:', replicate_split_dist_dict.keys()

            feature_weights = numpy.array( random_feature_weights )

            print 'Computing similarity map...'
            print 'penetrance_threshold:', penetrance_threshold

            for group_name,group_mask in groups:
                print 'group:', group_name

            labels = []
            masks = []
            random_split_dist = []
            replicate_split_dist = []
            cluster_mask = []
            for group_name, group_mask in groups:
                #if numpy.sum( group_mask ) < 2:
                #    continue
                if numpy.sum( pl.mask_and( group_mask, pl.get_non_control_cell_mask() ) ) < 2:
                    print 'skipping group:', group_name
                    continue
                # compute penetrance
                penetrance = numpy.sum( pl.mask_and( group_mask, pl.get_non_control_cell_mask() ) ) / float( numpy.sum( pl.mask_and( group_mask, pl.get_valid_cell_mask() ) ) )
                print 'penetrance:', penetrance
                if group_name in replicate_split_dist_dict:
                    print 'replicate distance:', replicate_split_dist_dict[ group_name ]
                print 'valid cells in group:', numpy.sum( pl.mask_and( group_mask, pl.get_valid_cell_mask() ) )
                print 'non-control cells in group:', numpy.sum( pl.mask_and( group_mask, pl.get_non_control_cell_mask() ) )
                print 'control cells in group:', numpy.sum( pl.mask_and( group_mask, pl.get_control_cell_mask() ) )
                if penetrance < penetrance_threshold:
                    print 'skipping group (low penetrance):', group_name
                    continue
                else:
                    print 'using group:', group_name
                if replicate_distance_threshold >= 0.0 and replicate_split_dist_dict[ group_name ] > replicate_distance_threshold:
                    print 'skipping group from clustering (high replicate distance):', group_name
                    cluster_mask.append( False )
                else:
                    cluster_mask.append( True )
                    print 'using group for clustering:', group_name
                masks.append( group_mask[ pl.get_non_control_cell_mask() ] )
                labels.append( group_name )
                if group_name in random_split_dist_dict:
                    random_split_dist.append( random_split_dist_dict[ group_name ] )
                if group_name in replicate_split_dist_dict:
                    replicate_split_dist.append( replicate_split_dist_dict[ group_name ] )
            if len( random_split_dist ) > 0:
                random_split_dist = numpy.array( random_split_dist )
            else:
                random_split_dist = None
            if len( replicate_split_dist ) > 0:
                replicate_split_dist = numpy.array( replicate_split_dist )
            else:
                replicate_split_dist = None
            cluster_mask = numpy.array( cluster_mask )
            print 'using %d groups for clustering' % ( numpy.sum( cluster_mask ) )

            print 'masks:', len( masks )

            """label_dict = {
                'CA_a[68]' : 'COPA_a',
                'CA_b[21]' : 'COPA_b',
                'CB1_a[15]' : 'COPB1_a',
                'CB1_b[22]' : 'COPB1_b',
                'CB1_c[41]' : 'COPB1_c',
                'CB1_c[55]' : 'COPB1_d',
                'CB2_a[16]' : 'COPB2_a',
                'CB2_b[29]' : 'COPB2_b',
                'CB2_c[42]' : 'COPB2_c',
                'CE_a[17]' : 'COPE_a',
                'CG_a[77]' : 'COPG_a',
                'CG_b[31]' : 'COPG_b',
                'CG_c[44]' : 'COPG_c',
                'CG2_a[78]' : 'COPG2_a',
                'CG2_b[32]' : 'COPG2_b',
                'SAR1A[82]' : 'SAR1A',
                'SAR1B[63]' : 'SAR1B',
                'S23A_a[66]' : 'SEC23A_a',
                'S23A_b[51]' : 'SEC23A_b',
                'S23B_a[39]' : 'SEC23B_a',
                'S23B_b[52]' : 'SEC23B_b',
                'S24A[40]' : 'SEC24A',
                'S13[56]' : 'SEC13_a',
                'S13[81]' : 'SEC13_b',
                'S31A[79]' : 'SEC31A',
                'S31B[80]' : 'SEC31B',
                'S16_a[70]' : 'SEC16_a',
                'S12_a[65]' : 'SEC12_a',
                'neg2[38]' : 'neg2_a',
                'neg2[50]' : 'neg2_b',
                'neg2[62]' : 'neg2_c',
                'sc_a[18]' : 'scr_a',
                'sc_a[19]' : 'scr_b',
                'sc_a[20]' : 'scr_c',
                'sc_a[14]' : 'scr_d',
                '54[67]' : 'RHQU',
                'CKAP5[64]' : 'CKAP5',
                'wt_1(1)[55]' : 'wt_1',
                'wt_1(1)[53]' : 'wt_2',
                'wt_1(1)[54]' : 'wt_3',
                'wt_1(1)[52]' : 'wt_4',
                'wt_0(1)[55]' : 'wt_5',
                'wt_0(1)[53]' : 'wt_6',
                'wt_0(1)[52]' : 'wt_7',
                'wt_0(1)[54]' : 'wt_8',
                'noc_1(1)[43]' : 'noc_1',
                'noc_1(1)[40]' : 'noc_2',
                'noc_1(1)[42]' : 'noc_3',
                'noc_1(1)[41]' : 'noc_4',
                'noc_0(1)[43]' : 'noc_5',
                'noc_0(1)[40]' : 'noc_6',
                'noc_0(1)[42]' : 'noc_7',
                'noc_0(1)[41]' : 'noc_8',
                'bfa_1(1)[28]' : 'bfa_1',
                'bfa_1(1)[29]' : 'bfa_2',
                'bfa_1(1)[30]' : 'bfa_3',
                'bfa_1(1)[31]' : 'bfa_4',
                'bfa_0(1)[28]' : 'bfa_5',
                'bfa_0(1)[29]' : 'bfa_6',
                'bfa_0(1)[30]' : 'bfa_7',
                'bfa_0(1)[31]' : 'bfa_8'
            }
            label_dict2 = {}
            for key,value in label_dict.iteritems():
                label_dict2[ key + '_0' ] = value + '_0'
                label_dict2[ key + '_1' ] = value + '_1'
            label_dict = label_dict2
            remove_labels = []
            #remove_labels.extend( [ 'sc_a[20]', 'sc_a[19]', 'sc_a[18]', 'sc_a[14]', 'neg2[38]', 'neg2[50]', 'neg2[62]', '54' ] )
            #remove_labels.append( 'CKAP5' )
            keep_labels = [ 'SEC23B_b', 'COPG_a', 'neg2_b', 'neg2_c', 'scr_d', 'COPG2_b', 'COPB1_c', 'SEC23A_a',
                            'scr_b', 'SEC24A', 'SEC31B', 'SEC16_a', 'scr_c', 'SEC23B_a', 'SEC12_a', 'RHQU',
                            'neg2_a', 'SAR1A', 'SAR1B', 'SEC23A_b', 'scr_a', 'COPE_a', 'COPG2_a', 'COPA_b',
                            'COPB1_a', 'SEC13_b', 'COPB1_b', 'COPB2_a', 'COPB2_b', 'COPB2_c', 'COPG_c',
                            'COPA_a', 'COPG_b', 'SEC13_a', 'SEC31A' ]
            keep_labels2 = []
            for k in keep_labels:
                keep_labels2.append( k + '_0' )
                keep_labels2.append( k + '_1' )
            keep_labels = keep_labels2
            random_split_dist = list( random_split_dist )
            for i in xrange( len( labels )-1, -1, -1 ):
                if labels[ i ] in label_dict:
                    print 'exchanging %s with %s' % ( labels[i], label_dict[ labels[ i ] ] )
                    labels[ i ] = label_dict[ labels[ i ] ]
                if labels[ i ] not in keep_labels:
                    print 'removing %s' % labels[ i ]
                    del labels[ i ]
                    del masks[ i ]       
                    if len( random_split_dist ) > i:
                        del random_split_dist[ i ]
                elif labels[ i ] in remove_labels:
                    print 'removing %s' % labels[ i ]
                    del labels[ i ]
                    del masks[ i ]       
                    if len( random_split_dist ) > i:
                        del random_split_dist[ i ]"""

            random_split_dist = numpy.array( random_split_dist )
            print 'masks:', len( masks )

            def compute_map_quality(map):
                diagonal_mask = numpy.identity( map.shape[0], dtype=bool )
                non_diagonal_mask = numpy.invert( diagonal_mask )
                diagonal_mean = numpy.mean( map[ diagonal_mask ] )
                non_diagonal_mean = numpy.mean( map[ non_diagonal_mask ] )
                quality = non_diagonal_mean - diagonal_mean
                return quality

            if profile_metric == 'kolmo_smirnov':

                clusterProfiles = pl.compute_cluster_profiles( masks, pl.nonControlNormFeatures, pl.nonControlClusters, clustering_exp_factor, clustering_param3 )

                edfs = batch_utils.compute_edfs( pl, masks, pl.nonControlNormFeatures )

                distanceHeatmap = batch_utils.compute_edf_distances( edfs, feature_weights )

                print 'distanceHeatmap:', distanceHeatmap

                heatmap = distanceHeatmap.copy()
                diagonal_mask = numpy.identity( heatmap.shape[0], dtype=bool )
                if random_split_dist.shape[0] == heatmap.shape[0]:
                    heatmap[ diagonal_mask ] = random_split_dist
                else:
                    heatmap[ diagonal_mask ] = 0.0

                map_quality = compute_map_quality( heatmap )

                template_format_dict[ 'group' ] = group_name

                print 'Printing feature weights of group %s...' % group_name

                pdf_filename = os.path.join( path_prefix, profile_file_template % template_format_dict ) + '.pdf'
                batch_utils.create_path( pdf_filename)
                #sum_minmax_profileHeatmap, l2_chi2_norm_profileHeatmap, treatments = print_engine.print_cluster_profiles_and_heatmap( pl.pdc, clusterProfiles, pdf_filename, True, 0.0 )
                #print_engine.print_cluster_profiles_and_heatmap( labels, clusterProfiles, similarityHeatmap, pdf_filename, True, 0.0, similarity_map_clustering_method, xlabel='Random split #', ylabel='Random split #' )
                print_engine.print_cluster_profiles_and_heatmap( labels, feature_weights, distanceHeatmap, pdf_filename, False, 0.0, similarity_map_clustering_method, xlabel='Random split #', ylabel='Random split #', map_fontsize=pdf_map_fontsize, map_label_fontsize=pdf_map_label_fontsize, cluster_label_fontsize=pdf_cluster_label_fontsize, cluster_mask=cluster_mask )

                print 'done'

                pic_filename = os.path.join( path_prefix, profile_file_template % template_format_dict ) + '.pic'
                batch_utils.create_path( pic_filename )
                f = open( pic_filename, 'w' )
                import cPickle
                p = cPickle.Pickler( f )
                p.dump( { 'labels' : labels, 'masks' : masks, 'map_quality' : map_quality, 'clusterProfiles' : clusterProfiles, 'distanceHeatmap' : distanceHeatmap, 'template_format_dict' : template_format_dict } )
                f.close()

            else:

                clusterProfiles = pl.compute_cluster_profiles( masks, pl.nonControlNormFeatures, pl.nonControlClusters, clustering_exp_factor, clustering_param3 )

                print 'clusterProfiles.shape:', clusterProfiles.shape

                binSimilarityMatrix = batch_utils.compute_cluster_similarity_matrix( pl.nonControlClusters )
                distanceHeatmap = batch_utils.compute_treatment_distance_map( clusterProfiles, profile_metric, 0.0, binSimilarityMatrix, normalizationFactor )
                similarityHeatmap = batch_utils.compute_treatment_similarity_map( distanceHeatmap, profile_metric )

                if random_split_dist != None and random_split_dist.ndim == 0:
                    random_split_dist = None
                if replicate_split_dist != None and replicate_split_dist.ndim == 0:
                    replicate_split_dist = None

                heatmap = distanceHeatmap.copy()
                diagonal_mask = numpy.identity( heatmap.shape[0], dtype=bool )  
                if random_split_dist != None and random_split_dist.shape[0] == heatmap.shape[0]:
                    heatmap[ diagonal_mask ] = random_split_dist

                map_quality = compute_map_quality( heatmap )

                template_format_dict[ 'group' ] = group_name

                print 'Printing cluster profiles...'

                print 'heatmap:', heatmap.shape

                heatmap_kwargs = { 'lower' : True, 'diagonal' : True }
                pdf_filename = os.path.join( path_prefix, profile_file_template % template_format_dict ) + '.pdf'
                print 'pdf_filename:', pdf_filename
                batch_utils.create_path( pdf_filename)
                #sum_minmax_profileHeatmap, l2_chi2_norm_profileHeatmap, treatments = print_engine.print_cluster_profiles_and_heatmap( pl.pdc, clusterProfiles, pdf_filename, True, 0.0 )
                #print_engine.print_cluster_profiles_and_heatmap( labels, clusterProfiles, similarityHeatmap, pdf_filename, True, 0.0, similarity_map_clustering_method, xlabel='Treatment
                print_engine.print_cluster_profiles_and_heatmap( labels, clusterProfiles, heatmap, pdf_filename, True, 0.0, similarity_map_clustering_method, binSimilarityMatrix=binSimilarityMatrix, xlabel='Treatment', ylabel='Treatment', heatmap_kwargs=heatmap_kwargs, random_split_dist=random_split_dist, replicate_split_dist=replicate_split_dist, replicate_distance_threshold=replicate_distance_threshold, cluster_mask=cluster_mask, map_fontsize=pdf_map_fontsize, map_label_fontsize=pdf_map_label_fontsize, cluster_label_fontsize=pdf_cluster_label_fontsize )

                """map = similarityHeatmap.copy()
                map[ numpy.identity( map.shape[0], dtype=bool ) ] = random_split_match

                pdf_filename = os.path.join( path_prefix, profile_pdf_file_template % template_format_dict )
                batch_utils.create_path( pdf_filename)
                pdfDocument = print_engine.PdfDocument( pdf_filename )
                pdfDocument.next_page( 2, 1 )
                print_engine.draw_cluster_profiles( pdfDocument, labels, clusterProfiles )
                pdfDocument.next_page( 1, 1 )
                print_engine.draw_modified_treatment_similarity_map( pdfDocument.next_plot(), map, ( 0, 0 ), labels )

                print 'Clustering treatment similarity map...'
                cdm = 1.0 - similarityHeatmap
                cdm[ numpy.identity( cdm.shape[0], dtype=bool ) ] = 0.0
                cdm = hc.squareform( cdm )
                Z = hc.linkage( cdm, similarity_map_clustering_method )
                #print_engine.print_dendrogram( os.path.splitext( pdf_filename )[ 0 ] + '_dendrogram.pdf', Z, labels )

                pdfDocument.next_page()
                print_engine.draw_dendrogram( pdfDocument.next_plot(), Z, labels )
                pdfDocument.close()"""

                xls_title = 'Profile heatmap with method=%(method)s, index=%(index)d, param1=%(param1)d, param2=%(param2)d, param3=%(param3)d, param4=%(param4)d, exp_factor=%(exp_factor).1f' \
                            % template_format_dict
                xls_filename = os.path.join( path_prefix, profile_file_template % template_format_dict ) + '.xls'
                batch_utils.create_path( xls_filename )
                batch_utils.write_profileHeatmapCSV( xls_title, labels, heatmap, xls_filename )

                pic_filename = os.path.join( path_prefix, profile_file_template % template_format_dict ) + '.pic'
                batch_utils.create_path( pic_filename )
                f = open( pic_filename, 'w' )
                import cPickle
                p = cPickle.Pickler( f )
                p.dump( { 'labels' : labels, 'masks' : masks, 'map_quality' : map_quality, 'clusterProfiles' : clusterProfiles, 'similarityHeatmap' : similarityHeatmap, 'distanceHeatmap' : distanceHeatmap, 'template_format_dict' : template_format_dict, 'heatmap' : heatmap } )
                f.close()

                #similarity_map_files.append( filename )

                #treatment_similarity_maps[ values ][ 1 ].append( floatTreatmentSimilarityMap )

if combine_maps:

    print 'Computing combined similarity maps...'

    """print 'Running pipeline for project_file %s' % ( project_file )

    headlessClusterConfiguration = headless_cluster_configuration.HeadlessClusterConfiguration()

    utils.load_module_configuration( project_file )


    modules = utils.list_modules()
    for module in modules:

        if not utils.all_parameters_set( module ):
            print 'Not all required parameters for module %s have been set' % module
            sys.exit( 1 )

        elif not utils.all_requirements_met( module ):
            print 'Not all requirements for module %s have been fulfilled' % module
            sys.exit( 1 )


    pdc = importer.Importer().get_pdc()
    clusterConfiguration = headlessClusterConfiguration.clusterConfiguration


    pl = pipeline.Pipeline( pdc, clusterConfiguration )"""



    """for values,t in treatment_similarity_maps[ project_name ].iteritems():

        template_format_dict,map_list = t
        summed_map = numpy.zeros( ( num_of_treatments/2, num_of_treatments/2 ), dtype=float )
        for map in map_list:
            summed_map += map

        print summed_map

        print 'Printing combined similarity map...'

        filename = os.path.join( path_prefix, combined_map_file_template % template_format_dict )
        batch_utils.create_path( filename )
        print_engine.print_treatment_similarity_map( summed_map, labels, filename )"""

    summed_map = None
    labels = None
    for sim_map_file in similarity_map_files:

        f = open(os.path.splitext( sim_map_file )[ 0 ] + '.pic', 'r' )
        import cPickle
        u = cPickle.Unpickler( f )
        treatmentSimilarityMap,l = u.load()
        f.close()

        if labels == None:
            labels = l

        if summed_map == None:
            summed_map = numpy.zeros( treatmentSimilarityMap.shape, dtype=float )

        summed_map += treatmentSimilarityMap

    summed_map = summed_map / len( similarity_map_files )

    print summed_map

    #labels = labels[ ::2 ]

    print 'Printing combined similarity map...'

    template_format_dict = clustering_param_set[ 0 ].copy()

    title = 'Combined similarity map of treatments'

    filename = os.path.join( path_prefix, combined_map_file_template % template_format_dict )
    batch_utils.create_path( filename )
    print_engine.print_treatment_similarity_map( summed_map, labels, filename, title )

    print 'Clustering treatment similarity map...'
    cdm = 1.0 - summed_map
    cdm[ numpy.identity( cdm.shape[0], dtype=bool ) ] = 0.0
    cdm = hc.squareform( cdm )
    Z = hc.linkage( cdm, similarity_map_clustering_method )
    print_engine.print_dendrogram( os.path.splitext( filename )[ 0 ] + '_dendrogram.pdf', Z, labels )

    #if project_name.startswith( 'random' ):
    #    random_combined_map_files.append( filename )
    #else:
    #    replicate_combined_map_files.append( filename )

    print 'Writing similarity map xls file...'

    xls_title = 'Similarity map with method=%(method)s, index=%(index)d, param1=%(param1)d, param2=%(param2)d, param3=%(param3)d, param4=%(param4)d, exp_factor=%(exp_factor).1f' \
                % template_format_dict
    batch_utils.write_similarityMapCSV( xls_title, labels, summed_map, filename+'.xls' )

    f = open( os.path.splitext( filename )[ 0 ] + '.pic', 'w' )
    import cPickle
    p = cPickle.Pickler( f )
    p.dump( ( summed_map, labels ) )
    f.close()



def compute_map_quality(map):
    diagonal_mask = numpy.identity( map.shape[0], dtype=bool )
    non_diagonal_mask = numpy.invert( diagonal_mask )
    diagonal_mean = numpy.mean( map[ diagonal_mask ] )
    non_diagonal_mean = numpy.mean( map[ non_diagonal_mask ] )
    quality = non_diagonal_mean - diagonal_mean
    return quality


#if profile_metric == 'summed_minmax':
#    invert_profile_heatmap = True

if do_map_quality:

    print 'Computing map quality...'

    template_format_dict = clustering_param_set[ 0 ].copy()

    map_dict = {}

    for profile_file in profile_files:
        pic_filename = profile_file
        f = open( pic_filename, 'r' )
        import cPickle
        up = cPickle.Unpickler( f )
        d = up.load()
        param1 = d[ 'template_format_dict' ][ 'param1' ]
        map_quality = d[ 'map_quality' ]
        map_dict[ param1 ] = map_quality
        f.close()

    keys = map_dict.keys()
    keys.sort()

    map_qualities = []
    for key in keys:
        map_qualities.append( map_dict[ key ] )

    filename = os.path.join( path_prefix, map_quality_file_template % template_format_dict )
    batch_utils.create_path( filename )
    print_engine.print_map_quality_plot(
                                    numpy.array( map_qualities ),
                                    keys,
                                    filename,
                                    'Heatmap qualities',
                                    '# of clusters',
                                    'map quality'
    )

    print 'done'



if analyse_maps:

    print 'Analysing combined similarity maps...'

    random_combined_maps = []
    replicate_combined_maps = []

    random_diagonals = None
    replicate_diagonals = None

    labels = None
    for map_file in random_combined_map_files:

        f = open( os.path.splitext( map_file )[ 0 ] + '.pic', 'r' )
        import cPickle
        u = cPickle.Unpickler( f )
        map,l = u.load()
        f.close()

        if labels == None:
            labels = l

        if random_diagonals == None:
            random_diagonals = numpy.empty( ( len( random_combined_map_files ), map.shape[0] ) )

        random_diagonals[ len( random_combined_maps ) ] = map[ numpy.identity( map.shape[0], dtype=bool ) ]
        random_combined_maps.append( map )

    labels = None
    for map_file in replicate_combined_map_files:

        f = open( os.path.splitext( map_file )[ 0 ] + '.pic', 'r' )
        import cPickle
        u = cPickle.Unpickler( f )
        map,l = u.load()
        f.close()

        if labels == None:
            labels = l

        if replicate_diagonals == None:
            replicate_diagonals = numpy.empty( ( len( replicate_combined_map_files ), map.shape[0] ) )

        replicate_diagonals[ len( replicate_combined_maps ) ] = map[ numpy.identity( map.shape[0], dtype=bool ) ]
        replicate_combined_maps.append( map )

    random_mean = numpy.mean( random_diagonals, axis=0 )
    random_std = numpy.std( random_diagonals, axis=0 )

    replicate_mean = numpy.mean( replicate_diagonals, axis=0 )
    replicate_std = numpy.std( replicate_diagonals, axis=0 )

    template_format_dict = clustering_param_set[ 0 ].copy()

    title = 'Analysation of combined similarity maps'

    filename = os.path.join( path_prefix, analyse_plot_file_template % template_format_dict )
    batch_utils.create_path( filename )
    print_engine.print_analyse_plot(
                                    random_mean,
                                    random_std,
                                    replicate_mean,
                                    replicate_std,
                                    labels,
                                    filename,
                                    title
    )


if analyse2_maps:

    print 'Analysing (2) combined similarity maps...'

    random_similarity_maps = []
    replicate_similarity_maps = []

    random_diagonals = None
    replicate_diagonals = None

    labels = None
    for map_file in random_similarity_map_files:

        f = open( os.path.splitext( map_file )[ 0 ] + '.pic', 'r' )
        import cPickle
        u = cPickle.Unpickler( f )
        map,l = u.load()
        f.close()

        if labels == None:
            labels = l

        if random_diagonals == None:
            random_diagonals = numpy.empty( ( len( random_similarity_map_files ), map.shape[0] ) )

        random_diagonals[ len( random_similarity_maps ) ] = map[ numpy.identity( map.shape[0], dtype=bool ) ]
        random_similarity_maps.append( map )

    labels = None
    for map_file in replicate_similarity_map_files:

        f = open( os.path.splitext( map_file )[ 0 ] + '.pic', 'r' )
        import cPickle
        u = cPickle.Unpickler( f )
        map,l = u.load()
        f.close()

        if labels == None:
            labels = l

        if replicate_diagonals == None:
            replicate_diagonals = numpy.empty( ( len( replicate_similarity_map_files ), map.shape[0] ) )

        replicate_diagonals[ len( replicate_similarity_maps ) ] = map[ numpy.identity( map.shape[0], dtype=bool ) ]
        replicate_similarity_maps.append( map )


    random_mean = numpy.mean( random_diagonals, axis=0 )
    random_std = numpy.std( random_diagonals, axis=0 )

    replicate_mean = numpy.mean( replicate_diagonals, axis=0 )
    replicate_std = numpy.std( replicate_diagonals, axis=0 )

    template_format_dict = clustering_param_set[ 0 ].copy()

    title = 'Analysation (2) of similarity maps for %(param1)d clusters' % template_format_dict

    filename = os.path.join( path_prefix, analyse2_plot_file_template % template_format_dict )
    batch_utils.create_path( filename )
    print_engine.print_analyse_plot( random_mean, random_std, replicate_mean, replicate_std, labels, filename, title )


if analyse3_maps:

    print 'Analysing (3) combined similarity maps...'

    mean_random_map_dict = {}
    mean_replicate_map_dict = {}
    std_random_map_dict = {}
    std_replicate_map_dict = {}

    for param1 in random_similarity_map_dict.keys():

        random_similarity_map_files = random_similarity_map_dict[ param1 ]
        random_similarity_maps = []

        summed_random_map = None
        labels = None
        for map_file in random_similarity_map_files:

            f = open( os.path.splitext( map_file )[ 0 ] + '.pic', 'r' )
            import cPickle
            u = cPickle.Unpickler( f )
            map,l = u.load()
            f.close()

            if labels == None:
                labels = l

            random_similarity_maps.append( map )

        replicate_similarity_map_files = replicate_similarity_map_dict[ param1 ]
        replicate_similarity_maps = []

        summed_replicate_map = None
        labels = None
        for map_file in replicate_similarity_map_files:

            f = open( os.path.splitext( map_file )[ 0 ] + '.pic', 'r' )
            import cPickle
            u = cPickle.Unpickler( f )
            map,l = u.load()
            f.close()

            if labels == None:
                labels = l

            replicate_similarity_maps.append( map )

        mean_random_map_dict[ param1 ] = numpy.mean( random_similarity_maps, axis=0 )
        std_random_map_dict[ param1 ] = numpy.std( random_similarity_maps, axis=0 )
        mean_replicate_map_dict[ param1 ] = numpy.mean( replicate_similarity_maps, axis=0 )
        std_replicate_map_dict[ param1 ] = numpy.std( replicate_similarity_maps, axis=0 )

    keys = mean_random_map_dict.keys()
    keys.sort()

    template_format_dict = clustering_param_set[ 0 ].copy()

    random_mean_list = []
    random_std_list = []
    replicate_mean_list = []
    replicate_std_list = []

    for i in xrange( len( labels ) ):

        l = labels[ i ]

        template_format_dict[ 'treatment' ] = l

        random_mean = numpy.empty( ( len( keys ), ) )
        replicate_mean = numpy.empty( ( len( keys ), ) )
        random_std = numpy.empty( ( len( keys ), ) )
        replicate_std = numpy.empty( ( len( keys ), ) )

        for j in xrange( len( keys ) ):
            key = keys[ j ]
            random_mean[ j ] = mean_random_map_dict[ key ][ i, i ]
            replicate_mean[ j ] = mean_replicate_map_dict[ key ][ i, i ]
            random_std[ j ] = std_random_map_dict[ key ][ i, i ]
            replicate_std[ j ] = std_replicate_map_dict[ key ][ i, i ]

        random_mean_list.append( random_mean )
        random_std_list.append( random_std )
        replicate_mean_list.append( replicate_mean )
        replicate_std_list.append( replicate_std )

        title = 'Analysation (3) of similarity maps for treatment %(treatment)s' % template_format_dict

        xlabel = '# of clusters'

        key_labels = []
        for key in keys:
            key_labels.append( int( key ) )

        filename = os.path.join( path_prefix, analyse3_plot_file_template % template_format_dict )
        batch_utils.create_path( filename )
        print_engine.print_analyse_plot( random_mean, random_std, replicate_mean, replicate_std, key_labels, filename, title, xlabel )


    random_mean = numpy.mean( random_mean_list, axis=0 )
    replicate_mean = numpy.mean( replicate_mean_list, axis=0 )
    random_std = numpy.mean( random_std_list, axis=0 )
    replicate_std = numpy.mean( replicate_std_list, axis=0 )

    title = 'Analysation (3) of similarity maps averaged over all treatments'

    template_format_dict[ 'treatment' ] = 'ALL'

    xlabel = '# of clusters'

    key_labels = []
    for key in keys:
        key_labels.append( int( key ) )

    filename = os.path.join( path_prefix, analyse3_plot_file_template % template_format_dict )
    batch_utils.create_path( filename )
    print_engine.print_analyse_plot( random_mean, random_std, replicate_mean, replicate_std, key_labels, filename, title, xlabel )



if analyse_map_quality:

    print 'Analysing combined similarity map qualities...'

    mean_random_map_qualities = []
    std_random_map_qualities = []
    mean_replicate_map_qualities = []
    std_replicate_map_qualities = []
    labels = []

    for map_file in map_quality_files:

        f = open( os.path.splitext( map_file )[ 0 ] + '.pic', 'r' )
        import cPickle
        u = cPickle.Unpickler( f )
        d = u.load()
        f.close()

        print map_file
        print d.keys()
        print d[ 'profile_metric' ]

        j = numpy.argmax( d[ 'mean_replicate_map_qualities' ] )
        mean_random_map_qualities.append( d[ 'mean_random_map_qualities' ][ j ] )
        std_random_map_qualities.append( d[ 'mean_random_map_qualities' ][ j ] )
        mean_replicate_map_qualities.append( d[ 'mean_replicate_map_qualities' ][ j ] )
        std_replicate_map_qualities.append( d[ 'mean_replicate_map_qualities' ][ j ] )
        labels.append( d[ 'profile_metric' ] )

    mean_random_map_qualities = numpy.array( mean_random_map_qualities )
    std_random_map_qualities = numpy.array( std_random_map_qualities )
    mean_replicate_map_qualities = numpy.array( mean_replicate_map_qualities )
    std_replicate_map_qualities = numpy.array( std_replicate_map_qualities )

    template_format_dict = clustering_param_set[ 0 ].copy()

    title = 'Analysation of combined similarity map qualities over all profile metrics'

    xlabel = 'profile metric'

    filename = os.path.join( path_prefix, analyse_map_quality_plot_file_template % template_format_dict )
    batch_utils.create_path( filename )
    print_engine.print_analyse_plot( mean_random_map_qualities, std_random_map_qualities, mean_replicate_map_qualities, std_replicate_map_qualities, labels, filename, title, xlabel )


if job_state_filename != None:
    try:
        os.remove( os.path.join( path_prefix, job_state_filename ) )
    except:
        pass
