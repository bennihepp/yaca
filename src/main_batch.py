import sys, os
import time
import numpy



config_file = None

clustering_method_list = [ 'k-means' ]
clustering_index_list = [ 0 ]
clustering_param1_list = [ 100 ]
clustering_param2_list = [ -1 ]
clustering_param3_list = [ 2 ]
clustering_param4_list = [ 20 ]
clustering_exp_factor_list = [ -1 ]
clustering_file_template = 'clustering_%(project_name)_%(method)s_(%(index)d)_(%(param1)d,%(param2)d,%(param3)d,%(param4)d),%(exp_factor).1f.pic'
profile_pdf_file_template = 'profiles_%(project_name)_%(method)s_(%(index)d)_(%(param1)d,%(param2)d,%(param3)d,%(param4)d),%(exp_factor).1f.pdf'
profile_xls_file_template = 'profiles_%(project_name)_%(method)s_(%(index)d)_(%(param1)d,%(param2)d,%(param3)d,%(param4)d),%(exp_factor).1f.xls'
similarity_map_file_template = 'similarity_map_%(project_name)s_%(method)s_(%(index)d)_(%(param1)d,%(param2)d,%(param3)d,%(param4)d),%(exp_factor).1f.pdf'
combined_map_file_template = 'combined_map_%(project_name)s_%(method)s_(%(index)d)_(%(param1)d,%(param2)d,%(param3)d,%(param4)d),%(exp_factor).1f.pdf'
combined2_map_file_template = 'combined2_map_%(project_name)s_%(method)s_(%(index)d)_(%(param1)d,%(param2)d,%(param3)d,%(param4)d),%(exp_factor).1f.pdf'
combined2_results_file_template = 'combined2_results_%(method)s_(%(index)d)_(%(param2)d,%(param3)d,%(param4)d),%(exp_factor).1f.pdf'
analyse_plot_file_template = 'analyse_plot_%(method)s_(%(index)d)_(%(param2)d,%(param3)d,%(param4)d),%(exp_factor).1f.pdf'
analyse2_plot_file_template = 'analyse2_plot_%(method)s_(%(index)d)_(%(param1)d,%(param2)d,%(param3)d,%(param4)d),%(exp_factor).1f.pdf'
analyse3_plot_file_template = 'analyse3_plot_%(method)s_(%(index)d)_(%(param2)d,%(param3)d,%(param4)d),%(exp_factor).1f.pdf'
only_run_clustering = False
combine_maps = True
only_combine_maps = False
combine2_maps = False
only_combine_maps = False
clustering_file = None
population_plot_file_template = 'population_plots/population_plot_%(project_name)s.pdf'
print_population_plot = False
analyse_maps = False
only_analyse_maps = False
analyse2_maps = False
only_analyse2_maps = False
analyse3_maps = False
only_analyse3_maps = False
profile_metric = 'summed_minmax'
control_filter_mode = 'MEDIAN_xMAD'
invert_profile_heatmap = False

skip_next = 0


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
            l1.extend( y.split( ',' ) )
        l2 = []
        for y in l1:
            l2.extend( y.split( ';' ) )
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
        elif arg == '--clustering-index':
            clustering_index_list = make_clustering_param_list( next_arg )
        elif arg == '--clustering-param1':
            clustering_param1_list = make_clustering_param_list( next_arg )
        elif arg == '--clustering-param2':
            clustering_param2_list = make_clustering_param_list( next_arg )
        elif arg == '--clustering-param3':
            clustering_param3_list = make_clustering_param_list( next_arg )
        elif arg == '--clustering-param4':
            clustering_param4_list = make_clustering_param_list( next_arg )
        elif arg == '--clustering-exp-factor':
            clustering_exp_factor_list = make_clustering_param_list( next_arg )
        elif arg == '--config-file':
            config_file = next_arg
            skip_next = 1
        elif arg == '--help':
            print_help()
            sys.exit( 0 )
        else:
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
    if 'only_analyse_maps' in general_config: only_analyse_maps = general_config[ 'only_analyse_maps' ]
    if 'analyse_maps' in general_config: analyse_maps = general_config[ 'analyse_maps' ]
    if 'only_analyse2_maps' in general_config: only_analyse2_maps = general_config[ 'only_analyse2_maps' ]
    if 'analyse2_maps' in general_config: analyse2_maps = general_config[ 'analyse2_maps' ]
    if 'only_analyse3_maps' in general_config: only_analyse3_maps = general_config[ 'only_analyse3_maps' ]
    if 'analyse3_maps' in general_config: analyse3_maps = general_config[ 'analyse3_maps' ]
    if not ( analyse_maps and only_analyse_maps ) and not ( analyse2_maps and only_analyse2_maps ) and not ( analyse3_maps and only_analyse3_maps ):
        project_files = general_config[ 'project_files' ]
    #treatments_to_use = general_config[ 'treatments_to_use' ]
    if 'profile_metric' in general_config: profile_metric = general_config[ 'profile_metric' ]
    if 'control_filter_mode' in general_config: control_filter_mode = general_config[ 'control_filter_mode' ]
    if 'profile_pdf_file_template' in general_config: profile_pdf_file_template = general_config[ 'profile_pdf_file_template' ]
    if 'profile_xls_file_template' in general_config: profile_xls_file_template = general_config[ 'profile_xls_file_template' ]
    if 'similarity_map_file_template' in general_config: similarity_map_file_template = general_config[ 'similarity_map_file_template' ]
    if 'combined_map_file_template' in general_config: combined_map_file_template = general_config[ 'combined_map_file_template' ]
    if 'combined2_random_map_file_template' in general_config: combined2_random_map_file_template = general_config[ 'combined2_random_map_file_template' ]
    if 'combined2_replicate_map_file_template' in general_config: combined2_replicate_map_file_template = general_config[ 'combined2_replicate_map_file_template' ]
    if 'combined2_results_file_template' in general_config: combined2_results_file_template = general_config[ 'combined2_results_file_template' ]
    if 'analyse_plot_file_template' in general_config: analyse_plot_file_template = general_config[ 'analyse_plot_file_template' ]
    if 'analyse2_plot_file_template' in general_config: analyse2_plot_file_template = general_config[ 'analyse2_plot_file_template' ]
    if 'analyse3_plot_file_template' in general_config: analyse3_plot_file_template = general_config[ 'analyse3_plot_file_template' ]
    if 'only_run_clustering' in general_config: only_run_clustering = general_config[ 'only_run_clustering' ]
    if 'only_combine_maps' in general_config: only_combine_maps = general_config[ 'only_combine_maps' ]
    if 'combine_maps' in general_config: combine_maps = general_config[ 'combine_maps' ]
    if 'only_combine2_maps' in general_config: only_combine2_maps = general_config[ 'only_combine2_maps' ]
    if 'combine2_maps' in general_config: combine2_maps = general_config[ 'combine2_maps' ]
    if combine_maps and only_combine_maps:
        combine2_maps = False
        analyse_maps = False
        analyse2_maps = False
        analyse3_maps = False
        similarity_map_files = general_config[ 'similarity_map_files' ]
    else:
        similarity_map_files = []
    if combine2_maps and only_combine2_maps:
        combine_maps = False
        analyse_maps = False
        analyse2_maps = False
        analyse3_maps = False
        random_similarity_map_dict2 = general_config[ 'random_similarity_map_dict' ]
        replicate_similarity_map_dict2 = general_config[ 'replicate_similarity_map_dict' ]
    else:
        random_similarity_map_dict2 = {}
        replicate_similarity_map_dict2 = {}
    if analyse_maps and only_analyse_maps:
        combine_maps = False
        combine2_maps = False
        analyse2_maps = False
        analyse3_maps = False
        random_combined_map_files = general_config[ 'random_combined_map_files' ]
        replicate_combined_map_files = general_config[ 'replicate_combined_map_files' ]
    else:
        random_combined_map_files = []
        replicate_combined_map_files = []
    if analyse2_maps and only_analyse2_maps:
        combine_maps = False
        combine2_maps = False
        analyse_maps = False
        analyse3_maps = False
        random_similarity_map_files = general_config[ 'random_similarity_map_files' ]
        replicate_similarity_map_files = general_config[ 'replicate_similarity_map_files' ]
    else:
        random_similarity_map_files = []
        replicate_similarity_map_files = []
    if analyse3_maps and only_analyse3_maps:
        combine_maps = False
        combine2_maps = False
        analyse_maps = False
        analyse2_maps = False
        random_similarity_map_dict = general_config[ 'random_similarity_map_dict' ]
        replicate_similarity_map_dict = general_config[ 'replicate_similarity_map_dict' ]
    else:
        random_similarity_map_dict = {}
        replicate_similarity_map_dict = {}
    if 'clustering_file' in general_config: clustering_file = general_config[ 'clustering_file' ]
    if 'print_population_plot' in general_config: print_population_plot = general_config[ 'print_population_plot' ]
    if print_population_plot:
        if 'population_plot_file_template' in general_config: population_plot_file_template = general_config[ 'population_plot_file_template' ]
except:
    print 'Invalid YACA batch configuration file'
    raise

try:
    clustering_config = yaml_container[ 'clustering_config' ]
    if 'method' in clustering_config: clustering_method_list = make_clustering_param_list( clustering_config[ 'method' ] )
    if 'index' in clustering_config: clustering_index_list = make_clustering_param_list( clustering_config[ 'index' ] )
    if 'param1' in clustering_config: clustering_param1_list = make_clustering_param_list( clustering_config[ 'param1' ] )
    if 'param2' in clustering_config: clustering_param2_list = make_clustering_param_list( clustering_config[ 'param2' ] )
    if 'param3' in clustering_config: clustering_param3_list = make_clustering_param_list( clustering_config[ 'param3' ] )
    if 'param4' in clustering_config: clustering_param4_list = make_clustering_param_list( clustering_config[ 'param4' ] )
    if 'exp-factor' in clustering_config: clustering_exp_factor_list = make_clustering_param_list( clustering_config[ 'exp-factor' ] )
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
            'exp_factor' : clustering_exp_factor_list
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

clustering_files = []

if not ( combine_maps and only_combine_maps ) and not ( combine2_maps and only_combine2_maps ) and not ( analyse_maps and only_analyse_maps ) and not ( analyse2_maps and only_analyse2_maps ) and not ( analyse3_maps and only_analyse3_maps ):

    for project_name,project_file in project_files.iteritems():


        treatment_similarity_maps[ project_name ] = {}


        print 'Running pipeline for project %s (project_file %s)' % ( project_name, project_file )

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

        #sys.stdout.write( '\n' )
        #sys.stdout.flush()

        if control_filter_mode == 'MEDIAN_xMAD':
            filter_mode = pipeline.analyse.FILTER_MODE_MEDIAN_xMAD
        elif control_filter_mode == 'xMEDIAN':
            filter_mode = pipeline.analyse.FILTER_MODE_xMEDIAN
        else:
            raise Exception( 'Unknown control filter mode: %s' % control_filter_mode )

        print '  Running pre filtering with control filter mode "%s"...' % control_filter_mode

        pl.start_pre_filtering( filter_mode )

        pl.wait_safe()

        #sys.stdout.write( '\n' )
        #sys.stdout.flush()

        pl.disconnect( pl, SIGNAL('updateProgress'), callback_pipeline_update_progress )
        #pl.disconnect( pl, SIGNAL('finished()'), callback_pipeline_finished )


        if print_population_plot:
            print '  Printing population plots...'
            filename = os.path.join( path_prefix, population_plot_file_template %  { 'project_name' : project_name } )
            batch_utils.create_path( filename )
            print_engine.print_cell_populations_and_penetrance( pl, filename )


        for i in xrange( len( clustering_param_set ) ):

            clustering_param_dict = clustering_param_set[ i ]

            template_format_dict = clustering_param_dict.copy()
            template_format_dict[ 'project_name' ] = project_name

            keys = clustering_param_dict.keys()
            keys.sort()
            keys.remove( 'param1' )
            values = []
            for k in keys:
                values.append( clustering_param_dict[ k ] )
            values = tuple( values )
            if not values in treatment_similarity_maps[ project_name ]:
                treatment_similarity_maps[ project_name ][ values ] = ( template_format_dict, [] )

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


            if i < len( clustering_files ):

                print '  Loading clustering file for param1=%(param1)d, param2=%(param2)d, param3=%(param3)d, param4=%(param4)d, exp_factor=%(exp_factor).1f...' \
                        % template_format_dict

                pl.load_clusters( clustering_files[ i ], clustering_param_dict[ 'exp_factor' ] )

            elif clustering_file:

                print '  Loading clustering file for param1=%(param1)d, param2=%(param2)d, param3=%(param3)d, param4=%(param4)d, exp_factor=%(exp_factor).1f...' \
                        % template_format_dict

                filename = os.path.join( path_prefix, clustering_file )
                pl.load_clusters( filename, clustering_param_dict[ 'exp_factor' ] )

            else:

                pl.connect( pl, SIGNAL('updateProgress'), callback_pipeline_update_progress )
                #pl.connect( pl, SIGNAL('finished()'), callback_pipeline_finished )

                """clustering_method = clustering_param_dict[ 'method' ]
                clustering_index = clustering_param_dict[ 'index' ]
                clustering_param1 = clustering_param_dict[ 'param1' ]
                clustering_param2= clustering_param_dict[ 'param2' ]
                clustering_param3 = clustering_param_dict[ 'param3' ]
                clustering_param4 = clustering_param_dict[ 'param4' ]"""

                print '  Running %(method)s clustering with index=%(index)d, param1=%(param1)d, param2=%(param2)d, param3=%(param3)d, param4=%(param4)d, exp_factor=%(exp_factor).1f...' \
                        % template_format_dict
                #pl.start_clustering( clustering_method, clustering_index, clustering_param1, clustering_param2, clustering_param3, clustering_param4 )
                #f = open('/dev/null','w')
                #sys.stdout = f
                pl.start_clustering( **clustering_param_dict )

                pl.wait_safe()
                #sys.stdout = sys.__stdout__
                #f.close()

                pl.disconnect( pl, SIGNAL('updateProgress'), callback_pipeline_update_progress )
                #pl.disconnect( pl, SIGNAL('finished()'), callback_pipeline_finished )

                filename = os.path.join( path_prefix, clustering_file_template % template_format_dict )
                batch_utils.create_path( filename )
                clustering_files.append( filename )

                print '  Saving clustering file...'

                pl.save_clusters( filename )

                print '  Printing clustering plot...'
                print_engine.print_clustering_plot( pl, filename + '.pdf' )

            if not only_run_clustering:

                print 'Printing cluster profiles...'

                pdf_filename = os.path.join( path_prefix, profile_pdf_file_template % template_format_dict )
                batch_utils.create_path( pdf_filename)
                sum_minmax_profileHeatmap, l2_chi2_norm_profileHeatmap, treatments = print_engine.print_cluster_profiles_and_heatmap( pl.pdc, pl.nonControlClusterProfiles, pdf_filename, True, 0.0 )

                print 'Writing profile heatmap xls file...'

                xls_title = 'Profile heatmap for %(project_name)s with method=%(method)s, index=%(index)d, param1=%(param1)d, param2=%(param2)d, param3=%(param3)d, param4=%(param4)d, exp_factor=%(exp_factor).1f' \
                            % template_format_dict
                xls_filename = os.path.join( path_prefix, profile_xls_file_template % template_format_dict )
                batch_utils.create_path( xls_filename )
                batch_utils.write_profileHeatmapCSV( xls_title, treatments, sum_minmax_profileHeatmap, xls_filename )

                print 'profile_metric:', profile_metric

                if profile_metric == 'summed_minmax':
                    invert_profile_heatmap = True
                    profileHeatmap = sum_minmax_profileHeatmap
                    #for i in xrange( profileHeatmap.shape[0] ):
                    #    for j in xrange( i ):
                    #        profileHeatmap[ j, i ] = profileHeatmap[ i, j ]
                #elif profile_metric == 'l1_norm':
                #    profileHeatmap = sum_minmax_profileHeatmap
                #    for i in xrange( profileHeatmap.shape[0] ):
                #        for j in xrange( i, profileHeatmap.shape[0] ):
                #            profileHeatmap[ j, i ] = profileHeatmap[ i, j ]
                elif profile_metric == 'l2_norm':
                    profileHeatmap = l2_chi2_norm_profileHeatmap
                    for i in xrange( profileHeatmap.shape[0] ):
                        for j in xrange( i ):
                            profileHeatmap[ j, i ] = profileHeatmap[ i, j ]
                elif profile_metric == 'chi2_norm':
                    profileHeatmap = l2_chi2_norm_profileHeatmap
                    for i in xrange( profileHeatmap.shape[0] ):
                        for j in xrange( i, profileHeatmap.shape[0] ):
                            profileHeatmap[ j, i ] = profileHeatmap[ i, j ]
                else:
                    raise Exception( 'No such profile metric: %s' % profile_metric )

                print 'Computing similarity map...'

                treatmentMask = numpy.ones( ( len( pdc.treatments ), ), dtype=bool )
                #fullTreatmentMask = numpy.ones( ( len( pdc.treatments ), len( pdc.treatments ) ), dtype=bool )
                for i in xrange( len( pdc.treatments ) ):
                    tr = pdc.treatments[ i ]
                    tr_mask = pdc.objFeatures[ pl.nonControlCellMask, pdc.objTreatmentFeatureId ] == tr.rowId
                    if numpy.sum( tr_mask ) == 0:
                        treatmentMask[ i ] = False
                        #fullTreatmentMask[ i, : ] = False
                        #fullTreatmentMask[ :, i ] = False

                num_of_treatments = numpy.sum( treatmentMask )

                old_treatments = treatments
                treatments = []
                for i in xrange( 0, len( old_treatments ), 2 ):
                    treatments.append( old_treatments[ i ] )

                labels = []
                for tr in treatments:
                    labels.append( tr.name )
                #for i in xrange( len( pdc.treatments ) ):
                #    if treatmentMask[ i ] and not pdc.treatments[i].name.endswith('*'):
                #        labels.append( pdc.treatments[ i ].name )

                treatmentSimilarityMap = batch_utils.compute_treatment_similarity_map( profileHeatmap )

                #maskedProfileHeatmap = profileHeatmap[ treatmentMask ][ :, treatmentMask ]

                print treatmentSimilarityMap

                print 'Printing treatment similarity map...'

                #intTreatmentSimilarityMap = numpy.array( treatmentSimilarityMap, dtype=int )
                floatTreatmentSimilarityMap = numpy.array( treatmentSimilarityMap, dtype=float )

                filename = os.path.join( path_prefix, similarity_map_file_template % template_format_dict )
                batch_utils.create_path( filename )
                if invert_profile_heatmap:
                    print_engine.print_treatment_similarity_map( 1.0 - floatTreatmentSimilarityMap, labels, filename )
                else:
                    print_engine.print_treatment_similarity_map( floatTreatmentSimilarityMap, labels, filename )

                print 'Writing similarity map xls file...'

                xls_title = 'Similarity map for %(project_name)s with method=%(method)s, index=%(index)d, param1=%(param1)d, param2=%(param2)d, param3=%(param3)d, param4=%(param4)d, exp_factor=%(exp_factor).1f' \
                            % template_format_dict
                batch_utils.write_similarityMapCSV( xls_title, labels, floatTreatmentSimilarityMap, filename+'.xls' )

                f = open( filename + '.pic', 'w' )
                import cPickle
                p = cPickle.Pickler( f )
                p.dump( ( treatmentSimilarityMap, labels ) )
                f.close()

                similarity_map_files.append( filename )

                treatment_similarity_maps[ project_name ][ values ][ 1 ].append( floatTreatmentSimilarityMap )

if combine_maps:

    print 'Computing combined similarity maps...'

    for project_name,project_file in project_files.iteritems():



        """print 'Running pipeline for project %s (project_file %s)' % ( project_name, project_file )

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

            f = open( sim_map_file + '.pic', 'r' )
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

        print 'Printing combined similarity map...'

        template_format_dict = clustering_param_set[ 0 ].copy()
        template_format_dict[ 'project_name' ] = project_name

        title = 'Combined similarity map of treatments'

        if profile_metric == 'summed_minmax':
            invert_profile_heatmap = True

        filename = os.path.join( path_prefix, combined_map_file_template % template_format_dict )
        batch_utils.create_path( filename )
        if invert_profile_heatmap:
            print_engine.print_treatment_similarity_map( 1.0 - summed_map, labels, filename, title )
        else:
            print_engine.print_treatment_similarity_map( summed_map, labels, filename, title )

        if project_name.startswith( 'random' ):
            random_combined_map_files.append( filename )
        else:
            replicate_combined_map_files.append( filename )

        print 'Writing similarity map xls file...'

        xls_title = 'Similarity map for %(project_name)s with method=%(method)s, index=%(index)d, param1=%(param1)d, param2=%(param2)d, param3=%(param3)d, param4=%(param4)d, exp_factor=%(exp_factor).1f' \
                    % template_format_dict
        batch_utils.write_similarityMapCSV( xls_title, labels, summed_map, filename+'.xls' )

        f = open( filename + '.pic', 'w' )
        import cPickle
        p = cPickle.Pickler( f )
        p.dump( ( summed_map, labels ) )
        f.close()


if combine2_maps:

    print 'Computing combined (2) similarity maps...'

    template_format_dict = clustering_param_set[ 0 ].copy()

    keys = random_similarity_map_dict2.keys()
    keys.sort()

    mean_random_map_dict = {}
    mean_replicate_map_dict = {}
    #std_random_map_dict = {}
    #std_replicate_map_dict = {}

    for param1 in keys:

        template_format_dict[ 'param1' ] = param1

        random_similarity_map_files = random_similarity_map_dict2[ param1 ]
        random_similarity_maps = []

        print len( random_similarity_map_dict2 )
        print len( random_similarity_map_files )

        summed_random_map = None
        labels = None
        for map_file in random_similarity_map_files:

            f = open( map_file + '.pic', 'r' )
            import cPickle
            u = cPickle.Unpickler( f )
            map,l = u.load()
            f.close()

            if labels == None:
                labels = l

            random_similarity_maps.append( map )

        replicate_similarity_map_files = replicate_similarity_map_dict2[ param1 ]
        replicate_similarity_maps = []

        summed_replicate_map = None
        labels = None
        for map_file in replicate_similarity_map_files:

            f = open( map_file + '.pic', 'r' )
            import cPickle
            u = cPickle.Unpickler( f )
            map,l = u.load()
            f.close()

            if labels == None:
                labels = l

            replicate_similarity_maps.append( map )

        mean_random_map_dict[ param1 ] = numpy.mean( random_similarity_maps, axis=0 )
        #std_random_map_dict[ param1 ] = numpy.std( random_similarity_maps, axis=0 )
        mean_replicate_map_dict[ param1 ] = numpy.mean( replicate_similarity_maps, axis=0 )
        #std_replicate_map_dict[ param1 ] = numpy.std( replicate_similarity_maps, axis=0 )


        print 'Printing combined (2) similarity maps for %(param1)d clusters...' % template_format_dict

        title = 'Combined similarity map for %(param1)d clusters and random splits' % template_format_dict

        if profile_metric == 'summed_minmax':
            invert_profile_heatmap = True

        filename = os.path.join( path_prefix, combined2_random_map_file_template % template_format_dict )
        batch_utils.create_path( filename )
        if invert_profile_heatmap:
            print_engine.print_treatment_similarity_map( 1.0 - mean_random_map_dict[ param1 ], labels, filename, title )
        else:
            print_engine.print_treatment_similarity_map( mean_random_map_dict[ param1 ], labels, filename, title )

        xls_title = 'Combined2 random map with method=%(method)s, index=%(index)d, param1=%(param1)d, param2=%(param2)d, param3=%(param3)d, param4=%(param4)d, exp_factor=%(exp_factor).1f' \
                    % template_format_dict
        batch_utils.write_similarityMapCSV( xls_title, labels, mean_random_map_dict[ param1 ], filename+'.xls' )

        f = open( filename + '.pic', 'w' )
        import cPickle
        p = cPickle.Pickler( f )
        p.dump( ( mean_random_map_dict[ param1 ], labels ) )
        f.close()

        title = 'Combined similarity map for %(param1)d clusters and replicate splits' % template_format_dict

        filename = os.path.join( path_prefix, combined2_replicate_map_file_template % template_format_dict )
        batch_utils.create_path( filename )
        if invert_profile_heatmap:
            print_engine.print_treatment_similarity_map( 1.0 - mean_replicate_map_dict[ param1 ], labels, filename, title )
        else:
            print_engine.print_treatment_similarity_map( mean_replicate_map_dict[ param1 ], labels, filename, title )

        xls_title = 'Combined2 replicate map with method=%(method)s, index=%(index)d, param1=%(param1)d, param2=%(param2)d, param3=%(param3)d, param4=%(param4)d, exp_factor=%(exp_factor).1f' \
                    % template_format_dict
        batch_utils.write_similarityMapCSV( xls_title, labels, mean_replicate_map_dict[ param1 ], filename+'.xls' )

        f = open( filename + '.pic', 'w' )
        import cPickle
        p = cPickle.Pickler( f )
        p.dump( ( mean_replicate_map_dict[ param1 ], labels ) )
        f.close()

    random_results = []
    replicate_results = []

    random_str = 'random splits\t'
    replicate_str = 'replicate splits\t'
    for key in keys:
        def compute_map_quality(map):
            diagonal_mask = numpy.identity( map.shape[0], dtype=bool )
            non_diagonal_mask = numpy.invert( diagonal_mask )
            diagonal_mean = numpy.mean( map[ diagonal_mask ] )
            non_diagonal_mean = numpy.mean( map[ non_diagonal_mask ] )
            quality = diagonal_mean - non_diagonal_mean
            return quality
        random_quality = compute_map_quality( mean_random_map_dict[ key ] )
        replicate_quality = compute_map_quality( mean_replicate_map_dict[ key ] )
        random_results.append( random_quality )
        replicate_results.append( replicate_quality )
        random_str += '%.2f\t' % random_quality
        replicate_str += '%.2f\t' % replicate_quality
    random_str = random_str[ :-1 ] + '\n'
    replicate_str = replicate_str[ :-1 ] + '\n'

    filename = os.path.join( path_prefix, combined2_results_file_template % template_format_dict )
    batch_utils.create_path( filename )
    print_engine.print_analyse_plot( random_results, None, replicate_results, None, keys, filename, 'Similarity map qualities', '# of clusters', 'quality' )

    filename = filename + '.xls'
    batch_utils.create_path( filename )
    f2 = open( filename, 'w' )

    str = '\t'
    for key in keys:
        str += '%s clusters\t' % key
    str = str[ :-1 ] + '\n'
    f2.write( str )

    f2.write( random_str )
    f2.write(replicate_str )

    f2.close()

    print 'done'



if analyse_maps:

    print 'Analysing combined similarity maps...'

    random_combined_maps = []
    replicate_combined_maps = []

    random_diagonals = None
    replicate_diagonals = None

    labels = None
    for map_file in random_combined_map_files:

        f = open( map_file + '.pic', 'r' )
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

        f = open( map_file + '.pic', 'r' )
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
    print_engine.print_analyse_plot( random_mean, random_std, replicate_mean, replicate_std, labels, filename, title )


if analyse2_maps:

    print 'Analysing (2) combined similarity maps...'

    random_similarity_maps = []
    replicate_similarity_maps = []

    random_diagonals = None
    replicate_diagonals = None

    labels = None
    for map_file in random_similarity_map_files:

        f = open( map_file + '.pic', 'r' )
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

        f = open( map_file + '.pic', 'r' )
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

            f = open( map_file + '.pic', 'r' )
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

            f = open( map_file + '.pic', 'r' )
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

    
