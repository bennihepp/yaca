#!/usr/bin/env python

import yaml
import sys
import os

qsub_host = 'sub-master'
#qsub_host = 'sub-master-vm'


try:
    config_file = sys.argv[1]
except:
    print 'YACA batch configuration file needs to be specified...'
    sys.exit( 1 )

do_submit_jobs = True
if len( sys.argv ) > 2:
    if sys.argv[ 2 ] == '--no-submit':
        do_submit_jobs = False


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


file = None
try:
    file = open( config_file, 'r' )
    yaml_container = yaml.load( file )
finally:
    if file:
        file.close()


num_of_random_splits = 3
size_of_random_splits = 0.5
pdf_map_fontsize = 6
pdf_map_label_fontsize = 8
pdf_cluster_label_fontsize = 10
batch_config_file = 'batch/batch_config.yaml'
project_setting_file = 'batch/yaca_settings.yaml'
config_file_template = 'configs/config_%(profile_metric)s_%(method)s_(%(index)d)_(%(param1)d,%(param2)d,%(param3)d,%(param4)d,%(exp_factor).2f).yaml'
config_cluster_file_template = 'configs/config_cluster_%(method)s_(%(index)d)_(%(param1)d,%(param2)d,%(param3)d,%(param4)d,%(exp_factor).2f).yaml'
config_combine_file_template = 'configs/config_combine_%(profile_metric)s_%(method)s_(%(index)d)_(%(param2)d,%(param3)d,%(param4)d,%(exp_factor).2f).yaml'
config_map_quality_file_template = 'configs/config_map_quality_%(profile_metric)s_%(method)s_(%(index)d)_(%(param1)d,%(param2)d,%(param3)d,%(param4)d,%(exp_factor).2f).yaml'
config_analyse_file_template = 'configs/config_analyse_%(profile_metric)s_%(method)s_(%(index)d)_(%(param2)d,%(param3)d,%(param4)d,%(exp_factor).2f).yaml'
config_analyse2_file_template = 'configs/config_analyse2_%(profile_metric)s_%(method)s_(%(index)d)_(%(param1)d,%(param2)d,%(param3)d,%(param4)d,%(exp_factor).2f).yaml'
config_analyse3_file_template = 'configs/config_analyse3_%(profile_metric)s_%(method)s_(%(index)d)_(%(param2)d,%(param3)d,%(param4)d,%(exp_factor).2f).yaml'
config_analyse_map_quality_file_template = 'configs/config_analyse_map_quality_%(method)s_(%(index)d)_(%(param2)d,%(param3)d,%(param4)d,%(exp_factor).2f).yaml'
pbs_script_file_template = 'jobs/job_%(profile_metric)s_%(method)s_(%(index)d)_(%(param1)d,%(param2)d,%(param3)d,%(param4)d,%(exp_factor).2f).sh'
pbs_cluster_script_file_template = 'jobs/job_cluster_%(method)s_(%(index)d)_(%(param1)d,%(param2)d,%(param3)d,%(param4)d,%(exp_factor).2f).sh'
pbs_combine_script_file_template = 'jobs/job_combine_%(profile_metric)s_%(method)s_(%(index)d)_(%(param2)d,%(param3)d,%(param4)d,%(exp_factor).2f).sh'
pbs_map_quality_script_file_template = 'jobs/job_map_quality_%(profile_metric)s_%(method)s_(%(index)d)_(%(param1)d,%(param2)d,%(param3)d,%(param4)d,%(exp_factor).2f).sh'
pbs_analyse_script_file_template = 'jobs/job_analyse_%(profile_metric)s_%(method)s_(%(index)d)_(%(param2)d,%(param3)d,%(param4)d,%(exp_factor).2f).sh'
pbs_analyse2_script_file_template = 'jobs/job_analyse2_%(profile_metric)s_%(method)s_%(index)d)_(%(param1)d,%(param2)d,%(param3)d,%(param4)d,%(exp_factor).2f).sh'
pbs_analyse3_script_file_template = 'jobs/job_analyse3_%(profile_metric)s_%(method)s_(%(index)d)_(%(param2)d,%(param3)d,%(param4)d,%(exp_factor).2f).sh'
pbs_analyse_map_quality_script_file_template = 'jobs/job_analyse_map_quality_%(method)s_(%(index)d)_(%(param2)d,%(param3)d,%(param4)d,%(exp_factor).2f).sh'
output_log_template = 'logs/output_%(profile_metric)s_%(method)s__%(index)d__%(param1)d_%(param2)d_%(param3)d_%(param4)d_%(exp_factor).2f.txt'
output_cluster_log_template = 'logs/output_cluster_%(method)s__%(index)d__%(param1)d_%(param2)d_%(param3)d_%(param4)d_%(exp_factor).2f.txt'
output_combine_log_template = 'logs/output_combine_%(profile_metric)s_%(method)s__%(index)d__%(param2)d_%(param3)d_%(param4)d_%(exp_factor).2f.txt'
output_map_quality_log_template = 'logs/output_map_quality_%(profile_metric)s_%(method)s__%(index)d__%(param1)d_%(param2)d_%(param3)d_%(param4)d_%(exp_factor).2f.txt'
output_analyse_log_template = 'logs/output_analyse_%(profile_metric)s_%(method)s__%(index)d__%(param2)d_%(param3)d_%(param4)d_%(exp_factor).2f.txt'
output_analyse2_log_template = 'logs/output_analyse2_%(profile_metric)s_%(method)s__%(index)d__%(param1)d_%(param2)d_%(param3)d_%(param4)d_%(exp_factor).2f.txt'
output_analyse3_log_template = 'logs/output_analyse3_%(profile_metric)s_%(method)s__%(index)d__%(param2)d_%(param3)d_%(param4)d_%(exp_factor).2f.txt'
output_analyse_map_quality_log_template = 'logs/output_analyse_map_quality_%(method)s__%(index)d__%(param2)d_%(param3)d_%(param4)d_%(exp_factor).2f.txt'
error_log_template = 'logs/error_%(profile_metric)s_%(method)s__%(index)d__%(param1)d_%(param2)d_%(param3)d_%(param4)d_%(exp_factor).2f.txt'
error_cluster_log_template = 'logs/error_cluster_%(method)s__%(index)d__%(param1)d_%(param2)d_%(param3)d_%(param4)d_%(exp_factor).2f.txt'
error_combine_log_template = 'logs/error_combine_%(profile_metric)s_%(method)s__%(index)d__%(param2)d_%(param3)d_%(param4)d_%(exp_factor).2f.txt'
error_map_quality_log_template = 'logs/error_map_quality_%(profile_metric)s_%(method)s__%(index)d__%(param1)d_%(param2)d_%(param3)d_%(param4)d_%(exp_factor).2f.txt'
error_analyse_log_template = 'logs/error_analyse_%(profile_metric)s_%(method)s__%(index)d__%(param2)d_%(param3)d_%(param4)d_%(exp_factor).2f.txt'
error_analyse2_log_template = 'logs/error_analyse2_%(profile_metric)s_%(method)s__%(index)d__%(param1)d_%(param2)d_%(param3)d_%(param4)d_%(exp_factor).2f.txt'
error_analyse3_log_template = 'logs/error_analyse3_%(profile_metric)s_%(method)s__%(index)d__%(param2)d_%(param3)d_%(param4)d_%(exp_factor).2f.txt'
error_analyse_map_quality_log_template = 'logs/error_analyse_map_quality_%(method)s__%(index)d__%(param2)d_%(param3)d_%(param4)d_%(exp_factor).2f.txt'
submit_script_filename = 'jobs/submit_jobs.sh'
root_log_file = 'error_summary.txt'
root_log_id_template = 'yaca-job profile_metric=%(profile_metric)s method=%(method)s index=%(index)d (%(param1)d,%(param2)d,%(param3)d,%(param4)d,%(exp_factor).2f)'
root_log_id_cluster_template = 'yaca-job cluster method=%(method)s index=%(index)d (%(param1)d,%(param2)d,%(param3)d,%(param4)d,%(exp_factor).2f)'
root_log_id_combine_template = 'yaca-job combine profile_metric=%(profile_metric)s method=%(method)s index=%(index)d (%(param2)d,%(param3)d,%(param4)d,%(exp_factor).2f)'
root_log_id_map_quality_template = 'yaca-job map_quality profile_metric=%(profile_metric)s method=%(method)s index=%(index)d (%(param1)d,%(param2)d,%(param3)d,%(param4)d,%(exp_factor).2f)'
root_log_id_analyse_template = 'yaca-job analyse profile_metric=%(profile_metric)s method=%(method)s index=%(index)d (%(param2)d,%(param3)d,%(param4)d,%(exp_factor).2f)'
root_log_id_analyse2_template = 'yaca-job analyse2 profile_metric=%(profile_metric)s method=%(method)s index=%(index)d (%(param1)d,%(param2)d,%(param3)d,%(param4)d,%(exp_factor).2f)'
root_log_id_analyse3_template = 'yaca-job analyse3 profile_metric=%(profile_metric)s method=%(method)s index=%(index)d (%(param2)d,%(param3)d,%(param4)d,%(exp_factor).2f)'
root_log_id_analyse_map_quality_template = 'yaca-job analyse map quality method=%(method)s index=%(index)d (%(param2)d,%(param3)d,%(param4)d,%(exp_factor).2f)'
job_state_filename_template = 'states/state_job_%(job-id)s.txt'
job_state_template = """yaca-job %(job-id)s has been submitted...
job-type: %(job_type)s
method: %(method)s
index: %(index)s
param1: %(param1)s
param2: %(param2)s
param3: %(param3)s
param4: %(param4)s
exp_factor: %(exp_factor).2f
profile_metric: %(profile_metric)s
"""


get_groups_function_code = None
get_control_groups_function_code = None

similarity_map_clustering_method = [ 'average' ]

try:
    general_config = yaml_container[ 'general_config' ]
    if 'get_groups_function_code' in general_config and general_config[ 'get_groups_function_code' ] != None:
        get_groups_function_code = general_config[ 'get_groups_function_code' ]
    if 'get_control_groups_function_code' in general_config and general_config[ 'get_control_groups_function_code' ] != None:
        get_control_groups_function_code = general_config[ 'get_control_groups_function_code' ]
    if 'similarity_map_clustering_method' in general_config:
        similarity_map_clustering_method = make_clustering_param_list( general_config[ 'similarity_map_clustering_method' ] )
    if 'pdf_map_fontsize' in general_config: pdf_map_fontsize = int( general_config[ 'pdf_map_fontsize' ] )
    if 'pdf_map_label_fontsize' in general_config: pdf_map_label_fontsize = int( general_config[ 'pdf_map_label_fontsize' ] )
    if 'pdf_cluster_label_fontsize' in general_config: pdf_cluster_label_fontsize = int( general_config[ 'pdf_cluster_label_fontsize' ] )
    penetrance_threshold = general_config[ 'penetrance_threshold' ]
    replicate_distance_threshold = general_config[ 'replicate_distance_threshold' ]
    num_of_random_splits = int( general_config[ 'num_of_random_splits' ] )
    size_of_random_splits = float( general_config[ 'size_of_random_splits' ] )
    project_file = general_config[ 'project_file' ]
    path_prefix = general_config[ 'path_prefix' ]
    control_filter_mode = general_config[ 'control_filter_mode' ]
    random_split_file = general_config[ 'random_split_file' ]
    profile_file_template = general_config[ 'profile_file_template' ]
    random_profile_file_template = general_config[ 'random_profile_file_template' ]
    replicate_profile_file_template = general_config[ 'replicate_profile_file_template' ]
    control_profile_file_template = general_config[ 'control_profile_file_template' ]
    similarity_map_file_template = general_config[ 'similarity_map_file_template' ]
    combined_map_file_template = general_config[ 'combined_map_file_template' ]
    map_quality_file_template = general_config[ 'map_quality_file_template' ]
    analyse_plot_file_template = general_config[ 'analyse_plot_file_template' ]
    analyse2_plot_file_template = general_config[ 'analyse2_plot_file_template' ]
    analyse3_plot_file_template = general_config[ 'analyse3_plot_file_template' ]
    analyse_map_quality_plot_file_template = general_config[ 'analyse_map_quality_plot_file_template' ]
    population_plot_file_template = general_config[ 'population_plot_file_template' ]
    selection_plot_file_template = general_config[ 'selection_plot_file_template' ]
    pipeline_state_file_template = general_config[ 'pipeline_state_file_template' ]
except:
    print 'Invalid YACA batch configuration file'
    raise

try:
    clustering_config = yaml_container[ 'clustering_config' ]
    clustering_method_list = make_clustering_param_list( clustering_config[ 'method' ] )
    clustering_index_list = make_clustering_param_list( clustering_config[ 'index' ] )
    clustering_param1_list = make_clustering_param_list( clustering_config[ 'param1' ] )
    clustering_param2_list = make_clustering_param_list( clustering_config[ 'param2' ] )
    clustering_param3_list = make_clustering_param_list( clustering_config[ 'param3' ] )
    clustering_param4_list = make_clustering_param_list( clustering_config[ 'param4' ] )
    clustering_exp_factor_list = make_clustering_param_list( clustering_config[ 'exp-factor' ] )
    clustering_profile_metric_list = make_clustering_param_list( clustering_config[ 'profile_metric' ] )
    clustering_file_template = clustering_config[ 'file_template' ]
except:
    print 'Invalid YACA batch configuration file'
    raise


CONFIG_file_defaults = {
    'general_config' : {
        'project_file' : project_file,
        'path_prefix' : path_prefix,
        'num_of_random_splits' : num_of_random_splits,
        'size_of_random_splits' : size_of_random_splits,
        'control_filter_mode' : control_filter_mode,
        'get_groups_function_code' : None,
        'get_control_groups_function_code' : None,
        'penetrance_threshold' : penetrance_threshold,
        'replicate_distance_threshold' : replicate_distance_threshold,
        'similarity_map_clustering_method' : similarity_map_clustering_method,
        'pdf_map_fontsize' : pdf_map_fontsize,
        'pdf_map_label_fontsize' : pdf_map_label_fontsize,
        'pdf_cluster_label_fontsize' : pdf_cluster_label_fontsize
    },
    'clustering_config' : {
    }
}



clustering_param_sets = []
def recursive_make_clustering_param_set(clustering_param_lists, clustering_param_names, clustering_param_values, clustering_param_sets):
    if len( clustering_param_names ) > 0:
        param_name = clustering_param_names[ 0 ]
        param_list = clustering_param_lists[ param_name ]
        for p in param_list:
            recursive_make_clustering_param_set(
                                clustering_param_lists,
                                clustering_param_names[ 1: ],
                                clustering_param_values + [ ( param_name, p ) ],
                                clustering_param_sets
            )
    else:
        clustering_param_dict = {}
        for n,p in clustering_param_values:
            clustering_param_dict[ n ] = p
        clustering_param_sets.append( clustering_param_dict )
clustering_param_lists = {
            'method' : clustering_method_list,
            'index' : clustering_index_list,
            'param1' : clustering_param1_list,
            'param2' : clustering_param2_list,
            'param3' : clustering_param3_list,
            'param4' : clustering_param4_list,
            'exp_factor' : clustering_exp_factor_list,
            'profile_metric' : clustering_profile_metric_list,
}
recursive_make_clustering_param_set(
                clustering_param_lists,
                clustering_param_lists.keys(),
                [],
                clustering_param_sets
)


def get_param_sets_with(filter, param_sets):
    l = []
    for param_set in param_sets:
        if filter( param_set ):
            l.append( param_set )
    return l

def get_param_set_groups(cmp, param_sets):
    l = []
    ids = range( len( param_sets ) )
    while len( ids ) > 0:
        param_set = param_sets[ ids[ 0 ] ]
        del ids[ 0 ]
        group = [ param_set ]
        remove_ids = []
        for index in ids:
            if cmp( param_set, param_sets[ index ] ):
                group.append( param_sets[ index ] )
                remove_ids.append( index )
        l.append( group )
        for index in remove_ids:
            ids.remove( index )
    return l

"""job_param_sets = {}
for d in clustering_param_set:
    job_index = ( d[ 'method' ], d[ 'index' ], d[ 'param1' ], d[ 'param2' ], d[ 'param3' ], d[ 'param4' ] )
    if job_index not in job_param_sets:
        job_param_sets[ job_index ] = []
    job_param_sets[ job_index ].append( d )"""


def create_path(filename):
    base = os.path.split( filename )[0]
    if not os.path.exists( base ):
        os.makedirs( base )
    if not os.path.isdir( base ):
        raise Exception( 'Not a directory: %s' % base )

PBS_script_template = \
"""#!/bin/bash
#PBS -o %(output_log)s
#PBS -e %(error_log)s
#PBS -l select=ncpus=1:mem=4gb

/g/pepperkok/hepp/yaca3/run_yaca3_cluster.sh --job-id "${YACA_JOB_ID}" --batch --config-file "%(config_file)s" --log-file "%(log_file)s" --log-id "%(log_id)s"
"""

PBS_script_cluster_template = \
"""#!/bin/bash
#PBS -o %(output_log)s
#PBS -e %(error_log)s
#PBS -l select=1:ncpus=1:mem=4gb
#PBS -h

/g/pepperkok/hepp/yaca3/run_yaca3_cluster.sh --job-id "${YACA_JOB_ID}" --batch --config-file "%(config_file)s" --log-file "%(log_file)s" --log-id "%(log_id)s"
"""

def create_PBS_script(filename, job_format_dict):

    d = {}
    d[ 'output_log' ] = os.path.join( path_prefix, output_log_template % job_format_dict )
    d[ 'error_log' ] = os.path.join( path_prefix, error_log_template % job_format_dict )
    d[ 'config_file' ] = os.path.join( path_prefix, config_file_template % job_format_dict )
    d[ 'log_file' ] = os.path.join( path_prefix, root_log_file )
    d[ 'log_id' ] = root_log_id_template % job_format_dict
    d[ 'log_id' ] = d[ 'log_id' ].replace( ' ', '\ ' )
    for f in [ d[ 'log_file' ], d[ 'output_log' ], d[ 'error_log' ], d[ 'config_file' ] ]:
        create_path( f )
    #if job_pid >= 0:
    #    d[ 'depend_job_list' ] = job_pid
    #else:
    #    d[ 'depend_job_list' ] = ''
    PBS_script = PBS_script_template % d
    write_to_file( filename, PBS_script )
    return PBS_script

def create_PBS_cluster_script(filename, job_format_dict):

    d = {}
    d[ 'output_log' ] = os.path.join( path_prefix, output_cluster_log_template % job_format_dict )
    d[ 'error_log' ] = os.path.join( path_prefix, error_cluster_log_template % job_format_dict )
    d[ 'config_file' ] = os.path.join( path_prefix, config_cluster_file_template % job_format_dict )
    d[ 'log_file' ] = os.path.join( path_prefix, root_log_file )
    d[ 'log_id' ] = root_log_id_cluster_template % job_format_dict
    d[ 'log_id' ] = d[ 'log_id' ].replace( ' ', '\ ' )
    for f in [ d[ 'log_file' ], d[ 'output_log' ], d[ 'error_log' ], d[ 'config_file' ] ]:
        create_path( f )
    #if job_pid >= 0:
    #    d[ 'depend_job_list' ] = job_pid
    #else:
    #    d[ 'depend_job_list' ] = ''
    PBS_script = PBS_script_cluster_template % d
    write_to_file( filename, PBS_script )
    return PBS_script

def create_PBS_combine_script(job_format_dict):

    d = {}
    d[ 'output_log' ] = os.path.join( path_prefix, output_combine_log_template % job_format_dict )
    d[ 'error_log' ] = os.path.join( path_prefix, error_combine_log_template % job_format_dict )
    d[ 'config_file' ] = os.path.join( path_prefix, config_combine_file_template % job_format_dict )
    d[ 'log_file' ] = os.path.join( path_prefix, root_log_file )
    d[ 'log_id' ] = root_log_id_combine_template % job_format_dict
    d[ 'log_id' ] = d[ 'log_id' ].replace( ' ', '\ ' )
    for f in [ d[ 'log_file' ], d[ 'output_log' ], d[ 'error_log' ], d[ 'config_file' ] ]:
        create_path( f )
    #if job_pid >= 0:
    #    d[ 'depend_job_list' ] = job_pid
    #else:
    #    d[ 'depend_job_list' ] = ''
    PBS_script = PBS_script_template % d
    return PBS_script

def create_PBS_map_quality_script(filename, job_format_dict):

    d = {}
    d[ 'output_log' ] = os.path.join( path_prefix, output_map_quality_log_template % job_format_dict )
    d[ 'error_log' ] = os.path.join( path_prefix, error_map_quality_log_template % job_format_dict )
    d[ 'config_file' ] = os.path.join( path_prefix, config_map_quality_file_template % job_format_dict )
    d[ 'log_file' ] = os.path.join( path_prefix, root_log_file )
    d[ 'log_id' ] = root_log_id_map_quality_template % job_format_dict
    d[ 'log_id' ] = d[ 'log_id' ].replace( ' ', '\ ' )
    for f in [ d[ 'log_file' ], d[ 'output_log' ], d[ 'error_log' ], d[ 'config_file' ] ]:
        create_path( f )
    #if job_pid >= 0:
    #    d[ 'depend_job_list' ] = job_pid
    #else:
    #    d[ 'depend_job_list' ] = ''
    PBS_script = PBS_script_template % d
    write_to_file( filename, PBS_script )
    return PBS_script

def create_PBS_analyse_script(job_format_dict):

    d = {}
    d[ 'output_log' ] = os.path.join( path_prefix, output_analyse_log_template % job_format_dict )
    d[ 'error_log' ] = os.path.join( path_prefix, error_analyse_log_template % job_format_dict )
    d[ 'config_file' ] = os.path.join( path_prefix, config_analyse_file_template % job_format_dict )
    d[ 'log_file' ] = os.path.join( path_prefix, root_log_file )
    d[ 'log_id' ] = root_log_id_analyse_template % job_format_dict
    d[ 'log_id' ] = d[ 'log_id' ].replace( ' ', '\ ' )
    for f in [ d[ 'log_file' ], d[ 'output_log' ], d[ 'error_log' ], d[ 'config_file' ] ]:
        create_path( f )
    #if job_pid >= 0:
    #    d[ 'depend_job_list' ] = job_pid
    #else:
    #    d[ 'depend_job_list' ] = ''
    PBS_script = PBS_script_template % d
    return PBS_script

def create_PBS_analyse2_script(job_format_dict):

    d = {}
    d[ 'output_log' ] = os.path.join( path_prefix, output_analyse2_log_template % job_format_dict )
    d[ 'error_log' ] = os.path.join( path_prefix, error_analyse2_log_template % job_format_dict )
    d[ 'config_file' ] = os.path.join( path_prefix, config_analyse2_file_template % job_format_dict )
    d[ 'log_file' ] = os.path.join( path_prefix, root_log_file )
    d[ 'log_id' ] = root_log_id_analyse2_template % job_format_dict
    d[ 'log_id' ] = d[ 'log_id' ].replace( ' ', '\ ' )
    for f in [ d[ 'log_file' ], d[ 'output_log' ], d[ 'error_log' ], d[ 'config_file' ] ]:
        create_path( f )
    #if job_pid >= 0:
    #    d[ 'depend_job_list' ] = job_pid
    #else:
    #    d[ 'depend_job_list' ] = ''
    PBS_script = PBS_script_template % d
    return PBS_script

def create_PBS_analyse3_script(job_format_dict):

    d = {}
    d[ 'output_log' ] = os.path.join( path_prefix, output_analyse3_log_template % job_format_dict )
    d[ 'error_log' ] = os.path.join( path_prefix, error_analyse3_log_template % job_format_dict )
    d[ 'config_file' ] = os.path.join( path_prefix, config_analyse3_file_template % job_format_dict )
    d[ 'log_file' ] = os.path.join( path_prefix, root_log_file )
    d[ 'log_id' ] = root_log_id_analyse3_template % job_format_dict
    d[ 'log_id' ] = d[ 'log_id' ].replace( ' ', '\ ' )
    for f in [ d[ 'log_file' ], d[ 'output_log' ], d[ 'error_log' ], d[ 'config_file' ] ]:
        create_path( f )
    #if job_pid >= 0:
    #    d[ 'depend_job_list' ] = job_pid
    #else:
    #    d[ 'depend_job_list' ] = ''
    PBS_script = PBS_script_template % d
    return PBS_script

def create_PBS_analyse_map_quality_script(job_format_dict):

    d = {}
    d[ 'output_log' ] = os.path.join( path_prefix, output_analyse_map_quality_log_template % job_format_dict )
    d[ 'error_log' ] = os.path.join( path_prefix, error_analyse_map_quality_log_template % job_format_dict )
    d[ 'config_file' ] = os.path.join( path_prefix, config_analyse_map_quality_file_template % job_format_dict )
    d[ 'log_file' ] = os.path.join( path_prefix, root_log_file )
    d[ 'log_id' ] = root_log_id_analyse_map_quality_template % job_format_dict
    d[ 'log_id' ] = d[ 'log_id' ].replace( ' ', '\ ' )
    for f in [ d[ 'log_file' ], d[ 'output_log' ], d[ 'error_log' ], d[ 'config_file' ] ]:
        create_path( f )
    #if job_pid >= 0:
    #    d[ 'depend_job_list' ] = job_pid
    #else:
    #    d[ 'depend_job_list' ] = ''
    PBS_script = PBS_script_template % d
    return PBS_script


CONFIG_file_template = \
"""
general_config:
  project_file: %(project_file)s
  num_of_random_splits: %(num_of_random_splits)s
  path_prefix: %(path_prefix)s
  profile_pdf_file_template: %(profile_pdf_file)s
  profile_xls_file_template: %(profile_xls_file)s
  similarity_map_file_template: %(similarity_map_file)s
  combined_map_file_template: %(combined_map_file)s
  clustering_file: %(clustering_file)s
  combine_maps: False
  control_filter_mode: %(control_filter_mode)s
clustering_config:
  method: %(method)s
  index: %(index)s
  param1: %(param1)s
  param2: %(param2)s
  param3: %(param3)s
  param4: %(param4)s
  exp-factor: %(exp_factor)s
  profile_metric: %(profile_metric)s
  file_template: %(clustering_file)s
"""

CONFIG_cluster_file_template = \
"""
general_config:
  project_files: %(project_file)s
  path_prefix: %(path_prefix)s
  profile_pdf_file_template: %(profile_pdf_file)s
  profile_xls_file_template: %(profile_xls_file)s
  similarity_map_file_template: %(similarity_map_file)s
  combined_map_file_template: %(combined_map_file)s
  only_run_clustering: True
  combine_maps: False
  print_population_plot: %(print_population_plot)s
  population_plot_file_template: %(population_plot_file)s
  print_selection_plot: %(print_selection_plot)s
  selection_plot_file_template: %(selection_plot_file)s
  pipeline_state_file_template: %(pipeline_state_file)s
  control_filter_mode: %(control_filter_mode)s
clustering_config:
  method: %(method)s
  index: %(index)s
  param1: %(param1)s
  param2: %(param2)s
  param3: %(param3)s
  param4: %(param4)s
  exp-factor: %(exp_factor)s
  profile_metric: %(profile_metric)s
  file_template: %(clustering_file)s
"""

CONFIG_combine_file_template = \
"""
general_config:
  project_files: %(project_file)s
  path_prefix: %(path_prefix)s
  similarity_map_files: %(similarity_map_files)s
  combined_map_file_template: %(combined_map_file)s
  combine_maps: True
  only_combine_maps: True
  control_filter_mode: %(control_filter_mode)s
clustering_config:
  method: %(method)s
  index: %(index)s
  param1: %(param1)s
  param2: %(param2)s
  param3: %(param3)s
  param4: %(param4)s
  exp-factor: %(exp_factor)s
  profile_metric: %(profile_metric)s
"""

CONFIG_map_quality_file_template = \
"""
general_config:
  path_prefix: %(path_prefix)s
  random_similarity_map_dict: %(random_similarity_map_dict)s
  replicate_similarity_map_dict: %(replicate_similarity_map_dict)s
  map_quality_random_file_template: %(map_quality_random_file)s
  map_quality_replicate_file_template: %(map_quality_replicate_file)s
  map_quality_results_file_template: %(map_quality_results_file)s
  map_quality: True
  only_map_quality: True
  control_filter_mode: %(control_filter_mode)s
clustering_config:
  method: %(method)s
  index: %(index)s
  param1: %(param1)s
  param2: %(param2)s
  param3: %(param3)s
  param4: %(param4)s
  exp-factor: %(exp_factor)s
  profile_metric: %(profile_metric)s
"""

CONFIG_analyse_file_template = \
"""
general_config:
  path_prefix: %(path_prefix)s
  random_combined_map_files: %(random_combined_map_files)s
  replicate_combined_map_files: %(replicate_combined_map_files)s
  analyse_plot_file_template: %(analyse_plot_file)s
  analyse_maps: True
  only_analyse_maps: True
  control_filter_mode: %(control_filter_mode)s
clustering_config:
  method: %(method)s
  index: %(index)s
  param1: %(param1)s
  param2: %(param2)s
  param3: %(param3)s
  param4: %(param4)s
  exp-factor: %(exp_factor)s
  profile_metric: %(profile_metric)s
"""

CONFIG_analyse2_file_template = \
"""
general_config:
  path_prefix: %(path_prefix)s
  random_similarity_map_files: %(random_similarity_map_files)s
  replicate_similarity_map_files: %(replicate_similarity_map_files)s
  analyse2_plot_file_template: %(analyse2_plot_file)s
  analyse2_maps: True
  only_analyse2_maps: True
  control_filter_mode: %(control_filter_mode)s
clustering_config:
  method: %(method)s
  index: %(index)s
  param1: %(param1)s
  param2: %(param2)s
  param3: %(param3)s
  param4: %(param4)s
  exp-factor: %(exp_factor)s
  profile_metric: %(profile_metric)s
"""

CONFIG_analyse3_file_template = \
"""
general_config:
  path_prefix: %(path_prefix)s
  random_similarity_map_dict: %(random_similarity_map_dict)s
  replicate_similarity_map_dict: %(replicate_similarity_map_dict)s
  analyse3_plot_file_template: %(analyse3_plot_file)s
  analyse3_maps: True
  only_analyse3_maps: True
  control_filter_mode: %(control_filter_mode)s
clustering_config:
  method: %(method)s
  index: %(index)s
  param1: %(param1)s
  param2: %(param2)s
  param3: %(param3)s
  param4: %(param4)s
  exp-factor: %(exp_factor)s
  profile_metric: %(profile_metric)s
"""

CONFIG_analyse_map_quality_file_template = \
"""
general_config:
  path_prefix: %(path_prefix)s
  map_quality_files: %(map_quality_files)s
  analyse_map_quality_plot_file_template: %(analyse_map_quality_plot_file)s
  analyse_map_quality: True
  only_analyse_map_quality: True
  control_filter_mode: %(control_filter_mode)s
clustering_config:
  method: %(method)s
  index: %(index)s
  param1: %(param1)s
  param2: %(param2)s
  param3: %(param3)s
  param4: %(param4)s
  exp-factor: %(exp_factor)s
  profile_metric: %(profile_metric)s
"""

def copy_dict_recursive(d):
    d2 = {}
    for key, value in d.iteritems():
        if type( value ) == dict:
            value = copy_dict_recursive( value )
        d2[ key ] = value
    return d2

def create_CONFIG_file(filename, clustering_config, clustering_file):

    #config = CONFIG_file_defaults.copy()

    #config = job_format_dict.copy()
    #d[ 'profile_metric' ] = profile_metric
    #config[ 'control_filter_mode' ] = control_filter_mode
    #config[ 'path_prefix' ] = path_prefix
    #config[ 'project_file' ] = project_file
    #config[ 'clustering_file' ] = clustering_file_template % job_format_dict
    #config[ 'profile_pdf_file_template' ] = profile_pdf_file_template % job_format_dict
    #config[ 'profile_xls_file_template' ] = profile_xls_file_template % job_format_dict
    #config[ 'similarity_map_file_template' ] = similarity_map_file_template % job_format_dict
    #config[ 'combined_map_file' ] = combined_map_file_template % job_format_dict
    #config[ 'num_of_random_splits' ] = num_of_random_splits

    #config = CONFIG_file_defaults.copy()
    #for key,value in d.iteritems():
    #    if value != None:
    #        config[ key ] = value
    #    else:
    #        if key in config:
    #            del config[ key ]
    #return config

    if 'group' not in clustering_config:
        clustering_config[ 'group' ] = '%(group)s'

    config = copy_dict_recursive( CONFIG_file_defaults )

    if get_groups_function_code != None:
        config[ 'general_config' ][ 'get_groups_function_code' ] = get_groups_function_code

    if get_control_groups_function_code != None:
        config[ 'general_config' ][ 'get_control_groups_function_code' ] = get_control_groups_function_code

    for key,value in clustering_config.iteritems():
        config[ 'clustering_config' ][ key ] = value

    config[ 'general_config' ][ 'clustering_file' ] = clustering_file

    config[ 'general_config' ][ 'profile_file_template' ] = profile_file_template
    config[ 'general_config' ][ 'random_profile_file_template' ] = random_profile_file_template
    config[ 'general_config' ][ 'replicate_profile_file_template' ] = replicate_profile_file_template
    config[ 'general_config' ][ 'control_profile_file_template' ] = control_profile_file_template
    config[ 'general_config' ][ 'similarity_map_file_template' ] = similarity_map_file_template
    config[ 'general_config' ][ 'combined_map_file_template' ] = combined_map_file_template
    config[ 'general_config' ][ 'random_split_file' ] = random_split_file
    #config[ 'general_config' ][ 'print_population_plot' ] = print_population_plot
    #config[ 'general_config' ][ 'print_selection_plot' ] = print_selection_plot
    #config[ 'general_config' ][ 'population_plot_file_template' ] = population_plot_file_template % clustering_config
    #config[ 'general_config' ][ 'selection_plot_file_template' ] = selection_plot_file_template % clustering_config

    #d[ 'only_run_clustering' ] = str( only_run_clustering )
    #d[ 'combine_maps' ] = str( combine_maps )
    #d[ 'only_combine_maps' ] = str( only_combine_maps )
    #if not 'clustering_file' in d:
    #    d[ 'clustering_file' ] = clustering_file_template % job_format_dict

    yaml_file = None
    try:
        create_path( filename )
        yaml_file = open( filename, 'w' )

        yaml.dump( config, yaml_file )

    finally:
        if yaml_file:
            yaml_file.close()

    return config
    #CONFIG_file = CONFIG_file_template % d
    #return CONFIG_file

def create_CONFIG_cluster_file(filename, clustering_config, print_population_plot=False, print_selection_plot=False, create_random_splits=False):

    if 'group' not in clustering_config:
        clustering_config[ 'group' ] = '%(group)s'

    config = copy_dict_recursive( CONFIG_file_defaults )

    if get_groups_function_code != None:
        config[ 'general_config' ][ 'get_groups_function_code' ] = get_groups_function_code
    if get_control_groups_function_code != None:
        config[ 'general_config' ][ 'get_control_groups_function_code' ] = get_control_groups_function_code

    for key,value in clustering_config.iteritems():
        config[ 'clustering_config' ][ key ] = value

    config[ 'clustering_config' ][ 'file_template' ] = clustering_file_template

    #config[ 'general_config' ][ 'profile_pdf_file' ] = profile_pdf_file_template % clustering_config
    #config[ 'general_config' ][ 'profile_xls_file' ] = profile_xls_file_template % clustering_config
    #config[ 'general_config' ][ 'similarity_map_file' ] = similarity_map_file_template % clustering_config
    #config[ 'general_config' ][ 'combined_map_file' ] = combined_map_file_template % clustering_config
    config[ 'general_config' ][ 'print_population_plot' ] = str( print_population_plot )
    config[ 'general_config' ][ 'print_selection_plot' ] = str( print_selection_plot )
    config[ 'general_config' ][ 'create_random_splits' ] = str( create_random_splits )
    config[ 'general_config' ][ 'population_plot_file_template' ] = population_plot_file_template % clustering_config
    config[ 'general_config' ][ 'selection_plot_file_template' ] = selection_plot_file_template % clustering_config
    config[ 'general_config' ][ 'pipeline_state_file_template' ] = pipeline_state_file_template % clustering_config
    config[ 'general_config' ][ 'only_run_clustering' ] = True
    config[ 'general_config' ][ 'random_split_file' ] = random_split_file

    #d = job_format_dict.copy()
    #d[ 'profile_metric' ] = profile_metric
    #d[ 'control_filter_mode' ] = control_filter_mode
    #d[ 'path_prefix' ] = path_prefix
    #d[ 'project_file' ] = project_file
    #d[ 'clustering_file' ] = clustering_file_template % job_format_dict
    #d[ 'profile_pdf_file' ] = profile_pdf_file_template % job_format_dict
    #d[ 'profile_xls_file' ] = profile_xls_file_template % job_format_dict
    #d[ 'similarity_map_file' ] = similarity_map_file_template % job_format_dict
    #d[ 'combined_map_file' ] = combined_map_file_template % job_format_dict
    #d[ 'print_population_plot' ] = print_population_plot
    #d[ 'population_plot_file' ] = population_plot_file_template % job_format_dict
    #d[ 'print_selection_plot' ] = print_selection_plot
    #d[ 'selection_plot_file' ] = selection_plot_file_template % job_format_dict
    #d[ 'only_run_clustering' ] = str( only_run_clustering )
    #d[ 'combine_maps' ] = str( combine_maps )
    #d[ 'only_combine_maps' ] = str( only_combine_maps )
    #if not 'clustering_file' in d:
    #    d[ 'clustering_file' ] = clustering_file_template % job_format_dict

    yaml_file = None
    try:
        create_path( filename )
        yaml_file = open( filename, 'w' )

        yaml.dump( config, yaml_file )

    finally:
        if yaml_file:
            yaml_file.close()

    return config
    #CONFIG_file = CONFIG_cluster_file_template % d
    #return CONFIG_file

def create_CONFIG_combine_file(job_format_dict, project_file, similarity_map_files):

    d = job_format_dict.copy()
    #d[ 'profile_metric' ] = profile_metric
    d[ 'control_filter_mode' ] = control_filter_mode
    d[ 'path_prefix' ] = path_prefix
    d[ 'project_file' ] = project_file
    d[ 'similarity_map_files' ] = similarity_map_files
    d[ 'combined_map_file' ] = combined_map_file_template % job_format_dict
    CONFIG_file = CONFIG_combine_file_template % d
    return CONFIG_file

def create_CONFIG_map_quality_file(filename, clustering_config, profile_files):

    config = copy_dict_recursive( CONFIG_file_defaults )

    if get_groups_function_code != None:
        config[ 'general_config' ][ 'get_groups_function_code' ] = get_groups_function_code
    if get_control_groups_function_code != None:
        config[ 'general_config' ][ 'get_control_groups_function_code' ] = get_control_groups_function_code

    for key,value in clustering_config.iteritems():
        config[ 'clustering_config' ][ key ] = value

    config[ 'general_config' ][ 'clustering_file' ] = clustering_file_template

    config[ 'general_config' ][ 'do_map_quality' ] = True
    config[ 'general_config' ][ 'profile_files' ] = profile_files
    config[ 'general_config' ][ 'map_quality_file_template' ] = map_quality_file_template

    yaml_file = None
    try:
        create_path( filename )
        yaml_file = open( filename, 'w' )

        yaml.dump( config, yaml_file )

    finally:
        if yaml_file:
            yaml_file.close()

    return config


def create_CONFIG_analyse_file(job_format_dict, project_file, random_combined_map_files, replicate_combined_map_files):

    d = job_format_dict.copy()
    #d[ 'profile_metric' ] = profile_metric
    d[ 'control_filter_mode' ] = control_filter_mode
    d[ 'path_prefix' ] = path_prefix
    d[ 'random_combined_map_files' ] = random_combined_map_files
    d[ 'replicate_combined_map_files' ] = replicate_combined_map_files
    d[ 'analyse_plot_file' ] = analyse_plot_file_template % job_format_dict
    CONFIG_file = CONFIG_analyse_file_template % d
    return CONFIG_file

def create_CONFIG_analyse2_file(job_format_dict, project_file, random_similarity_map_files, replicate_similarity_map_files):

    d = job_format_dict.copy()
    #d[ 'profile_metric' ] = profile_metric
    d[ 'control_filter_mode' ] = control_filter_mode
    d[ 'path_prefix' ] = path_prefix
    d[ 'random_similarity_map_files' ] = random_similarity_map_files
    d[ 'replicate_similarity_map_files' ] = replicate_similarity_map_files
    d[ 'analyse2_plot_file' ] = analyse2_plot_file_template % job_format_dict
    CONFIG_file = CONFIG_analyse2_file_template % d
    return CONFIG_file

def create_CONFIG_analyse3_file(job_format_dict, project_file, random_similarity_map_dict, replicate_similarity_map_dict):

    d = job_format_dict.copy()
    #d[ 'profile_metric' ] = profile_metric
    d[ 'control_filter_mode' ] = control_filter_mode
    d[ 'path_prefix' ] = path_prefix
    d[ 'random_similarity_map_dict' ] = random_similarity_map_dict
    d[ 'replicate_similarity_map_dict' ] = replicate_similarity_map_dict
    d[ 'analyse3_plot_file' ] = analyse3_plot_file_template % job_format_dict
    CONFIG_file = CONFIG_analyse3_file_template % d
    return CONFIG_file

def create_CONFIG_analyse_map_quality_file(job_format_dict, project_file, map_quality_files):

    d = job_format_dict.copy()
    #d[ 'profile_metric' ] = profile_metric
    d[ 'control_filter_mode' ] = control_filter_mode
    d[ 'path_prefix' ] = path_prefix
    d[ 'map_quality_files' ] = map_quality_files
    d[ 'analyse_map_quality_plot_file' ] = analyse_map_quality_plot_file_template % job_format_dict
    CONFIG_file = CONFIG_analyse_map_quality_file_template % d
    return CONFIG_file


def write_to_file(filename, content):
    create_path( filename )
    f = open( filename, 'w' )
    f.write( content )
    f.close()


def submit_job(PBS_script_filename):

    import subprocess
    p = subprocess.Popen( [ 'ssh', qsub_host, '/usr/pbs/bin/qsub', '-q', 'clng_new', '"%s"' % PBS_script_filename ], stdout=subprocess.PIPE )
    out,err = p.communicate()
    pid = int( out.split( '.' )[0] )
    return pid

def write_job_state_info(filename, format_dict):
    for name in [ 'method', 'index', 'param1', 'param2', 'param3', 'param4', 'exp_factor' ]:
        if name not in format_dict: format_dict[ name ] = '<none>'
    create_path( filename )
    f2 = open( filename, 'w' )
    f2.write( job_state_template % format_dict )
    f2.close()

def submit_jobs(root_jobs, job_list, submit=True):

    #for PBS_script_filename in PBS_script_filenames:
        #f.write( '/usr/pbs/bin/qsub -q clng_new "%s"\n' % PBS_script_filename )
    """job_list = []
    job_stack = []
    job_stack.extend( job_tree )
    print len( job_tree )
    while len( job_stack ) > 0:
        print len( job_list )
        job = job_stack[0]
        del job_stack[0]
        job_list.append( job )
        if job.childs != None:
            job_stack.extend( job.childs )"""

    release_list = []

    import StringIO
    f = StringIO.StringIO()

    done = []
    d = {}
    i = 0
    for job in job_list:
        if job.parents != None:
            ids = []
            for parent in job.parents:
                ids.append( '${JOB_PID_%d}' % d[ parent ] )
            parent_job_str = ':'.join( ids )
        else:
            parent_job_str = ''
        if not job in done:
            f.write( 'echo -n -e "\\rsubmitting job # %d out of %d..."\n' % ( i+1, len( job_list ) ) )
            f.write( 'JOB_PID_%d=`/usr/pbs/bin/qsub -v YACA_JOB_ID=%d -q clng_new -W depend=afterok:%s "%s"`\n' % ( i, i, parent_job_str, job.filename ) )
            job_state_filename = os.path.join( path_prefix, job_state_filename_template % { 'job-id' : i } )
            job.format_dict[ 'job-id' ] = i
            write_job_state_info( job_state_filename, job.format_dict )
            if job in root_jobs:
                release_list.append( i )
            d[ job ] = i
            i += 1
            done.append( job )

    f.write( 'echo && echo "all jobs submitted"\n' )

    for i in xrange( len( release_list ) ):
        j = release_list[ i ]
        f.write( 'echo -n -e "\\rreleasing job # %d  (%d out of %d)..."\n' % ( j+1, i+1, len( release_list ) ) )
        f.write( '/usr/pbs/bin/qrls ${JOB_PID_%d} \n' % ( j ) )

    f.write( 'echo && echo "all jobs released"\n' )

    filename = os.path.join( path_prefix, submit_script_filename )
    write_to_file( filename, f.getvalue() )

    f.close()

    if submit:

        import subprocess
        p = subprocess.Popen( [ 'ssh', qsub_host, '/bin/bash', '"%s"' % filename ], stdout=sys.stdout, stderr=sys.stderr )
        p.wait()

    else:

        print 'created %d jobs' % len( job_list )



import shutil
filename = os.path.join( path_prefix, batch_config_file )
create_path( filename )
shutil.copyfile( config_file, filename)

filename = os.path.join( path_prefix, project_setting_file )
create_path( filename )
shutil.copyfile( project_file, filename )



clustering_file_dict = {}

class PBS_job:
    def __init__(self, filename, format_dict, parents=None):
        self.filename = filename
        self.format_dict = format_dict
        self.parents = parents
        if parents != None:
            if type( self.parents ) != list:
                self.parents = [ self.parents ]
            for parent in self.parents:
                parent.add_child( self )
        self.childs = None
    def add_child(self, child):
        if self.childs == None:
            self.childs = []
        self.childs.append( child )
PBS_jobs = []
PBS_root_jobs = []


job_list = []
def add_job(job_list, job, d, arg=None):
    PBS_jobs.append( job )
    job_list.append( ( job, d.copy(), arg ) )
def get_jobs_with(filter, job_list):
    l1 = []
    l2 = []
    l3 = []
    for job,format_dict,arg in job_list:
        if filter( format_dict ):
            l1.append( job )
            l2.append( arg )
            l3.append( format_dict )
    return l1,l2, l3


print_population_plot = True
print_selection_plot = True
create_random_splits = True


exp_factor = clustering_exp_factor_list[ 0 ]
profile_metric = clustering_profile_metric_list[ 0 ]

filter = lambda x:  x[ 'exp_factor' ] == exp_factor and x[ 'profile_metric' ] == profile_metric

param_sets = get_param_sets_with( filter, clustering_param_sets )

for param_set in param_sets:

    #param_set = param_sets[ 0 ]

    CONFIG_filename = os.path.join( path_prefix, config_cluster_file_template % param_set )
    CONFIG_file = create_CONFIG_cluster_file( CONFIG_filename, param_set, print_population_plot, print_selection_plot, create_random_splits )
    #write_to_file( CONFIG_file_name, CONFIG_file )
    PBS_script_file = os.path.join( path_prefix, pbs_cluster_script_file_template % param_set )
    PBS_script = create_PBS_cluster_script( PBS_script_file, param_set )
    #write_to_file( PBS_script_file, PBS_script )

    if print_population_plot:
        print_population_plot = False
    if print_selection_plot:
        print_selection_plot = False
    if create_random_splits:
        create_random_splits = False

    #job_pid = submit_job( PBS_script_file )
    #print 'submitted job %d' % job_pid

    job_format_dict = param_set.copy()

    job_format_dict[ 'job_type' ] = 'cluster'
    job = PBS_job( PBS_script_file, job_format_dict )
    PBS_root_jobs.append( job )
    add_job( job_list, job, job_format_dict )

    #clustering_file = clustering_file_template % job_format_dict

    def filter2(x):
        for key in x.keys():
            if key not in [ 'exp_factor', 'profile_metric' ] and x[ key ] != param_set[ key ]:
                return False
        return True

    param_sets2 = get_param_sets_with( filter2, clustering_param_sets )

    for param_set2 in param_sets2:

        #job_format_dict = dict( param_set )

        clustering_file = clustering_file_template % job_format_dict

        #job_format_dict[ 'clustering_file' ] = clustering_file_template % job_format_dict

        CONFIG_filename = os.path.join( path_prefix, config_file_template % param_set2 )
        CONFIG_file = create_CONFIG_file( CONFIG_filename, param_set2, clustering_file )
        #write_to_file( CONFIG_file_name, CONFIG_file )
        PBS_script_file = os.path.join( path_prefix, pbs_script_file_template % param_set2 )
        PBS_script = create_PBS_script( PBS_script_file, param_set2 )
        #write_to_file( PBS_script_file, PBS_script )

        #pid = submit_job( PBS_script_file )
        #print 'submitted job %d' % pid

        job_format_dict = param_set2.copy()
        job_format_dict[ 'job_type' ] = 'default'

        child_job = PBS_job( PBS_script_file, job_format_dict, job )
        profile_filename = os.path.join( path_prefix, profile_file_template % job_format_dict ) + '.pic'
        add_job( job_list, child_job, job_format_dict, profile_filename )
        #job.add_child( child_job )



param1 = clustering_param1_list[ 0 ]
exp_factor = clustering_exp_factor_list[ 0 ]
profile_metric = clustering_profile_metric_list[ 0 ]

def param_set_cmp(x,y):
    keys = x.keys()
    keys.remove( 'param1' )
    for key in keys:
        if x[ key ] != y[ key ]:
            return False
    return True

groups = get_param_set_groups( param_set_cmp, clustering_param_sets )

for group in groups:

    param_set = group[ 0 ]

    def param_set_filter(x):
        if x[ 'job_type' ] != 'default':
            return False
        keys = param_set.keys()
        keys.remove( 'param1' )
        for key in keys:
            if x[ key ] != param_set[ key ]:
                return False
        return True
    parent_jobs,profile_files,format_dicts = get_jobs_with( param_set_filter, job_list )

    CONFIG_filename = os.path.join( path_prefix, config_map_quality_file_template % param_set )
    CONFIG_file = create_CONFIG_map_quality_file( CONFIG_filename, param_set, profile_files )
    #write_to_file( CONFIG_file_name, CONFIG_file )
    PBS_script_file = os.path.join( path_prefix, pbs_map_quality_script_file_template % param_set )
    PBS_script = create_PBS_map_quality_script( PBS_script_file, param_set )
    #write_to_file( PBS_script_file, PBS_script )

    #pid = submit_job( PBS_script_file )
    #print 'submitted job %d' % pid

    job_format_dict = param_set.copy()
    job_format_dict[ 'job_type' ] = 'map_quality'

    child_job = PBS_job( PBS_script_file, job_format_dict, parent_jobs )
    add_job( job_list, child_job, job_format_dict )
    #job.add_child( child_job )



"""for method in clustering_method_list:
    for index in clustering_index_list:
        for param2 in clustering_param2_list:
            for param3 in clustering_param3_list:
                for param4 in clustering_param4_list:
                    for exp_factor in clustering_exp_factor_list:
                        for profile_metric in clustering_profile_metric_list:

                            filter1 = lambda x:    x[ 'method' ] == method \
                                               and x[ 'index'  ] == index \
                                               and x[ 'param2' ] == param2 \
                                               and x[ 'param3' ] == param3 \
                                               and x[ 'param4' ] == param4 \
                                               and x[ 'exp_factor'] == exp_factor \
                                               and x[ 'profile_metric' ] == profile_metric
                            filter2 = lambda x:    filter1( x ) \
                                               and x[ 'param1' ] == clustering_param1_list[ 0 ]

                            for param_set in get_param_sets_with( filter2, clustering_param_sets ):

                                combined_map_files = []

                                combine_jobs = []

                                job_format_dict = dict( param_set )

                                job_format_dict[ 'param1' ] = -1

                                filter3 = lambda x:    filter1( x )  \
                                                   and x[ 'job_type' ] == 'default'

                                parent_jobs,sim_maps, format_dicts = get_jobs_with( filter3, job_list )

                                #job_format_dict[ 'param1' ] = clustering_param1_list

                                CONFIG_file = create_CONFIG_combine_file( job_format_dict, project_file, sim_maps )
                                CONFIG_file_name = os.path.join( path_prefix, config_combine_file_template % job_format_dict )
                                write_to_file( CONFIG_file_name, CONFIG_file )
                                PBS_script = create_PBS_combine_script( job_format_dict )
                                PBS_script_file = os.path.join( path_prefix, pbs_combine_script_file_template % job_format_dict )
                                write_to_file( PBS_script_file, PBS_script )

                                #pid = submit_job( PBS_script_file )
                                #print 'submitted job %d' % pid
                                job_format_dict[ 'job_type' ] = 'combine'
                                child_job = PBS_job( PBS_script_file, job_format_dict, parent_jobs )
                                #job.add_child( child_job )
                                combine_jobs.append( child_job )

                                combined_map_filename = os.path.join( path_prefix, combined_map_file_template % job_format_dict )
                                add_job( job_list, child_job, job_format_dict, combined_map_filename )

                                combined_map_files.append( combined_map_filename )

                                CONFIG_file = create_CONFIG_combine_file( job_format_dict, project_file, combined_map_files )
                                CONFIG_file_name = os.path.join( path_prefix, config_combine_file_template % job_format_dict )
                                write_to_file( CONFIG_file_name, CONFIG_file )
                                PBS_script = create_PBS_combine_script( job_format_dict )
                                PBS_script_file = os.path.join( path_prefix, pbs_combine_script_file_template % job_format_dict )
                                write_to_file( PBS_script_file, PBS_script )

                                #pid = submit_job( PBS_script_file )
                                #print 'submitted job %d' % pid
                                job_format_dict[ 'job_type' ] = 'combine_all'
                                child_job = PBS_job( PBS_script_file, job_format_dict, combine_jobs )
                                #job.add_child( child_job )

                                combined_map_filename = os.path.join( path_prefix, combined_map_file_template % job_format_dict )
                                add_job( job_list, child_job, job_format_dict, combined_map_filename )



for method in clustering_method_list:
    for index in clustering_index_list:
        for param2 in clustering_param2_list:
            for param3 in clustering_param3_list:
                for param4 in clustering_param4_list:
                    for exp_factor in clustering_exp_factor_list:
                        for profile_metric in clustering_profile_metric_list:

                            filter1 = lambda x:    x[ 'method' ] == method \
                                               and x[ 'index'  ] == index \
                                               and x[ 'param2' ] == param2 \
                                               and x[ 'param3' ] == param3 \
                                               and x[ 'param4' ] == param4 \
                                               and x[ 'exp_factor'] == exp_factor \
                                               and x[ 'profile_metric' ] == profile_metric

                            filter2 = lambda x: filter1( x ) and x[ 'param1' ] == clustering_param1_list[ 0 ]

                            job_filter = lambda x: x[ 'job_type' ] == 'default'

                            filter_all_jobs = lambda x: filter1( x ) and job_filter( x )

                            #filter_random = lambda x:    filter1( x ) and job_filter( x ) \
                            #                         and x[ 'project_name' ].startswith( 'random' )
                            #filter_replicate = lambda x:    filter1( x ) and job_filter( x ) \
                            #                            and x[ 'project_name' ].startswith( 'replicate' )

                            filter_combine_all = lambda x: filter1( x ) and x[ 'job_type' ] == 'combine_all'

                            for param_set in get_param_sets_with( filter2, clustering_param_sets ):

                                parent_jobs,dump1, dump2 = get_jobs_with( filter_combine_all, job_list )
                                #parent_jobs,similarity_maps, format_dicts = get_jobs_with( filter_all_jobs, job_list )
                                random_similarity_map_dict = {}
                                replicate_similarity_map_dict = {}

                                for param1 in clustering_param1_list:
                                    filter_random2 = lambda x: filter_random( x ) and x[ 'param1' ] == param1
                                    random_parent_jobs,random_similarity_maps, random_format_dicts = get_jobs_with( filter_random2, job_list )
                                    random_similarity_map_dict[ param1 ] = random_similarity_maps
                                    filter_replicate2 = lambda x: filter_replicate( x ) and x[ 'param1' ] == param1
                                    replicate_parent_jobs,replicate_similarity_maps, replicate_format_dicts = get_jobs_with( filter_replicate2, job_list )
                                    replicate_similarity_map_dict[ param1 ] = replicate_similarity_maps

                                job_format_dict = dict( param_set )

                                job_format_dict[ 'param1' ] = -1

                                CONFIG_file = create_CONFIG_map_quality_file( job_format_dict, project_file, random_similarity_map_dict, replicate_similarity_map_dict )
                                CONFIG_file_name = os.path.join( path_prefix, config_map_quality_file_template % job_format_dict )
                                write_to_file( CONFIG_file_name, CONFIG_file )
                                PBS_script = create_PBS_map_quality_script( job_format_dict )
                                PBS_script_file = os.path.join( path_prefix, pbs_map_quality_script_file_template % job_format_dict )
                                write_to_file( PBS_script_file, PBS_script )

                                job_format_dict[ 'job_type' ] = 'map_quality'

                                #pid = submit_job( PBS_script_file )
                                #print 'submitted job %d' % pid
                                child_job = PBS_job( PBS_script_file, job_format_dict, parent_jobs )
                                #job.add_child( child_job )

                                map_quality_filename = os.path.join( path_prefix, map_quality_results_file_template % job_format_dict )
                                add_job( job_list, child_job, job_format_dict, map_quality_filename )



for method in clustering_method_list:
    for index in clustering_index_list:
        for param2 in clustering_param2_list:
            for param3 in clustering_param3_list:
                for param4 in clustering_param4_list:
                    for exp_factor in clustering_exp_factor_list:
                        for profile_metric in clustering_profile_metric_list:

                            filter1 = lambda x:    x[ 'method' ] == method \
                                               and x[ 'index'  ] == index \
                                               and x[ 'param2' ] == param2 \
                                               and x[ 'param3' ] == param3 \
                                               and x[ 'param4' ] == param4 \
                                               and x[ 'exp_factor'] == exp_factor \
                                               and x[ 'profile_metric' ] == profile_metric
                    
                            filter2 = lambda x:    filter1( x ) \
                                               and x[ 'param1' ] == clustering_param1_list[ 0 ]
                            #filter3 = lambda x:    filter1( x ) \
                            #                   and x[ 'project_name' ] == project_files.items()[0][0]

                            job_filter = lambda x: x[ 'job_type' ] == 'combine'

                            filter_all_jobs = lambda x: filter1( x ) and job_filter( x )

                            #filter_random = lambda x:    filter1( x ) and job_filter( x ) \
                            #                         and x[ 'project_name' ].startswith( 'random' )
                            #filter_replicate = lambda x:    filter1( x ) and job_filter( x ) \
                            #                            and x[ 'project_name' ].startswith( 'replicate' )

                            filter_combine_all = lambda x: filter1( x ) and x[ 'job_type' ] == 'combine_all'

                            for param_set in get_param_sets_with( filter2, clustering_param_sets ):

                                parent_jobs, dump1, dump2 = get_jobs_with( filter_combine_all, job_list )
                                #parent_jobs,combined_maps, format_dicts = get_jobs_with( filter_all_jobs, job_list )
                                random_parent_jobs,random_combined_maps, format_dicts = get_jobs_with( filter_random, job_list )
                                replicate_parent_jobs,replicate_combined_maps, format_dicts = get_jobs_with( filter_replicate, job_list )


                                job_format_dict = dict( param_set )

                                CONFIG_file = create_CONFIG_analyse_file( job_format_dict, project_file, random_combined_maps, replicate_combined_maps )
                                CONFIG_file_name = os.path.join( path_prefix, config_analyse_file_template % job_format_dict )
                                write_to_file( CONFIG_file_name, CONFIG_file )
                                PBS_script = create_PBS_analyse_script( job_format_dict )
                                PBS_script_file = os.path.join( path_prefix, pbs_analyse_script_file_template % job_format_dict )
                                write_to_file( PBS_script_file, PBS_script )

                                job_format_dict[ 'job_type' ] = 'analyse'

                                #pid = submit_job( PBS_script_file )
                                #print 'submitted job %d' % pid
                                child_job = PBS_job( PBS_script_file, job_format_dict, parent_jobs )
                                #job.add_child( child_job )

                                add_job( job_list, child_job, job_format_dict )


for method in clustering_method_list:
    for index in clustering_index_list:
        for param1 in clustering_param1_list:
            for param2 in clustering_param2_list:
                for param3 in clustering_param3_list:
                    for param4 in clustering_param4_list:
                        for exp_factor in clustering_exp_factor_list:
                            for profile_metric in clustering_profile_metric_list:

                                filter1 = lambda x:    x[ 'method' ] == method \
                                                   and x[ 'index'  ] == index \
                                                   and x[ 'param2' ] == param2 \
                                                   and x[ 'param3' ] == param3 \
                                                   and x[ 'param4' ] == param4 \
                                                   and x[ 'exp_factor'] == exp_factor \
                                                   and x[ 'param1' ] == param1 \
                                                   and x[ 'profile_metric' ] == profile_metric

                                #filter3 = lambda x:    filter1( x ) \
                                #                   and x[ 'project_name' ] == project_files.items()[0][0]

                                job_filter = lambda x: x[ 'job_type' ] == 'default'

                                filter_all_jobs = lambda x: filter1( x ) and job_filter( x )

                                #filter_random = lambda x:    filter_all_jobs( x ) \
                                #                         and x[ 'project_name' ].startswith( 'random' )
                                #filter_replicate = lambda x:    filter_all_jobs( x ) \
                                #                            and x[ 'project_name' ].startswith( 'replicate' )

                                for param_set in get_param_sets_with( filter1, clustering_param_sets ):

                                    parent_jobs,similarity_maps, format_dicts = get_jobs_with( filter_all_jobs, job_list )
                                    random_parent_jobs,random_similarity_maps, format_dicts = get_jobs_with( filter_random, job_list )
                                    replicate_parent_jobs,replicate_similarity_maps, format_dicts = get_jobs_with( filter_replicate, job_list )

                                    job_format_dict = dict( param_set )

                                    CONFIG_file = create_CONFIG_analyse2_file( job_format_dict, project_file, random_similarity_maps, replicate_similarity_maps )
                                    CONFIG_file_name = os.path.join( path_prefix, config_analyse2_file_template % job_format_dict )
                                    write_to_file( CONFIG_file_name, CONFIG_file )
                                    PBS_script = create_PBS_analyse2_script( job_format_dict )
                                    PBS_script_file = os.path.join( path_prefix, pbs_analyse2_script_file_template % job_format_dict )
                                    write_to_file( PBS_script_file, PBS_script )

                                    job_format_dict[ 'job_type' ] = 'analyse2'
                                    #pid = submit_job( PBS_script_file )
                                    #print 'submitted job %d' % pid
                                    child_job = PBS_job( PBS_script_file, job_format_dict, parent_jobs )
                                    #job.add_child( child_job )

                                    add_job( job_list, child_job, job_format_dict )

analyse3_plot_files = {}

for method in clustering_method_list:
    for index in clustering_index_list:
        for param2 in clustering_param2_list:
            for param3 in clustering_param3_list:
                for param4 in clustering_param4_list:
                    for exp_factor in clustering_exp_factor_list:
                        for profile_metric in clustering_profile_metric_list:

                            filter1 = lambda x:    x[ 'method' ] == method \
                                               and x[ 'index'  ] == index \
                                               and x[ 'param2' ] == param2 \
                                               and x[ 'param3' ] == param3 \
                                               and x[ 'param4' ] == param4 \
                                               and x[ 'exp_factor'] == exp_factor \
                                               and x[ 'profile_metric' ] == profile_metric

                            filter2 = lambda x:    filter1( x ) \
                                               and x[ 'param1' ] == clustering_param1_list[0]

                            job_filter = lambda x: x[ 'job_type' ] == 'default'

                            filter_all_jobs = lambda x: filter1( x ) and job_filter( x )

                            #filter_random = lambda x:    filter_all_jobs( x ) \
                            #                         and x[ 'project_name' ].startswith( 'random' )
                            #filter_replicate = lambda x:    filter_all_jobs( x ) \
                            #                            and x[ 'project_name' ].startswith( 'replicate' )

                            filter_combine_all = lambda x: filter1( x ) and x[ 'job_type' ] == 'combine_all'

                            for param_set in get_param_sets_with( filter2, clustering_param_sets ):

                                parent_jobs, dump1, dump2 = get_jobs_with( filter_combine_all, job_list )
                                #parent_jobs,similarity_maps, format_dicts = get_jobs_with( filter_all_jobs, job_list )
                                random_parent_jobs,random_similarity_maps, random_format_dicts = get_jobs_with( filter_random, job_list )
                                replicate_parent_jobs,replicate_similarity_maps, replicate_format_dicts = get_jobs_with( filter_replicate, job_list )

                                random_similarity_map_dict = {}
                                for i in xrange( len( random_similarity_maps ) ):
                                    param1 = random_format_dicts[ i ][ 'param1' ]
                                    map = random_similarity_maps[ i ]
                                    if param1 not in random_similarity_map_dict:
                                        random_similarity_map_dict[ param1 ] = []
                                    random_similarity_map_dict[ param1 ].append( map )

                                replicate_similarity_map_dict = {}
                                for i in xrange( len( replicate_similarity_maps ) ):
                                    param1 = replicate_format_dicts[ i ][ 'param1' ]
                                    map = replicate_similarity_maps[ i ]
                                    if param1 not in replicate_similarity_map_dict:
                                        replicate_similarity_map_dict[ param1 ] = []
                                    replicate_similarity_map_dict[ param1 ].append( map )

                                job_format_dict = dict( param_set )
                                job_format_dict[ 'treatment' ] = '%(treatment)s'

                                CONFIG_file = create_CONFIG_analyse3_file( job_format_dict, project_file, random_similarity_map_dict, replicate_similarity_map_dict )
                                CONFIG_file_name = os.path.join( path_prefix, config_analyse3_file_template % job_format_dict )
                                write_to_file( CONFIG_file_name, CONFIG_file )
                                PBS_script = create_PBS_analyse3_script( job_format_dict )
                                PBS_script_file = os.path.join( path_prefix, pbs_analyse3_script_file_template % job_format_dict )
                                write_to_file( PBS_script_file, PBS_script )

                                job_format_dict[ 'job_type' ] = 'analyse3'
                                #pid = submit_job( PBS_script_file )
                                #print 'submitted job %d' % pid
                                child_job = PBS_job( PBS_script_file, job_format_dict, parent_jobs )
                                #job.add_child( child_job )

                                add_job( job_list, child_job, job_format_dict )


for method in clustering_method_list:
    for index in clustering_index_list:
        for param2 in clustering_param2_list:
            for param3 in clustering_param3_list:
                for param4 in clustering_param4_list:
                    for exp_factor in clustering_exp_factor_list:

                        filter1 = lambda x:    x[ 'method' ] == method \
                                           and x[ 'index'  ] == index \
                                           and x[ 'param2' ] == param2 \
                                           and x[ 'param3' ] == param3 \
                                           and x[ 'param4' ] == param4 \
                                           and x[ 'exp_factor'] == exp_factor

                        filter2 = lambda x:    filter1( x ) \
                                           and x[ 'param1' ] == clustering_param1_list[0] \
                                           and x[ 'profile_metric' ] == clustering_profile_metric_list[0]

                        job_filter = lambda x: x[ 'job_type' ] == 'default'

                        filter_all_jobs = lambda x: filter1( x ) and job_filter( x )

                        #filter_random = lambda x:    filter_all_jobs( x ) \
                        #                         and x[ 'project_name' ].startswith( 'random' )
                        #filter_replicate = lambda x:    filter_all_jobs( x ) \
                        #                            and x[ 'project_name' ].startswith( 'replicate' )

                        filter_map_quality = lambda x: filter1( x ) and x[ 'job_type' ] == 'map_quality'

                        for param_set in get_param_sets_with( filter2, clustering_param_sets ):

                            parent_jobs, map_quality_files, format_dicts = get_jobs_with( filter_map_quality, job_list )

                            job_format_dict = dict( param_set )

                            CONFIG_file = create_CONFIG_analyse_map_quality_file( job_format_dict, project_file, map_quality_files )
                            CONFIG_file_name = os.path.join( path_prefix, config_analyse_map_quality_file_template % job_format_dict )
                            write_to_file( CONFIG_file_name, CONFIG_file )
                            PBS_script = create_PBS_analyse_map_quality_script( job_format_dict )
                            PBS_script_file = os.path.join( path_prefix, pbs_analyse_map_quality_script_file_template % job_format_dict )
                            write_to_file( PBS_script_file, PBS_script )

                            job_format_dict[ 'job_type' ] = 'analyse_map_quality'
                            #pid = submit_job( PBS_script_file )
                            #print 'submitted job %d' % pid
                            child_job = PBS_job( PBS_script_file, job_format_dict, parent_jobs )
                            #job.add_child( child_job )

                            add_job( job_list, child_job, job_format_dict )"""


submit_jobs( PBS_root_jobs, PBS_jobs, do_submit_jobs )


#import subprocess
#p = subprocess.Popen( [ 'ssh', 'sub-master', '/usr/bin/python', '/g/pepperkok/hepp/cluster_scripts/qwait.py' ], stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr )
#p.wait()

