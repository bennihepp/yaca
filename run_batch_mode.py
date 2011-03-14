#!/usr/bin/env python

import yaml
import sys
import os


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


file = None
try:
    file = open( config_file, 'r' )
    yaml_container = yaml.load( file )
finally:
    if file:
        file.close()


batch_config_file = 'batch/batch_config.yaml'
project_setting_file_template = 'batch/yaca_settings_%(project_name)s.yaml'
config_file_template = 'configs/config_%(project_name)s_%(method)s_(%(index)d)_(%(param1)d,%(param2)d,%(param3)d,%(param4)d,%(exp_factor).2f).yaml'
config_cluster_file_template = 'configs/config_cluster_%(method)s_(%(index)d)_(%(param1)d,%(param2)d,%(param3)d,%(param4)d,%(exp_factor).2f).yaml'
config_combine_file_template = 'configs/config_combine_%(project_name)s_%(method)s_(%(index)d)_(%(param2)d,%(param3)d,%(param4)d,%(exp_factor).2f).yaml'
config_combine2_file_template = 'configs/config_combine2_%(method)s_(%(index)d)_(%(param1)d,%(param2)d,%(param3)d,%(param4)d,%(exp_factor).2f).yaml'
config_analyse_file_template = 'configs/config_analyse_%(method)s_(%(index)d)_(%(param2)d,%(param3)d,%(param4)d,%(exp_factor).2f).yaml'
config_analyse2_file_template = 'configs/config_analyse2_%(method)s_(%(index)d)_(%(param1)d,%(param2)d,%(param3)d,%(param4)d,%(exp_factor).2f).yaml'
config_analyse3_file_template = 'configs/config_analyse3_%(method)s_(%(index)d)_(%(param2)d,%(param3)d,%(param4)d,%(exp_factor).2f).yaml'
pbs_script_file_template = 'jobs/job_%(project_name)s_%(method)s_(%(index)d)_(%(param1)d,%(param2)d,%(param3)d,%(param4)d,%(exp_factor).2f).sh'
pbs_cluster_script_file_template = 'jobs/job_cluster_%(method)s_(%(index)d)_(%(param1)d,%(param2)d,%(param3)d,%(param4)d,%(exp_factor).2f).sh'
pbs_combine_script_file_template = 'jobs/job_combine_%(project_name)s_%(method)s_(%(index)d)_(%(param2)d,%(param3)d,%(param4)d,%(exp_factor).2f).sh'
pbs_combine2_script_file_template = 'jobs/job_combine2_%(method)s_(%(index)d)_(%(param1)d,%(param2)d,%(param3)d,%(param4)d,%(exp_factor).2f).sh'
pbs_analyse_script_file_template = 'jobs/job_analyse_%(method)s_(%(index)d)_(%(param2)d,%(param3)d,%(param4)d,%(exp_factor).2f).sh'
pbs_analyse2_script_file_template = 'jobs/job_analyse2_%(method)s_%(index)d)_(%(param1)d,%(param2)d,%(param3)d,%(param4)d,%(exp_factor).2f).sh'
pbs_analyse3_script_file_template = 'jobs/job_analyse3_%(method)s_(%(index)d)_(%(param2)d,%(param3)d,%(param4)d,%(exp_factor).2f).sh'
output_log_template = 'logs/output_%(project_name)s_%(method)s__%(index)d__%(param1)d_%(param2)d_%(param3)d_%(param4)d_%(exp_factor).2f.txt'
output_cluster_log_template = 'logs/output_cluster_%(method)s__%(index)d__%(param1)d_%(param2)d_%(param3)d_%(param4)d_%(exp_factor).2f.txt'
output_combine_log_template = 'logs/output_combine_%(project_name)s_%(method)s__%(index)d__%(param2)d_%(param3)d_%(param4)d_%(exp_factor).2f.txt'
output_combine2_log_template = 'logs/output_combine2_%(method)s__%(index)d__%(param1)d_%(param2)d_%(param3)d_%(param4)d_%(exp_factor).2f.txt'
output_analyse_log_template = 'logs/output_analyse_%(method)s__%(index)d__%(param2)d_%(param3)d_%(param4)d_%(exp_factor).2f.txt'
output_analyse2_log_template = 'logs/output_analyse2_%(method)s__%(index)d__%(param1)d_%(param2)d_%(param3)d_%(param4)d_%(exp_factor).2f.txt'
output_analyse3_log_template = 'logs/output_analyse3_%(method)s__%(index)d__%(param2)d_%(param3)d_%(param4)d_%(exp_factor).2f.txt'
error_log_template = 'logs/error_%(project_name)s_%(method)s__%(index)d__%(param1)d_%(param2)d_%(param3)d_%(param4)d_%(exp_factor).2f.txt'
error_cluster_log_template = 'logs/error_cluster_%(method)s__%(index)d__%(param1)d_%(param2)d_%(param3)d_%(param4)d_%(exp_factor).2f.txt'
error_combine_log_template = 'logs/error_combine_%(project_name)s_%(method)s__%(index)d__%(param2)d_%(param3)d_%(param4)d_%(exp_factor).2f.txt'
error_combine2_log_template = 'logs/error_combine2_%(method)s__%(index)d__%(param1)d_%(param2)d_%(param3)d_%(param4)d_%(exp_factor).2f.txt'
error_analyse_log_template = 'logs/error_analyse_%(method)s__%(index)d__%(param2)d_%(param3)d_%(param4)d_%(exp_factor).2f.txt'
error_analyse2_log_template = 'logs/error_analyse2_%(method)s__%(index)d__%(param1)d_%(param2)d_%(param3)d_%(param4)d_%(exp_factor).2f.txt'
error_analyse3_log_template = 'logs/error_analyse3_%(method)s__%(index)d__%(param2)d_%(param3)d_%(param4)d_%(exp_factor).2f.txt'
submit_script_filename = 'jobs/submit_jobs.sh'
root_log_file = 'error_summary.txt'
root_log_id_template = 'yaca-job %(project_name)s method=%(method)s index=%(index)d (%(param1)d,%(param2)d,%(param3)d,%(param4)d,%(exp_factor).2f)'
root_log_id_cluster_template = 'yaca-job cluster method=%(method)s index=%(index)d (%(param1)d,%(param2)d,%(param3)d,%(param4)d,%(exp_factor).2f)'
root_log_id_combine_template = 'yaca-job combine %(project_name)s method=%(method)s index=%(index)d (%(param2)d,%(param3)d,%(param4)d,%(exp_factor).2f)'
root_log_id_combine2_template = 'yaca-job combine2 method=%(method)s index=%(index)d (%(param1)d,%(param2)d,%(param3)d,%(param4)d,%(exp_factor).2f)'
root_log_id_analyse_template = 'yaca-job analyse method=%(method)s index=%(index)d (%(param2)d,%(param3)d,%(param4)d,%(exp_factor).2f)'
root_log_id_analyse2_template = 'yaca-job analyse2 method=%(method)s index=%(index)d (%(param1)d,%(param2)d,%(param3)d,%(param4)d,%(exp_factor).2f)'
root_log_id_analyse3_template = 'yaca-job analyse3 method=%(method)s index=%(index)d (%(param2)d,%(param3)d,%(param4)d,%(exp_factor).2f)'

try:
    general_config = yaml_container[ 'general_config' ]
    project_files = general_config[ 'project_files' ]
    path_prefix = general_config[ 'path_prefix' ]
    profile_metric = general_config[ 'profile_metric' ]
    profile_pdf_file_template = general_config[ 'profile_pdf_file_template' ]
    profile_xls_file_template = general_config[ 'profile_xls_file_template' ]
    similarity_map_file_template = general_config[ 'similarity_map_file_template' ]
    combined_map_file_template = general_config[ 'combined_map_file_template' ]
    combined2_random_map_file_template = general_config[ 'combined2_random_map_file_template' ]
    combined2_replicate_map_file_template = general_config[ 'combined2_replicate_map_file_template' ]
    combined2_results_file_template = general_config[ 'combined2_results_file_template' ]
    analyse_plot_file_template = general_config[ 'analyse_plot_file_template' ]
    analyse2_plot_file_template = general_config[ 'analyse2_plot_file_template' ]
    analyse3_plot_file_template = general_config[ 'analyse3_plot_file_template' ]
    population_plot_file_template = general_config[ 'population_plot_file_template' ]
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
    clustering_file_template = clustering_config[ 'file_template' ]
except:
    print 'Invalid YACA batch configuration file'
    raise

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
            'exp_factor' : clustering_exp_factor_list
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

/g/pepperkok/hepp/yaca/run_yaca_cluster.sh --batch --config-file "%(config_file)s" --log-file "%(log_file)s" --log-id "%(log_id)s"
"""

PBS_script_cluster_template = \
"""#!/bin/bash
#PBS -o %(output_log)s
#PBS -e %(error_log)s
#PBS -h

/g/pepperkok/hepp/yaca/run_yaca_cluster.sh --batch --config-file "%(config_file)s" --log-file "%(log_file)s" --log-id "%(log_id)s"
"""

def create_PBS_script(job_format_dict):

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
    return PBS_script

def create_PBS_cluster_script(job_format_dict):

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

def create_PBS_combine2_script(job_format_dict):

    d = {}
    d[ 'output_log' ] = os.path.join( path_prefix, output_combine2_log_template % job_format_dict )
    d[ 'error_log' ] = os.path.join( path_prefix, error_combine2_log_template % job_format_dict )
    d[ 'config_file' ] = os.path.join( path_prefix, config_combine2_file_template % job_format_dict )
    d[ 'log_file' ] = os.path.join( path_prefix, root_log_file )
    d[ 'log_id' ] = root_log_id_combine2_template % job_format_dict
    d[ 'log_id' ] = d[ 'log_id' ].replace( ' ', '\ ' )
    for f in [ d[ 'log_file' ], d[ 'output_log' ], d[ 'error_log' ], d[ 'config_file' ] ]:
        create_path( f )
    #if job_pid >= 0:
    #    d[ 'depend_job_list' ] = job_pid
    #else:
    #    d[ 'depend_job_list' ] = ''
    PBS_script = PBS_script_template % d
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

CONFIG_file_template = \
"""
general_config:
  project_files: {
    %(project_name)s: %(project_file)s
  }
  path_prefix: %(path_prefix)s
  profile_pdf_file_template: %(profile_pdf_file)s
  profile_xls_file_template: %(profile_xls_file)s
  similarity_map_file_template: %(similarity_map_file)s
  combined_map_file_template: %(combined_map_file)s
  clustering_file: %(clustering_file)s
  combine_maps: False
  profile_metric: %(profile_metric)s
clustering_config:
  method: %(method)s
  index: %(index)s
  param1: %(param1)s
  param2: %(param2)s
  param3: %(param3)s
  param4: %(param4)s
  exp-factor: %(exp_factor)s
  file_template: %(clustering_file)s
"""

CONFIG_cluster_file_template = \
"""
general_config:
  project_files: {
    %(project_name)s: %(project_file)s
  }
  path_prefix: %(path_prefix)s
  profile_pdf_file_template: %(profile_pdf_file)s
  profile_xls_file_template: %(profile_xls_file)s
  similarity_map_file_template: %(similarity_map_file)s
  combined_map_file_template: %(combined_map_file)s
  only_run_clustering: True
  combine_maps: False
  print_population_plot: %(print_population_plot)s
  population_plot_file_template: %(population_plot_file)s
  profile_metric: %(profile_metric)s
clustering_config:
  method: %(method)s
  index: %(index)s
  param1: %(param1)s
  param2: %(param2)s
  param3: %(param3)s
  param4: %(param4)s
  exp-factor: %(exp_factor)s
  file_template: %(clustering_file)s
"""

CONFIG_combine_file_template = \
"""
general_config:
  project_files: {
    %(project_name)s: %(project_file)s
  }
  path_prefix: %(path_prefix)s
  similarity_map_files: %(similarity_map_files)s
  combined_map_file_template: %(combined_map_file)s
  combine_maps: True
  only_combine_maps: True
  profile_metric: %(profile_metric)s
clustering_config:
  method: %(method)s
  index: %(index)s
  param1: %(param1)s
  param2: %(param2)s
  param3: %(param3)s
  param4: %(param4)s
  exp-factor: %(exp_factor)s
"""

CONFIG_combine2_file_template = \
"""
general_config:
  project_files: {
  }
  path_prefix: %(path_prefix)s
  random_similarity_map_dict: %(random_similarity_map_dict)s
  replicate_similarity_map_dict: %(replicate_similarity_map_dict)s
  combined2_random_map_file_template: %(combined2_random_map_file)s
  combined2_replicate_map_file_template: %(combined2_replicate_map_file)s
  combined2_results_file_template: %(combined2_results_file)s
  combine2_maps: True
  only_combine2_maps: True
  profile_metric: %(profile_metric)s
clustering_config:
  method: %(method)s
  index: %(index)s
  param1: %(param1)s
  param2: %(param2)s
  param3: %(param3)s
  param4: %(param4)s
  exp-factor: %(exp_factor)s
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
  profile_metric: %(profile_metric)s
clustering_config:
  method: %(method)s
  index: %(index)s
  param1: %(param1)s
  param2: %(param2)s
  param3: %(param3)s
  param4: %(param4)s
  exp-factor: %(exp_factor)s
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
  profile_metric: %(profile_metric)s
clustering_config:
  method: %(method)s
  index: %(index)s
  param1: %(param1)s
  param2: %(param2)s
  param3: %(param3)s
  param4: %(param4)s
  exp-factor: %(exp_factor)s
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
  profile_metric: %(profile_metric)s
clustering_config:
  method: %(method)s
  index: %(index)s
  param1: %(param1)s
  param2: %(param2)s
  param3: %(param3)s
  param4: %(param4)s
  exp-factor: %(exp_factor)s
"""


def create_CONFIG_file(job_format_dict, project_file):

    d = job_format_dict.copy()
    d[ 'profile_metric' ] = profile_metric
    d[ 'path_prefix' ] = path_prefix
    d[ 'project_file' ] = project_file
    d[ 'clustering_file' ] = clustering_file_template % job_format_dict
    d[ 'profile_pdf_file' ] = profile_pdf_file_template % job_format_dict
    d[ 'profile_xls_file' ] = profile_xls_file_template % job_format_dict
    d[ 'similarity_map_file' ] = similarity_map_file_template % job_format_dict
    d[ 'combined_map_file' ] = combined_map_file_template % job_format_dict
    #d[ 'only_run_clustering' ] = str( only_run_clustering )
    #d[ 'combine_maps' ] = str( combine_maps )
    #d[ 'only_combine_maps' ] = str( only_combine_maps )
    #if not 'clustering_file' in d:
    #    d[ 'clustering_file' ] = clustering_file_template % job_format_dict
    CONFIG_file = CONFIG_file_template % d
    return CONFIG_file

def create_CONFIG_cluster_file(job_format_dict, project_file, print_population_plot=False):

    d = job_format_dict.copy()
    d[ 'profile_metric' ] = profile_metric
    d[ 'path_prefix' ] = path_prefix
    d[ 'project_file' ] = project_file
    d[ 'clustering_file' ] = clustering_file_template % job_format_dict
    d[ 'profile_pdf_file' ] = profile_pdf_file_template % job_format_dict
    d[ 'profile_xls_file' ] = profile_xls_file_template % job_format_dict
    d[ 'similarity_map_file' ] = similarity_map_file_template % job_format_dict
    d[ 'combined_map_file' ] = combined_map_file_template % job_format_dict
    d[ 'print_population_plot' ] = print_population_plot
    d[ 'population_plot_file' ] = population_plot_file_template % job_format_dict
    #d[ 'only_run_clustering' ] = str( only_run_clustering )
    #d[ 'combine_maps' ] = str( combine_maps )
    #d[ 'only_combine_maps' ] = str( only_combine_maps )
    #if not 'clustering_file' in d:
    #    d[ 'clustering_file' ] = clustering_file_template % job_format_dict
    CONFIG_file = CONFIG_cluster_file_template % d
    return CONFIG_file

def create_CONFIG_combine_file(job_format_dict, project_file, similarity_map_files):

    d = job_format_dict.copy()
    d[ 'profile_metric' ] = profile_metric
    d[ 'path_prefix' ] = path_prefix
    d[ 'project_file' ] = project_file
    d[ 'similarity_map_files' ] = similarity_map_files
    d[ 'combined_map_file' ] = combined_map_file_template % job_format_dict
    CONFIG_file = CONFIG_combine_file_template % d
    return CONFIG_file

def create_CONFIG_combine2_file(job_format_dict, project_file, random_similarity_map_dict, replicate_similarity_map_dict):

    d = job_format_dict.copy()
    d[ 'profile_metric' ] = profile_metric
    d[ 'path_prefix' ] = path_prefix
    d[ 'project_file' ] = project_file
    d[ 'random_similarity_map_dict' ] = random_similarity_map_dict
    d[ 'replicate_similarity_map_dict' ] = replicate_similarity_map_dict
    d[ 'param1' ] = '%(param1)d'
    d[ 'combined2_random_map_file' ] = combined2_random_map_file_template % d
    d[ 'combined2_replicate_map_file' ] = combined2_replicate_map_file_template % d
    d[ 'combined2_results_file' ] = combined2_results_file_template % d
    d[ 'param1' ] = -1
    CONFIG_file = CONFIG_combine2_file_template % d
    return CONFIG_file

def create_CONFIG_analyse_file(job_format_dict, project_file, random_combined_map_files, replicate_combined_map_files):

    d = job_format_dict.copy()
    d[ 'profile_metric' ] = profile_metric
    d[ 'path_prefix' ] = path_prefix
    d[ 'random_combined_map_files' ] = random_combined_map_files
    d[ 'replicate_combined_map_files' ] = replicate_combined_map_files
    d[ 'analyse_plot_file' ] = analyse_plot_file_template % job_format_dict
    CONFIG_file = CONFIG_analyse_file_template % d
    return CONFIG_file

def create_CONFIG_analyse2_file(job_format_dict, project_file, random_similarity_map_files, replicate_similarity_map_files):

    d = job_format_dict.copy()
    d[ 'profile_metric' ] = profile_metric
    d[ 'path_prefix' ] = path_prefix
    d[ 'random_similarity_map_files' ] = random_similarity_map_files
    d[ 'replicate_similarity_map_files' ] = replicate_similarity_map_files
    d[ 'analyse2_plot_file' ] = analyse2_plot_file_template % job_format_dict
    CONFIG_file = CONFIG_analyse2_file_template % d
    return CONFIG_file

def create_CONFIG_analyse3_file(job_format_dict, project_file, random_similarity_map_dict, replicate_similarity_map_dict):

    d = job_format_dict.copy()
    d[ 'profile_metric' ] = profile_metric
    d[ 'path_prefix' ] = path_prefix
    d[ 'random_similarity_map_dict' ] = random_similarity_map_dict
    d[ 'replicate_similarity_map_dict' ] = replicate_similarity_map_dict
    d[ 'analyse3_plot_file' ] = analyse3_plot_file_template % job_format_dict
    CONFIG_file = CONFIG_analyse3_file_template % d
    return CONFIG_file


def write_to_file(filename, content):
    create_path( filename )
    f = open( filename, 'w' )
    f.write( content )
    f.close()


def submit_job(PBS_script_filename):

    import subprocess
    p = subprocess.Popen( [ 'ssh', 'sub-master', '/usr/pbs/bin/qsub', '-q', 'clng_new', '"%s"' % PBS_script_filename ], stdout=subprocess.PIPE )
    out,err = p.communicate()
    pid = int( out.split( '.' )[0] )
    return pid

def submit_jobs(job_tree, submit=True):

    import StringIO
    f = StringIO.StringIO()

    #for PBS_script_filename in PBS_script_filenames:
        #f.write( '/usr/pbs/bin/qsub -q clng_new "%s"\n' % PBS_script_filename )
    job_list = []
    job_stack = []
    job_stack.extend( job_tree )
    while len( job_stack ) > 0:
        job = job_stack[0]
        del job_stack[0]
        job_list.append( job )
        if job.childs != None:
            job_stack.extend( job.childs )

    release_list = []

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
            f.write( 'echo -n -e "\\rsubmitting job # %d..."\n' % i )
            f.write( 'JOB_PID_%d=`/usr/pbs/bin/qsub -q clng_new -W depend=afterok:%s "%s"`\n' % ( i, parent_job_str, job.filename ) )
            if job in job_tree:
                release_list.append( i )
            d[ job ] = i
            i += 1
            done.append( job )

    f.write( 'echo && echo "all jobs submitted"\n' )

    for i in release_list:
        f.write( 'echo -n -e "\\rreleasing job # %d..."\n' % i )
        f.write( '/usr/pbs/bin/qrls ${JOB_PID_%d} \n' % ( i ) )

    f.write( 'echo && echo "all jobs released"\n' )

    filename = os.path.join( path_prefix, submit_script_filename )
    write_to_file( filename, f.getvalue() )

    f.close()

    if submit:

        import subprocess
        p = subprocess.Popen( [ 'ssh', 'sub-master', '/bin/bash', '"%s"' % filename ], stdout=sys.stdout, stderr=sys.stderr )
        p.wait()



import shutil
filename = os.path.join( path_prefix, batch_config_file )
create_path( filename )
shutil.copyfile( config_file, filename)

for project_name,project_file in project_files.iteritems():
    d = { 'project_name' : project_name }
    filename = os.path.join( path_prefix, project_setting_file_template % d )
    create_path( filename )
    shutil.copyfile( project_file, filename )



clustering_file_dict = {}

class PBS_job:
    def __init__(self, filename, parents=None):
        self.filename = filename
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


job_list = []
def add_job(job_list, job, d, arg=None):
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

for method in clustering_method_list:
    for index in clustering_index_list:
        for param1 in clustering_param1_list:
            for param2 in clustering_param2_list:
                for param3 in clustering_param3_list:
                    for param4 in clustering_param4_list:

                        filter = lambda x:     x[ 'param1' ] == param1 \
                                                          and x[ 'method' ] == method \
                                                          and x[ 'index'  ] == index \
                                                          and x[ 'param2' ] == param2 \
                                                          and x[ 'param3' ] == param3 \
                                                          and x[ 'param4' ] == param4

                        #print_population_plot = param1 == clustering_param1_list[ 0 ]

                        param_sets = get_param_sets_with( filter, clustering_param_sets )

                        param_set = param_sets[0]
                        job_format_dict = dict( param_sets[0] )
                        job_format_dict[ 'project_name' ] = 'ALL'

                        project_name,project_file = project_files.items()[0]
                        CONFIG_file = create_CONFIG_cluster_file( job_format_dict, project_file, print_population_plot )
                        CONFIG_file_name = os.path.join( path_prefix, config_cluster_file_template % job_format_dict )
                        write_to_file( CONFIG_file_name, CONFIG_file )
                        PBS_script = create_PBS_cluster_script( job_format_dict )
                        PBS_script_file = os.path.join( path_prefix, pbs_cluster_script_file_template % job_format_dict )
                        write_to_file( PBS_script_file, PBS_script )

                        if print_population_plot:
                            print_population_plot = False

                        #job_pid = submit_job( PBS_script_file )
                        #print 'submitted job %d' % job_pid
                        job = PBS_job( PBS_script_file )
                        PBS_jobs.append( job )
                        job_format_dict[ 'job_type' ] = 'cluster'
                        add_job( job_list, job, job_format_dict )

                        clustering_file = clustering_file_template % job_format_dict

                        for project_name,project_file in project_files.iteritems():

                            for param_set in param_sets:

                                job_format_dict = dict( param_set )
                                job_format_dict[ 'project_name' ] = project_name

                                job_format_dict[ 'clustering_file' ] = clustering_file_template % job_format_dict

                                CONFIG_file = create_CONFIG_file( job_format_dict, project_file )
                                CONFIG_file_name = os.path.join( path_prefix, config_file_template % job_format_dict )
                                write_to_file( CONFIG_file_name, CONFIG_file )
                                PBS_script = create_PBS_script( job_format_dict )
                                PBS_script_file = os.path.join( path_prefix, pbs_script_file_template % job_format_dict )
                                write_to_file( PBS_script_file, PBS_script )

                                #pid = submit_job( PBS_script_file )
                                #print 'submitted job %d' % pid
                                child_job = PBS_job( PBS_script_file, job )
                                similarity_map_filename = os.path.join( path_prefix, similarity_map_file_template % job_format_dict )
                                job_format_dict[ 'job_type' ] = 'default'
                                add_job( job_list, child_job, job_format_dict, similarity_map_filename )
                                #job.add_child( child_job )

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
                                           and x[ 'param1' ] == clustering_param1_list[ 0 ]

                        for param_set in get_param_sets_with( filter2, clustering_param_sets ):

                            combined_map_files = []

                            combine_jobs = []

                            job_format_dict = dict( param_set )

                            job_format_dict[ 'param1' ] = -1

                            for project_name,project_file in project_files.iteritems():

                                filter3 = lambda x:    filter1( x ) and x[ 'project_name' ] == project_name \
                                                   and x[ 'job_type' ] == 'default'

                                parent_jobs,sim_maps, format_dicts = get_jobs_with( filter3, job_list )

                                job_format_dict[ 'project_name' ] = project_name
                                #job_format_dict[ 'param1' ] = clustering_param1_list

                                CONFIG_file = create_CONFIG_combine_file( job_format_dict, project_file, sim_maps )
                                CONFIG_file_name = os.path.join( path_prefix, config_combine_file_template % job_format_dict )
                                write_to_file( CONFIG_file_name, CONFIG_file )
                                PBS_script = create_PBS_combine_script( job_format_dict )
                                PBS_script_file = os.path.join( path_prefix, pbs_combine_script_file_template % job_format_dict )
                                write_to_file( PBS_script_file, PBS_script )

                                #pid = submit_job( PBS_script_file )
                                #print 'submitted job %d' % pid
                                child_job = PBS_job( PBS_script_file, parent_jobs )
                                #job.add_child( child_job )
                                combine_jobs.append( child_job )

                                combined_map_filename = os.path.join( path_prefix, combined_map_file_template % job_format_dict )
                                job_format_dict[ 'job_type' ] = 'combine'
                                add_job( job_list, child_job, job_format_dict, combined_map_filename )

                                combined_map_files.append( combined_map_filename )

                            job_format_dict[ 'project_name' ] = 'ALL'

                            CONFIG_file = create_CONFIG_combine_file( job_format_dict, project_file, combined_map_files )
                            CONFIG_file_name = os.path.join( path_prefix, config_combine_file_template % job_format_dict )
                            write_to_file( CONFIG_file_name, CONFIG_file )
                            PBS_script = create_PBS_combine_script( job_format_dict )
                            PBS_script_file = os.path.join( path_prefix, pbs_combine_script_file_template % job_format_dict )
                            write_to_file( PBS_script_file, PBS_script )

                            #pid = submit_job( PBS_script_file )
                            #print 'submitted job %d' % pid
                            child_job = PBS_job( PBS_script_file, combine_jobs )
                            #job.add_child( child_job )

                            combined_map_filename = os.path.join( path_prefix, combined_map_file_template % job_format_dict )
                            job_format_dict[ 'job_type' ] = 'combine_all'
                            add_job( job_list, child_job, job_format_dict, combined_map_filename )



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

                        filter2 = lambda x: filter1( x ) and x[ 'param1' ] == clustering_param1_list[ 0 ]

                        job_filter = lambda x: x[ 'job_type' ] == 'default'

                        filter_all_jobs = lambda x: filter1( x ) and job_filter( x )

                        filter_random = lambda x:    filter1( x ) and job_filter( x ) \
                                                 and x[ 'project_name' ].startswith( 'random' )
                        filter_replicate = lambda x:    filter1( x ) and job_filter( x ) \
                                                    and x[ 'project_name' ].startswith( 'replicate' )

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

                            CONFIG_file = create_CONFIG_combine2_file( job_format_dict, project_file, random_similarity_map_dict, replicate_similarity_map_dict )
                            CONFIG_file_name = os.path.join( path_prefix, config_combine2_file_template % job_format_dict )
                            write_to_file( CONFIG_file_name, CONFIG_file )
                            PBS_script = create_PBS_combine2_script( job_format_dict )
                            PBS_script_file = os.path.join( path_prefix, pbs_combine2_script_file_template % job_format_dict )
                            write_to_file( PBS_script_file, PBS_script )

                            #pid = submit_job( PBS_script_file )
                            #print 'submitted job %d' % pid
                            child_job = PBS_job( PBS_script_file, parent_jobs )
                            #job.add_child( child_job )

                            #combined2_map_filename = os.path.join( path_prefix, combined2_map_file_template % job_format_dict )
                            job_format_dict[ 'job_type' ] = 'combine2'
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
                                           and x[ 'param1' ] == clustering_param1_list[ 0 ]
                        #filter3 = lambda x:    filter1( x ) \
                        #                   and x[ 'project_name' ] == project_files.items()[0][0]

                        job_filter = lambda x: x[ 'job_type' ] == 'combine'

                        filter_all_jobs = lambda x: filter1( x ) and job_filter( x )

                        filter_random = lambda x:    filter1( x ) and job_filter( x ) \
                                                 and x[ 'project_name' ].startswith( 'random' )
                        filter_replicate = lambda x:    filter1( x ) and job_filter( x ) \
                                                    and x[ 'project_name' ].startswith( 'replicate' )

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

                            #pid = submit_job( PBS_script_file )
                            #print 'submitted job %d' % pid
                            child_job = PBS_job( PBS_script_file, parent_jobs )
                            #job.add_child( child_job )

                            job_format_dict[ 'job_type' ] = 'analyse'
                            add_job( job_list, child_job, job_format_dict )


for method in clustering_method_list:
    for index in clustering_index_list:
        for param1 in clustering_param1_list:
            for param2 in clustering_param2_list:
                for param3 in clustering_param3_list:
                    for param4 in clustering_param4_list:
                        for exp_factor in clustering_exp_factor_list:

                            filter1 = lambda x:    x[ 'method' ] == method \
                                               and x[ 'index'  ] == index \
                                               and x[ 'param2' ] == param2 \
                                               and x[ 'param3' ] == param3 \
                                               and x[ 'param4' ] == param4 \
                                               and x[ 'exp_factor'] == exp_factor \
                                               and x[ 'param1' ] == param1

                            #filter3 = lambda x:    filter1( x ) \
                            #                   and x[ 'project_name' ] == project_files.items()[0][0]

                            job_filter = lambda x: x[ 'job_type' ] == 'default'

                            filter_all_jobs = lambda x: filter1( x ) and job_filter( x )

                            filter_random = lambda x:    filter_all_jobs( x ) \
                                                     and x[ 'project_name' ].startswith( 'random' )
                            filter_replicate = lambda x:    filter_all_jobs( x ) \
                                                        and x[ 'project_name' ].startswith( 'replicate' )

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

                                #pid = submit_job( PBS_script_file )
                                #print 'submitted job %d' % pid
                                child_job = PBS_job( PBS_script_file, parent_jobs )
                                #job.add_child( child_job )

                                job_format_dict[ 'job_type' ] = 'analyse2'
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
                                           and x[ 'param1' ] == clustering_param1_list[0]

                        job_filter = lambda x: x[ 'job_type' ] == 'default'

                        filter_all_jobs = lambda x: filter1( x ) and job_filter( x )

                        filter_random = lambda x:    filter_all_jobs( x ) \
                                                 and x[ 'project_name' ].startswith( 'random' )
                        filter_replicate = lambda x:    filter_all_jobs( x ) \
                                                    and x[ 'project_name' ].startswith( 'replicate' )

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

                            #pid = submit_job( PBS_script_file )
                            #print 'submitted job %d' % pid
                            child_job = PBS_job( PBS_script_file, parent_jobs )
                            #job.add_child( child_job )

                            job_format_dict[ 'job_type' ] = 'analyse3'
                            add_job( job_list, child_job, job_format_dict )


submit_jobs( PBS_jobs, do_submit_jobs )


#import subprocess
#p = subprocess.Popen( [ 'ssh', 'sub-master', '/usr/bin/python', '/g/pepperkok/hepp/cluster_scripts/qwait.py' ], stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr )
#p.wait()

