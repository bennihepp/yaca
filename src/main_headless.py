import sys, os



project_file = None
img_symlink_dir = False

clustering_method = 'k-means'
clustering_index = 0
clustering_param1 = 100
clustering_param2 = -1
clustering_param3 = 2
clustering_param4 = 20

skip_next = 0


def print_help():

    sys.stderr.write( """Usage: python %s [options]
Necessary options:
  --project-file <filename>     Load specified pipeline file
Possible options:
  --clustering-method <method>  Method to be used for clustering,
                                the default is "k-means".
  --clustering-index <index>    Index of the clustering configuration to use,
                                the default is 0.
  --clustering-param1 <param>   Parameter 1 for the clustering procedure,
                                the default is 100.
  --clustering-param2 <param>   Parameter 2 for the clustering procedure,
                                the default is -1.
  --clustering-param3 <param>   Parameter 3 for the clustering procedure.
                                the default is 2.
  --clustering-param4 <param>   Parameter 4 for the clustering procedure.
                                the default is 20.
  --img-symlinks <path>         Create symlinks of the images within <path>,
                                the filename will be the image-ID.
""" % sys.argv[ 0 ] )


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

        if arg == '--img-symlinks':
            img_symlink_dir = next_arg
            skip_next = 1
        elif arg == '--clustering-method':
            clustering_method = next_arg
        elif arg == '--clustering-index':
            clustering_index = int( next_arg )
        elif arg == '--clustering-param1':
            clustering_param1 = int( next_arg )
        elif arg == '--clustering-param2':
            clustering_param2 = int( next_arg )
        elif arg == '--clustering-param3':
            clustering_param3 = int( next_arg )
        elif arg == '--clustering-param4':
            clustering_param4 = int( next_arg )
        elif arg == '--project-file':
            project_file = next_arg
            skip_next = 1
        elif arg == '--help':
            print_help()
            sys.exit( 0 )
        else:
            sys.stderr.write( 'Unknown option: %s\n' % arg )
            print_help()
            sys.exit( -1 )

if project_file == None:
    print 'YACA project file needs to be specified...'
    print_help()
    sys.exit( -1 )



from PyQt4.QtCore import *


from . core import pipeline
from . core import importer
from . core import headless_cluster_configuration

from . core import parameter_utils as utils



headlessClusterConfiguration = headless_cluster_configuration.HeadlessClusterConfiguration()


utils.load_module_configuration( project_file )


modules = utils.list_modules()
for module in modules:

    if not utils.all_parameters_set( module ):
        print 'Not all required parameters for module %s have been set' % module
        sys.exit( -1 )

    elif not utils.all_requirements_met( module ):
        print 'Not all requirements for module %s have been fulfilled' % module
        sys.exit( -1 )


pdc = importer.Importer().get_pdc()
clusterConfiguration = headlessClusterConfiguration.clusterConfiguration


pl = pipeline.Pipeline( pdc, clusterConfiguration )


def callback_pipeline_update_progress(progress):

    sys.stdout.write( '\rprogress: %d %%...' % progress )
    sys.stdout.flush()


pl.connect( pl, SIGNAL('updateProgress'), callback_pipeline_update_progress )
#pl.connect( pl, SIGNAL('finished()'), callback_pipeline_finished )

print '\nRunning quality control...'
pl.start_quality_control()

pl.wait()

sys.stdout.write( '\n' )
sys.stdout.flush()

print '\nRunning pre filtering...'
pl.start_pre_filtering()

pl.wait()

sys.stdout.write( '\n' )
sys.stdout.flush()

pl.disconnect( pl, SIGNAL('updateProgress'), callback_pipeline_update_progress )
#pl.disconnect( pl, SIGNAL('finished()'), callback_pipeline_finished )


#pl.connect( pl, SIGNAL('finished()'), callback_pipeline_finished )

print '\nRunning clustering...'
pl.start_clustering( clustering_method, clustering_index, clustering_param1, clustering_param2, clustering_param3, clustering_param4 )

pl.wait()

#pl.disconnect( pl, SIGNAL('finished()'), callback_pipeline_finished )
