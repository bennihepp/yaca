import sys
import os


from ioutil.import_cp2_csv import import_cp2_csv_results
from ioutil.import_export_hdf5 import *


import parameter_utils as utils



pdc = None


class Importer(object):

    def get_pdc(self):
        global pdc
        return pdc



def set_state( state ):

    global pdc

    if state == 'imported':

        imported = False

        use_hdf5 = False
        try:
            hdf5_input_file
            use_hdf5 = True
        except:
            pass

        if use_hdf5:
            try:
                load_hdf5()
                imported = True
            except Exception, e:
                raise
                print 'Unable to import HDF5 file %s: %s' % ( hdf5_input_file, str( e ) )

        if not imported:
            try:
                image_cp2_file
                object_cp2_csv_files
            except:
                raise Exception( "Couldn't recover saved state!" )

            import_data()


def import_data():

    global pdc

    # import data from files

    """# see if results are already available in native format and are up to date
    max_results_file_mtime = 0
    tmp = [ ( 'dummy', file_images ) ]
    tmp.extend(files_objects)
    for prefix,file in tmp:
        if os.path.getmtime(file) > max_results_file_mtime:
            max_results_file_mtime = os.path.getmtime(file)
    
    if os.path.isfile(hybrid_file) and ( os.path.getmtime(hybrid_file) > max_results_file_mtime ):
        pdc = import_hybrid_results(hybrid_file)
    else:
        pdc = import_cp2_csv_results(file_images, files_objects, delimiter)
        export_hybrid_results(hybrid_file, pdc)"""

    try:
        image_cp2_file
    except:
        raise Exception( 'You need to specify the CellProfiler 2 image CSV file!' )
    try:
        object_cp2_csv_files
    except:
        raise Exception( 'You need to specify the CellProfiler 2 object CSV files!' )

    image_file_postfix = '_' + os.path.splitext( os.path.basename( image_cp2_file ) )[0].split('_')[ -1 ]
    object_file_postfixes = []
    for object_file in object_cp2_csv_files:
        object_file = str( object_file )
        object_file_postfix = '_' + os.path.splitext( os.path.basename( object_file ) )[0].split('_')[ -1 ]
        object_file_postfixes.append( object_file_postfix )

    pdc = import_cp2_csv_results( cp2_csv_path, image_file_postfix, object_file_postfixes, csv_delimiter, csv_extension )

    utils.update_state( __name__, 'imported' )

    return 'Finished importing data'


def load_hdf5():

    global pdc

    try:
        hdf5_input_file
    except:
        raise Exception( 'You need to specify the YACA HDF5 input file!' )

    pdc = import_hdf5_results( hdf5_input_file )

    utils.update_state( __name__, 'imported' )

    return 'Finished loading HDF5 file'


def save_hdf5():

    global pdc

    try:
        hdf5_output_file
    except:
        raise Exception( 'You need to specify the YACA HDF5 output file!' )

    try:
        pdc
    except:
        raise Exception( 'You need to import data first!' )

    export_hdf5_results( hdf5_output_file, pdc )

    return 'Finished saving HDF5 file'



__dict__ = sys.modules[ __name__ ].__dict__

utils.register_module( __name__, 'Data importer', __dict__, utils.DEFAULT_STATE )

utils.register_parameter( __name__, 'image_cp2_file', utils.PARAM_INPUT_FILE, 'Image CSV file from CellProfiler 2', optional=True )

utils.register_parameter( __name__, 'object_cp2_csv_files', utils.PARAM_INPUT_FILES, 'Object CSV files from CellProfiler 2', optional=True )

utils.register_parameter( __name__, 'cp2_csv_path', utils.PARAM_PATH, 'Path to CellProfiler 2 CSV files', optional=True )

utils.register_parameter( __name__, 'csv_delimiter', utils.PARAM_STR, 'Delimiter for the CSV files', ',' )

utils.register_parameter( __name__, 'csv_extension', utils.PARAM_STR, 'Extension for the CSV files', '.csv' )

utils.register_parameter( __name__, 'hdf5_input_file', utils.PARAM_INPUT_FILE, 'YACA HDF5 input file', optional=True )

utils.register_parameter( __name__, 'hdf5_output_file', utils.PARAM_OUTPUT_FILE, 'YACA HDF5 output file', optional=True )

utils.register_action( __name__, 'import', 'Import data', import_data )

utils.register_action( __name__, 'load_hdf5', 'Load data from a YACA HDF5 file', load_hdf5 )

utils.register_action( __name__, 'save_hdf5', 'Save data as YACA HDF5 file', save_hdf5 )

utils.set_module_state_callback( __name__, set_state )

