# -*- coding: utf-8 -*-

"""
cp_importer.py -- Importing data from CellProfiler.

- set_state() is used by parameter_utils.py
- import_data_from_cp() imports data from CellProfiler 2.0 generated CSV files.
"""

# This software is distributed under the FreeBSD License.
# See the accompanying file LICENSE for details.
# 
# Copyright 2011 Benjamin Hepp

import sys
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from ioutil.import_cp2_csv import import_cp2_csv_results
from ioutil.import_export_hdf5 import *

import parameter_utils as utils
import importer
import analyse
import quality_control


def import_data_from_cp():

    # import data from files

    try:
        image_cp2_file
    except:
        raise Exception('You need to specify the CellProfiler 2 image CSV file!')
    try:
        object_cp2_csv_files
    except:
        raise Exception('You need to specify the CellProfiler 2 object CSV files!')

    image_file_postfix = '_' + os.path.splitext(os.path.basename(image_cp2_file))[0].split('_')[-1]
    object_file_postfixes = []
    for object_file in object_cp2_csv_files:
        object_file = str(object_file)
        object_file_postfix = '_' + os.path.splitext(os.path.basename(object_file))[0].split('_')[-1]
        object_file_postfixes.append(object_file_postfix)

    pdc = import_cp2_csv_results(cp2_csv_path, image_file_postfix, object_file_postfixes, csv_delimiter, csv_extension)

    Importer().set_pdc(pdc)

    utils.update_state(importer.__name__, 'imported')

    print 'Finished importing data from CellProfiler'
    return 'Finished importing data from CellProfiler'


__dict__ = sys.modules[__name__].__dict__

utils.register_module(__name__, 'CellProfiler import', __dict__, utils.DEFAULT_STATE)

utils.register_parameter(__name__, 'image_cp2_file', utils.PARAM_INPUT_FILE, 'Image CSV file from CellProfiler 2', optional=True)
utils.set_parameter_hook(__name__, 'image_cp2_file', importer.filename_hook)

utils.register_parameter(__name__, 'object_cp2_csv_files', utils.PARAM_INPUT_FILES, 'Object CSV files from CellProfiler 2', optional=True)
utils.set_parameter_hook(__name__, 'object_cp2_csv_files', importer.filename_hook)

utils.register_parameter(__name__, 'cp2_csv_path', utils.PARAM_PATH, 'Path to CellProfiler 2 CSV files', optional=True)
utils.set_parameter_hook(__name__, 'cp2_csv_path', importer.filename_hook)

utils.register_parameter(__name__, 'csv_delimiter', utils.PARAM_STR, 'Delimiter for the CSV files', ',')

utils.register_parameter(__name__, 'csv_extension', utils.PARAM_STR, 'Extension for the CSV files', '.csv')

utils.register_action(__name__, 'import_cp', 'Import data from CellProfiler CSV files', import_data_from_cp)
