# -*- coding: utf-8 -*-

"""
importer.py -- Importing data files.

- set_state() is used by parameter_utils.py
- import_data() imports data from CellProfiler 2.0 generated CSV files.
- load_hdf5() loads an HDF5 file containing data.
- save_hdf5() saves data to an HDF5 file containing data.
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
import analyse
import quality_control


class Importer(object):
    __inst = None

    def __new__(cls, *args, **kwargs):
        if cls.__inst is not None:
            return cls.__inst
        cls.__inst = object.__new__(cls)
        cls.__init(cls.__inst, *args, **kwargs)
        return cls.__inst

    def __init(self, pdc=None):
        self.pdc = pdc

    def set_pdc(self, pdc):
        self.pdc = pdc

    def get_pdc(self):
        return self.pdc


def set_state(state):

    if state == 'imported':
        try:
            load_hdf5()
        except Exception, e:
            print 'Unable to import HDF5 file %s: %s' % (hdf5_input_file, str(e))
            raise


def load_hdf5():

    try:
        hdf5_input_file
    except:
        raise Exception('You need to specify the YACA HDF5 input file!')

    further_hdf5_input_files = []
    try:
        further_hdf5_input_files = optional_hdf5_input_files
    except:
        pass

    pdc = import_hdf5_results(hdf5_input_file, further_hdf5_input_files)

    Importer().set_pdc(pdc)

    utils.update_state(__name__, 'imported')

    return 'Finished loading HDF5 file'

def normalize_intensities():

    pdc = Importer().get_pdc()

    for img in pdc.images:
        img.state = img.import_state
    for obj in pdc.objects:
        obj.state = obj.import_state

    validImageMask, validCellMask = quality_control.quality_control(pdc)

    analyse.normalize_to_control_cell_intensity(pdc, validCellMask)

def save_hdf5():

    pdc = Importer().get_pdc()

    try:
        hdf5_output_file
    except:
        raise Exception('You need to specify the YACA HDF5 output file!')

    try:
        pdc
    except:
        raise Exception('You need to import data first!')

    export_hdf5_results(hdf5_output_file, pdc)

    return 'Finished saving HDF5 file'


def filename_hook(module, param_name, param_value, yaml_filename):
    if yaml_filename is None:
        return param_value
    if type(param_value) == str:
        values = [param_value]
    else:
        values = param_value
    for i, value in enumerate(values):
        if not os.path.isabs(value):
            values[i] = os.path.join(os.path.dirname(yaml_filename), value)
    if type(param_value) == str:
        return values[0]
    else:
        return values


__dict__ = sys.modules[__name__].__dict__

utils.register_module(__name__, 'Data import', __dict__, utils.DEFAULT_STATE)

utils.register_parameter(__name__, 'hdf5_input_file', utils.PARAM_INPUT_FILE, 'YACA HDF5 input file', optional=True)
utils.set_parameter_hook(__name__, 'hdf5_input_file', filename_hook)
utils.register_parameter(__name__, 'optional_hdf5_input_files', utils.PARAM_INPUT_FILES, 'Further YACA HDF5 input files', optional=True)
utils.set_parameter_hook(__name__, 'optional_hdf5_input_files', filename_hook)

utils.register_parameter(__name__, 'hdf5_output_file', utils.PARAM_OUTPUT_FILE, 'YACA HDF5 output file', optional=True)
utils.set_parameter_hook(__name__, 'hdf5_output_file', filename_hook)

utils.register_action(__name__, 'load_hdf5', 'Load data from a YACA HDF5 file', load_hdf5)

utils.register_action(__name__, 'save_hdf5', 'Save data as YACA HDF5 file', save_hdf5)

utils.register_action(__name__, 'normalize_intensities', 'Normalize intensity features to mean control cell intensity', normalize_intensities)

utils.set_module_state_callback(__name__, set_state)
