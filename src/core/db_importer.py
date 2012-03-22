# -*- coding: utf-8 -*-

"""
db_importer.py -- Importing data from a SQL database.

- set_state() is used by parameter_utils.py.
- connect_to_database() connects to a SQL database.
- reload_db_schema() reloads the table schemas from a SQL database.
- import_data_from_db() imports data from a SQL database.
"""

# This software is distributed under the FreeBSD License.
# See the accompanying file LICENSE for details.
#
# Copyright 2011 Benjamin Hepp

import sys
import os
import re
import logging

import numpy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from ioutil.data_container import yaca_data_container, yaca_data_image, \
     yaca_data_object, yaca_data_plate, yaca_data_well, yaca_data_replicate, \
     yaca_data_treatment

import parameter_utils as utils
import importer

from ioutil.import_cp2_csv import IMAGE_FILENAME_IDENTIFIER, \
     IMAGE_PATHNAME_IDENTIFIER
from ioutil.data_container import OBJECT_ID_FEATURE_NAME, \
     WELL_ID_FEATURE_NAME, TREATMENT_ID_FEATURE_NAME, \
     REPLICATE_ID_FEATURE_NAME, IMAGE_ID_FEATURE_NAME, \
     QUALITY_CONTROL_FEATURE_NAME, PLATE_ID_FEATURE_NAME
from quality_control import QUALITY_CONTROL_DEFAULT


database_url = None
use_plate_filter = None
use_well_filter = None
use_replicate_filter = None
use_position_filter = None
use_treatment_filter = None
plate_filter = None
well_filter = None
replicate_filter = None
position_filter = None
treatment_filter_file = None
image_id_db_column = None
object_img_id_db_column = None
position_x_db_column = None
position_y_db_column = None
plate_db_column = None
well_db_column = None
replicate_db_column = None
position_db_column = None
treatment_db_column = None
images_db_columns = None
objects_db_columns = None
image_files_db_columns = None


class DBConnection(object):
    __inst = None

    def __new__(cls, *args, **kwargs):
        if cls.__inst is not None:
            return cls.__inst
        cls.__inst = object.__new__(cls)
        cls.__init(cls.__inst, *args, **kwargs)
        return cls.__inst

    def __init(self, engine, metadata, images_table, objects_table):
        self.__engine = engine
        self.__metadata = metadata
        self.__images_table = images_table
        self.__objects_table = objects_table

    @property
    def engine(self):
        return self.__engine

    @engine.setter
    def engine(self, engine):
        self.__engine = engine

    @property
    def metadata(self):
        return self.__metadata

    @metadata.setter
    def metadata(self, metadata):
        self.__metadata = metadata

    @property
    def images_table(self):
        return self.__images_table

    @images_table.setter
    def images_table(self, images_table):
        self.__images_table = images_table

    @property
    def objects_table(self):
        return self.__objects_table

    @objects_table.setter
    def objects_table(self, objects_table):
        self.__objects_table = objects_table


def connect_to_database():
    from sqlalchemy import create_engine, MetaData, Table

    try:
        database_url
    except:
        raise utils.ParameterException(
            'You need to specify the url of the database!')

    try:
        print 'Connecting to database: {}'.format(database_url)

        engine = create_engine(database_url)
        metadata = MetaData(bind=engine)
        metadata.create_all()
        images_table = Table('images', metadata, autoload=True,
                             autoload_with=engine)
        objects_table = Table('objects', metadata, autoload=True,
                              autoload_with=engine)
        DBConnection(engine, metadata, images_table, objects_table)

        utils.register_action(__name__, 'reload_db_schema',
                              'Reload database schema', reload_db_schema)

        utils.register_parameter(
            __name__, 'use_plate_filter', utils.PARAM_BOOL,
            'Select plates to import?', False, hook=use_filter_hook)
        utils.register_parameter(
            __name__, 'use_well_filter', utils.PARAM_BOOL,
            'Select wells to import?', False, hook=use_filter_hook)
        utils.register_parameter(
            __name__, 'use_replicate_filter', utils.PARAM_BOOL,
            'Select replicates to import?', False, hook=use_filter_hook)
        utils.register_parameter(
            __name__, 'use_position_filter', utils.PARAM_BOOL,
            'Select positions to import?', False, hook=use_filter_hook)
        utils.register_parameter(
            __name__, 'use_treatment_filter', utils.PARAM_BOOL,
            'Filter treatments to import?', False, hook=use_filter_hook)

        utils.register_parameter(
            __name__, 'image_id_db_column', utils.PARAM_STR,
            'Image id column in the image table', items=[])
        utils.register_parameter(
            __name__, 'object_img_id_db_column',
            utils.PARAM_STR, 'Image id column in the object table', items=[])
        utils.register_parameter(
            __name__, 'position_x_db_column', utils.PARAM_STR,
            'Position X column in the image table', items=[])
        utils.register_parameter(
            __name__, 'position_y_db_column', utils.PARAM_STR,
            'Position Y column in the image table', items=[])

        utils.register_parameter(
            __name__, 'plate_db_column', utils.PARAM_STR,
            'Plate column in the image table',
            items=[], hook=db_column_hook)
        utils.register_parameter(
            __name__, 'well_db_column', utils.PARAM_STR,
            'Well column in the image table',
            items=[], hook=db_column_hook)
        utils.register_parameter(
            __name__, 'replicate_db_column', utils.PARAM_STR,
            'Replicate column in the image table',
            items=[], hook=db_column_hook)
        utils.register_parameter(
            __name__, 'position_db_column', utils.PARAM_STR,
            'Position column in the image table',
            items=[], hook=db_column_hook)
        utils.register_parameter(
            __name__, 'treatment_db_column', utils.PARAM_STR,
            'Treatment column in the image table', items=[])

        utils.register_parameter(
            __name__, 'images_db_columns', utils.PARAM_STRS,
            'Columns to extract from the image table', items=[])
        utils.register_parameter(
            __name__, 'objects_db_columns', utils.PARAM_STRS,
            'Columns to extract from the objects table', items=[])
        utils.register_parameter(
            __name__, 'image_files_db_columns', utils.PARAM_STRS,
            'File and Path columns in the image table', items=[])

        utils.register_action(__name__, 'import_db',
                              'Import data from database', import_data_from_db)

        reload_db_schema()

    except:
        logger.error('Unable to connect to the database')
        raise

    return 'Successfully connected to database'


__use_filter_pattern = re.compile('use_(?P<name>\w+)_filter')


def use_filter_hook(module, param_name, value, yaml_filename):
    mo = __use_filter_pattern.match(param_name)
    bvalue = bool(value)
    if mo is not None:
        try:
            name = mo.group('name')
            format_str = '{}_filter'
            if name == 'treatment':
                format_str = '{}_filter_file'
            utils.set_parameter_hidden(__name__, format_str.format(name),
                                        not bvalue)
            utils.set_parameter_optional(__name__, format_str.format(name),
                                         not bvalue)
            utils.invalidate_module(__name__)
        except utils.ParameterException:
            return value
    return value


__db_column_pattern = re.compile('(?P<name>\w+)_db_column')


def db_column_hook(module, param_name, value, yaml_filename):
    from sqlalchemy import select
    mo = __db_column_pattern.match(param_name)
    if mo is not None:
        try:
            name = mo.group('name')
            engine = DBConnection().engine
            images_table = DBConnection().images_table
            result = engine.execute(select([images_table.c[value]]).distinct())
            fields = [field[0] for field in result]
            utils.set_parameter_kwargs(__name__, '{}_filter'.format(name),
                                       items=fields)
            utils.set_parameter_value(__name__, '{}_filter'.format(name),
                                      fields)
        except utils.ParameterException:
            return value
    return value


def reload_db_schema():
    print 'Loading database schemas'

    images_table = DBConnection().images_table
    objects_table = DBConnection().objects_table
    for param_name in ('images_db_columns', 'image_files_db_columns',
                       'image_id_db_column', 'position_db_column',
                       'plate_db_column', 'replicate_db_column',
                       'well_db_column', 'treatment_db_column'):
        utils.set_parameter_kwargs(__name__, param_name,
                                   items=images_table.c.keys())
        utils.set_parameter_visible(__name__, param_name)
    for param_name in ('objects_db_columns', 'object_img_id_db_column',
                       'position_x_db_column', 'position_y_db_column'):
        utils.set_parameter_kwargs(__name__, param_name,
                                   items=objects_table.c.keys())
        utils.set_parameter_visible(__name__, param_name)

    utils.invalidate_module(__name__)

    msg = 'Finished retrieving database schema'
    print msg
    return msg


IMAGE_ARRAY_BLOCKSIZE = 1000
OBJECT_ARRAY_BLOCKSIZE = 10000


def import_data_from_db():

    from sqlalchemy import select, or_, and_

    engine = DBConnection().engine
    images_table = DBConnection().images_table
    objects_table = DBConnection().objects_table

    try:
        treatment_filter_list = file(treatment_filter_file).readlines()
        treatment_filter_list \
            = [tr.strip() for tr in treatment_filter_list if tr.strip()]
    except:
        treatment_filter_list = None

    local_images_db_columns = dict([(column, True) \
                                    for column in images_db_columns])
    skip_img_columns = {}
    for column in image_files_db_columns:
        local_images_db_columns[column] = True
        skip_img_columns[column] = True
    for column in (plate_db_column, replicate_db_column,
                   well_db_column, treatment_db_column,
                   image_id_db_column):
        if column not in local_images_db_columns:
            local_images_db_columns[column] = True
            skip_img_columns[column] = True

    img_column_filter = lambda column: column in local_images_db_columns \
        and column not in skip_img_columns

    local_objects_db_columns = dict([(column, True) \
                                     for column in objects_db_columns])
    skip_obj_columns = {}
    for column in (position_x_db_column, position_y_db_column,
                   object_img_id_db_column):
        if column not in local_objects_db_columns:
            local_objects_db_columns[column] = True
            if column == object_img_id_db_column:
                skip_obj_columns[column] = True

    obj_column_filter = lambda column: column in local_objects_db_columns \
        and column not in skip_obj_columns

    img_index = 0
    obj_index = 0
    pl_index = 0
    repl_index = 0
    wl_index = 0
    tr_index = 0
    pdc = yaca_data_container()

    stmt = select([c for c in images_table.c if c.name \
                   in local_images_db_columns])
    and_args = []
    if use_plate_filter:
        or_args \
            = [images_table.c[plate_db_column] == pl for pl in plate_filter]
        and_args.append(or_(*or_args))
    if use_well_filter:
        or_args = [images_table.c[well_db_column] == wl for wl in well_filter]
        and_args.append(or_(*or_args))
    if use_replicate_filter:
        or_args = [images_table.c[replicate_db_column] == repl \
                   for repl in replicate_filter]
        and_args.append(or_(*or_args))
    if use_position_filter:
        or_args = [images_table.c[position_db_column] == pos \
                   for pos in position_filter]
        and_args.append(or_(*or_args))
    if and_args:
        if len(and_args) > 1:
            whereargs = and_(*and_args)
        else:
            whereargs = and_args[0]
        stmt = stmt.where(whereargs)
    if use_treatment_filter and len(treatment_filter_list) < 500:
        count_stmt = stmt.count()
        or_args = [images_table.c[treatment_db_column] == tr \
                   for tr in treatment_filter_list]
        count_stmt = stmt.where(or_(*or_args)).count()
        img_result = engine.execute(count_stmt)
        img_count = img_result.fetchone()[0]
        img_result.close()
    else:
        img_count = None
    if not use_treatment_filter:
        treatment_filter_list = [None]
    from itertools import izip_longest
    def grouper(iterable, n, fillvalue=None, fill=False):
        args = [iter(iterable)] * n
        return izip_longest(*args, fillvalue=fillvalue)
    conn = engine.connect()
    conn.execution_options(autocommit=False)
    #for tr_index, tr in enumerate(treatment_filter_list):
    for i, tr_group in enumerate(grouper(treatment_filter_list, 100)):
        trans = conn.begin()
        img_stmt = stmt
        if tr_group is not None:
            or_args = [images_table.c[treatment_db_column] == tr \
                   for tr in tr_group if tr is not None]
            img_stmt = img_stmt.where(or_(*or_args))
        img_result = conn.execute(img_stmt)
        for img_row in img_result:
            if img_count is None:
                sys.stdout.write(
                    '\rImporting treatment #{}-{} of {}, image #{}'.format(
                        i * 100 + 1, (i + 1) * 100,
                        len(treatment_filter_list), img_index + 1))
            else:
                sys.stdout.write(
                    '\rImporting image #{} of {}'.format(
                        img_index + 1, img_count))
            sys.stdout.flush()
            img = yaca_data_image()
            img.index = img_index
            plate = img_row[plate_db_column]
            replicate = img_row[replicate_db_column]
            well = img_row[well_db_column]
            treatment = img_row[treatment_db_column]
            if treatment not in pdc.treatmentByName:
                tr = yaca_data_treatment(treatment)
                tr.index = tr_index
                pdc.treatments.append(tr)
                tr_index += 1
                pdc.treatmentByName[treatment] = tr.index
            img.treatment = pdc.treatmentByName[treatment]
            if plate not in pdc.plateByName:
                pl = yaca_data_plate(plate)
                pl.index = pl_index
                pdc.plates.append(tr)
                pl_index += 1
                pdc.plateByName[plate] = pl.index
            img.plate = pdc.plateByName[plate]
            if well not in pdc.wellByName:
                wl = yaca_data_well(well)
                wl.index = wl_index
                pdc.wells.append(tr)
                wl_index += 1
                pdc.wellByName[well] = wl.index
            img.well = pdc.wellByName[well]
            if replicate not in pdc.replicateByName:
                repl = yaca_data_replicate(replicate)
                repl.index = pl_index
                pdc.replicates.append(tr)
                repl_index += 1
                pdc.replicateByName[replicate] = repl.index
            img.replicate = pdc.replicateByName[replicate]
            filenames = {}
            paths = {}
            for entry in img_row.iterkeys():
                if entry.startswith(IMAGE_FILENAME_IDENTIFIER):
                    entity_name = entry[len(IMAGE_FILENAME_IDENTIFIER):]
                    filenames[entity_name] = img_row[entry]
                elif entry.startswith(IMAGE_PATHNAME_IDENTIFIER):
                    entity_name = entry[len(IMAGE_PATHNAME_IDENTIFIER):]
                    paths[entity_name] = img_row[entry]
            imageFiles = []
            for entity_name, filename in filenames.iteritems():
                if entity_name in paths:
                    path = paths[entity_name]
                    full_path = os.path.join(path, filename)
                    imageFiles.append((entity_name, full_path))
            img.imageFiles = imageFiles
            if pdc.imgFeatures is None:
                for entry in img_row.iterkeys():
                    if not img_column_filter(entry):
                        continue
                    if not entry.startswith('Metadata_') \
                       and not entry.startswith(IMAGE_FILENAME_IDENTIFIER) \
                       and not entry.startswith(IMAGE_PATHNAME_IDENTIFIER):
                        pdc.imgFeatureIds[entry] = len(pdc.imgFeatureIds)
                pdc.imgFeatureIds[IMAGE_ID_FEATURE_NAME] = len(pdc.imgFeatureIds)
                pdc.imgImageFeatureId = pdc.imgFeatureIds[IMAGE_ID_FEATURE_NAME]
                pdc.imgFeatureIds[PLATE_ID_FEATURE_NAME] = len(pdc.imgFeatureIds)
                pdc.imgPlateFeatureId = pdc.imgFeatureIds[PLATE_ID_FEATURE_NAME]
                pdc.imgFeatureIds[WELL_ID_FEATURE_NAME] = len(pdc.imgFeatureIds)
                pdc.imgWellFeatureId = pdc.imgFeatureIds[WELL_ID_FEATURE_NAME]
                pdc.imgFeatureIds[REPLICATE_ID_FEATURE_NAME] \
                    = len(pdc.imgFeatureIds)
                pdc.imgReplicateFeatureId \
                    = pdc.imgFeatureIds[REPLICATE_ID_FEATURE_NAME]
                pdc.imgFeatureIds[TREATMENT_ID_FEATURE_NAME] \
                    = len(pdc.imgFeatureIds)
                pdc.imgTreatmentFeatureId \
                    = pdc.imgFeatureIds[TREATMENT_ID_FEATURE_NAME]
                pdc.imgFeatureIds[QUALITY_CONTROL_FEATURE_NAME] \
                    = len(pdc.imgFeatureIds)
                pdc.imgQualityControlFeatureId \
                    = pdc.imgFeatureIds[QUALITY_CONTROL_FEATURE_NAME]
                pdc.imgFeatures = numpy.empty((IMAGE_ARRAY_BLOCKSIZE,
                                               len(pdc.imgFeatureIds)))
            if not img.index < pdc.imgFeatures.shape[0]:
                imgFeatureShape = list(pdc.imgFeatures.shape)
                imgFeatureShape[0] += IMAGE_ARRAY_BLOCKSIZE
                pdc.imgFeatures.resize(imgFeatureShape)
            for entry in img_row.iterkeys():
                if entry.startswith('Metadata_') and not img_column_filter(entry):
                    img.properties[entry] = img_row[entry]
                elif entry in pdc.imgFeatureIds:
                    pdc.imgFeatures[img.index, pdc.imgFeatureIds[entry]] \
                        = img_row[entry]
            pdc.imgFeatures[img.index, pdc.imgImageFeatureId] = img.index
            pdc.imgFeatures[img.index, pdc.imgWellFeatureId] = wl.index
            pdc.imgFeatures[img.index, pdc.imgPlateFeatureId] = pl.index
            pdc.imgFeatures[img.index, pdc.imgTreatmentFeatureId] = tr.index
            pdc.imgFeatures[img.index, pdc.imgReplicateFeatureId] = repl.index
            pdc.imgFeatures[img.index, pdc.imgQualityControlFeatureId] \
                = QUALITY_CONTROL_DEFAULT
    
            obj_stmt = select([c for c in objects_table.c \
                           if c.name in local_objects_db_columns],
                          objects_table.c[object_img_id_db_column] \
                          == img_row[image_id_db_column])
            #count_obj_stmt = obj_stmt.count()
            #obj_result = conn.execute(count_obj_stmt)
            #obj_count = obj_result.fetchone()[0]
            #obj_result.close()
            obj_result = conn.execute(obj_stmt)
            for obj_row in obj_result:
                obj = yaca_data_object()
                obj.index = obj_index
                obj.image = img
                obj.position_x = obj_row[position_x_db_column]
                obj.position_y = obj_row[position_y_db_column]
                if pdc.objFeatures is None:
                    for entry in obj_row.iterkeys():
                        if not obj_column_filter(entry):
                            continue
                        if not entry.startswith('Metadata_'):
                            pdc.objFeatureIds[entry] = len(pdc.objFeatureIds)
                    pdc.objFeatureIds[OBJECT_ID_FEATURE_NAME] \
                        = len(pdc.objFeatureIds)
                    pdc.objObjectFeatureId \
                        = pdc.objFeatureIds[OBJECT_ID_FEATURE_NAME]
                    pdc.objFeatureIds[IMAGE_ID_FEATURE_NAME] \
                        = len(pdc.objFeatureIds)
                    pdc.objImageFeatureId \
                        = pdc.objFeatureIds[IMAGE_ID_FEATURE_NAME]
                    pdc.objFeatureIds[PLATE_ID_FEATURE_NAME] \
                        = len(pdc.objFeatureIds)
                    pdc.objPlateFeatureId \
                        = pdc.objFeatureIds[PLATE_ID_FEATURE_NAME]
                    pdc.objFeatureIds[WELL_ID_FEATURE_NAME] \
                        = len(pdc.objFeatureIds)
                    pdc.objWellFeatureId = pdc.objFeatureIds[WELL_ID_FEATURE_NAME]
                    pdc.objFeatureIds[REPLICATE_ID_FEATURE_NAME] \
                        = len(pdc.objFeatureIds)
                    pdc.objReplicateFeatureId \
                        = pdc.objFeatureIds[REPLICATE_ID_FEATURE_NAME]
                    pdc.objFeatureIds[TREATMENT_ID_FEATURE_NAME] \
                        = len(pdc.objFeatureIds)
                    pdc.objTreatmentFeatureId \
                        = pdc.objFeatureIds[TREATMENT_ID_FEATURE_NAME]
                    pdc.objFeatureIds[QUALITY_CONTROL_FEATURE_NAME] \
                        = len(pdc.objFeatureIds)
                    pdc.objQualityControlFeatureId \
                        = pdc.objFeatureIds[QUALITY_CONTROL_FEATURE_NAME]
                    pdc.objFeatures = numpy.empty((OBJECT_ARRAY_BLOCKSIZE,
                                                   len(pdc.objFeatureIds)))
                if not obj.index < pdc.objFeatures.shape[0]:
                    objFeatureShape = list(pdc.objFeatures.shape)
                    objFeatureShape[0] += OBJECT_ARRAY_BLOCKSIZE
                    pdc.objFeatures.resize(objFeatureShape)
                for entry in obj_row.iterkeys():
                    if entry in pdc.objFeatureIds:
                        pdc.objFeatures[obj.index, pdc.objFeatureIds[entry]] \
                            = obj_row[entry]
                pdc.objFeatures[obj.index, pdc.objObjectFeatureId] = obj.index
                pdc.objFeatures[obj.index, pdc.objImageFeatureId] = img.index
                pdc.objFeatures[obj.index, pdc.objWellFeatureId] = wl.index
                pdc.objFeatures[obj.index, pdc.objPlateFeatureId] = pl.index
                pdc.objFeatures[obj.index, pdc.objTreatmentFeatureId] = tr.index
                pdc.objFeatures[obj.index, pdc.objReplicateFeatureId] = repl.index
                pdc.objFeatures[obj.index, pdc.objQualityControlFeatureId] \
                    = QUALITY_CONTROL_DEFAULT
                pdc.objects.append(obj)
                obj_index += 1
            obj_result.close()
            pdc.images.append(img)
            img_index += 1
        img_result.close()
        trans.commit()
    sys.stdout.write('\n')

    assert img_index == len(pdc.images)
    assert obj_index == len(pdc.objects)
    assert wl_index == len(pdc.wells)
    assert pl_index == len(pdc.plates)
    assert tr_index == len(pdc.treatments)
    assert repl_index == len(pdc.replicates)

    # actually len(pdc.images) == 0 implies len(pdc.objects) == 0
    if len(pdc.images) == 0 or len(pdc.objects) == 0:
        raise Exception("Failed to import data: no objects")

    imgFeatureShape = list(pdc.imgFeatures.shape)
    imgFeatureShape[0] = img_index
    pdc.imgFeatures.resize(imgFeatureShape)

    objFeatureShape = list(pdc.objFeatures.shape)
    objFeatureShape[0] = obj_index
    pdc.objFeatures.resize(objFeatureShape)

    importer.Importer().set_pdc(pdc)

    utils.update_state(importer.__name__, 'imported')

    print 'Finished importing data from database'
    return 'Finished importing data from database'


__dict__ = sys.modules[__name__].__dict__

utils.register_module(__name__, 'Database import', __dict__,
                      utils.DEFAULT_STATE)

utils.register_parameter(__name__, 'database_url', utils.PARAM_STR,
                         'URL specifying a database', 'mysql://localhost/YACA')

utils.register_action(__name__, 'connect_to_database', 'Connect to database',
                      connect_to_database)

utils.register_parameter(
    __name__, 'use_plate_filter', utils.PARAM_BOOL,
    'Select plates to import?', False, hidden=True)
utils.register_parameter(
    __name__, 'plate_filter', utils.PARAM_STRS,
    'Plates to import from the database',
    optional=True, hidden=True, items=[])
utils.register_parameter(
    __name__, 'use_well_filter', utils.PARAM_BOOL,
    'Select wells to import?', False, hidden=True)
utils.register_parameter(
    __name__, 'well_filter', utils.PARAM_STRS,
    'Wells to import from the database',
    optional=True, hidden=True, items=[])
utils.register_parameter(
    __name__, 'use_replicate_filter', utils.PARAM_BOOL,
    'Select replicates to import?', False, hidden=True)
utils.register_parameter(
    __name__, 'replicate_filter', utils.PARAM_STRS,
    'Replicates to import from the database',
    optional=True, hidden=True, items=[])
utils.register_parameter(
    __name__, 'use_position_filter', utils.PARAM_BOOL,
    'Select positions to import?', False, hidden=True)
utils.register_parameter(
    __name__, 'position_filter', utils.PARAM_STRS,
    'Positions to import from the database',
    optional=True, hidden=True, items=[])
utils.register_parameter(
    __name__, 'use_treatment_filter', utils.PARAM_BOOL,
    'Filter treatments to import?', False, hidden=True)
utils.register_parameter(
    __name__, 'treatment_filter_file', utils.PARAM_INPUT_FILE,
    'File listing the treatments to import from the database',
    optional=True, hidden=True, items=[])

utils.register_parameter(
    __name__, 'image_id_db_column', utils.PARAM_STR,
    'Image id column in the image table', 'id',
    optional=True, hidden=True, items=[])
utils.register_parameter(
    __name__, 'object_img_id_db_column', utils.PARAM_STR,
    'Image id column in the object table', 'image_id',
    optional=True, hidden=True, items=[])
utils.register_parameter(
    __name__, 'position_x_db_column', utils.PARAM_STR,
    'Position X column in the image table', 'Cells_Location_Center_X',
    optional=True, hidden=True, items=[])
utils.register_parameter(
    __name__, 'position_y_db_column', utils.PARAM_STR,
    'Position Y column in the image table', 'Cells_Location_Center_Y',
    optional=True, hidden=True, items=[])

utils.register_parameter(
    __name__, 'plate_db_column', utils.PARAM_STR,
    'Plate column in the image table', 'Metadata_Plate',
    optional=True, hidden=True, items=[])
utils.register_parameter(
    __name__, 'well_db_column', utils.PARAM_STR,
    'Well column in the image table', 'Metadata_Well',
    optional=True, hidden=True, items=[])
utils.register_parameter(
    __name__, 'replicate_db_column', utils.PARAM_STR,
    'Replicate column in the image table', 'Metadata_Replicate',
    optional=True, hidden=True, items=[])
utils.register_parameter(
    __name__, 'position_db_column', utils.PARAM_STR,
    'Position column in the image table', 'Metadata_Position',
    optional=True, hidden=True, items=[])
utils.register_parameter(
    __name__, 'treatment_db_column', utils.PARAM_STR,
    'Treatment column in the image table', 'Metadata_Treatment',
    optional=True, hidden=True, items=[])

utils.register_parameter(
    __name__, 'images_db_columns', utils.PARAM_STRS,
    'Columns to extract from the image table',
    optional=True, hidden=True, items=[])
utils.register_parameter(
    __name__, 'objects_db_columns', utils.PARAM_STRS,
    'Columns to extract from the objects table',
    optional=True, hidden=True, items=[])
utils.register_parameter(
    __name__, 'image_files_db_columns', utils.PARAM_STRS,
    'File and Path columns in the image table',
    optional=True, hidden=True, items=[])
