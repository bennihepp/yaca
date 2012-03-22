# -*- coding: utf-8 -*-

"""
import_cp2_csv.py -- Importer for CellProfiler 2.0 CSV files.
"""

# This software is distributed under the FreeBSD License.
# See the accompanying file LICENSE for details.
# 
# Copyright 2011 Benjamin Hepp

import csv
import re
import numpy
import sys
import os
import traceback

from data_container import *

WELL_ID_CP2_FEATURE_NAME = 'Metadata_WELL_ID'
TREATMENT_ID_CP2_FEATURE_NAME = 'Metadata_TREATMENT_ID'
REPLICATE_ID_CP2_FEATURE_NAME = 'Metadata_REPLICATE_ID'

OBJECT_IMAGE_ID_IDENTIFIER = 'ImageNumber'
def default_image_id_extractor(row, column_names):
    i = column_names.index(OBJECT_IMAGE_ID_IDENTIFIER)
    if i < 0:
        raise Exception('Unable to extract image id')
    return int(row[i]) - 1

IMAGE_FILENAME_IDENTIFIER = 'FileName_'
IMAGE_PATHNAME_IDENTIFIER = 'PathName_'
#IMAGE_CHANNELS = (('imNucleus','R'), ('imProtein','G'), ('imCellMask','B'), ('SegC03','O1'), ('SegC01','O2'))
#IMAGE_CHANNELS = (('imNucleus','R'), ('imProtein','G'), ('imCellMask','B'), ('CellsObjects','O1'), ('NucleiObjects','O2'))
def default_image_files_extractor(row, column_names):

    filenames = {}
    paths = {}

    for i in xrange(len(column_names)):
        if column_names[i].startswith(IMAGE_FILENAME_IDENTIFIER):
            entity_name = column_names[i][len(IMAGE_FILENAME_IDENTIFIER) :]
            filenames[entity_name] = row[i]
        elif column_names[i].startswith(IMAGE_PATHNAME_IDENTIFIER):
            entity_name = column_names[i][len(IMAGE_PATHNAME_IDENTIFIER) :]
            paths[entity_name] = row[i]

    imageFiles = []

    for entity_name,filename in filenames.iteritems():

        if entity_name in paths:

            path = paths[entity_name]
            full_path = os.path.join(path, filename)

            imageFiles.append((entity_name, full_path))

    return imageFiles

OBJECT_POSITION_X_IDENTIFIER = 'Location_Center_X'
OBJECT_POSITION_Y_IDENTIFIER = 'Location_Center_Y'
def default_object_position_extractor(row, column_names):
    xi = column_names.index(OBJECT_POSITION_X_IDENTIFIER)
    yi = column_names.index(OBJECT_POSITION_Y_IDENTIFIER)
    if xi < 0 or yi < 0:
        raise Exception('Unable to extract object position')
    return float(row[xi]),float(row[yi])

# new way of doing this
WELL_FEATURE_IDENTIFIER = 'Metadata_Well'
def default_well_extractor(row, column_names):
    well_name = None
    for i in xrange(len(column_names)):
        name = column_names[i]
        if name == WELL_FEATURE_IDENTIFIER:
            return row[i]
    if well_name == None:
        raise Exception('Unable to extract well identifier')
    return well_name

TREATMENT_FEATURE_IDENTIFIER = 'Metadata_Treatment'
def default_treatment_extractor(row, column_names):
    treatment_name = None
    for i in xrange(len(column_names)):
        name = column_names[i]
        if name == TREATMENT_FEATURE_IDENTIFIER:
            return row[i]
    if treatment_name == None:
        raise Exception('Unable to extract treatment identifier')
    return treatment_name

REPLICATE_FEATURE_IDENTIFIER = 'Metadata_Replicate'
def default_replicate_extractor(row, column_names):
    replicate_name = None
    for i in xrange(len(column_names)):
        name = column_names[i]
        if name == REPLICATE_FEATURE_IDENTIFIER:
            return row[i]
    if replicate_name == None:
        raise Exception('Unable to extract replicate identifier')
    return replicate_name

## old way of doing this
#
#WELL_FEATURE_IDENTIFIER = 'Metadata_ImageFileName'
#WELL_FEATURE_IDENTIFIER_PATTERN = re.compile('W(\d*)--')
#def default_well_extractor(row, column_names):
    #well_name = None
    #for i in xrange(len(column_names)):
        #name = column_names[i]
        #if name == WELL_FEATURE_IDENTIFIER:
            #match = WELL_FEATURE_IDENTIFIER_PATTERN.search(row[i])
            #well_name = match.group(1)
            #break
    #if well_name == None:
        #raise Exception('Unable to extract well identifier')
    #return well_name
#
#TREATMENT_FEATURE_IDENTIFIER = 'Metadata_RelativeImagePath'
#def default_treatment_extractor(row, column_names):
    #treatment_name = None
    #for i in xrange(len(column_names)):
        #name = column_names[i]
        #if name == TREATMENT_FEATURE_IDENTIFIER:
            ## make sure we are windows-compatible
            #v = row[i]
            #v = v.replace('\\','/')
            ##treatment_name = os.path.split(os.path.split(v)[0])[-1]
            #treatment_name = os.path.split(v)[-1]
            #break
    #if treatment_name == None:
        #raise Exception('Unable to extract treatment identifier')
    #return treatment_name
#
#REPLICATE_FEATURE_IDENTIFIER = 'Metadata_RelativeImagePath'
#def default_replicate_extractor(row, column_names):
    #replicate_name = None
    #for i in xrange(len(column_names)):
        #name = column_names[i]
        #if name == REPLICATE_FEATURE_IDENTIFIER:
            ## make sure we are windows-compatible
            #v = row[i]
            #v = v.replace('\\','/')
            ##replicate_name = os.path.split(os.path.split(v)[0])[-1]
            #replicate_name = os.path.split(v)[-2]
            #break
    #if replicate_name == None:
        #raise Exception('Unable to extract replicate identifier')
    #return replicate_name



def init_pdc(pdc, working_dict, image_data, object_data, image_file_postfix, object_file_postfixes):

    # create object_prefixes
    objects_prefixes = list(object_file_postfixes)
    for i in xrange(len(objects_prefixes)):
        s = objects_prefixes[i]
        if s.startswith('_'):
            s = s[1 :]
        if not s.endswith('_'):
            s = s + '_'
        objects_prefixes[i] = s

    # write meta information for images into pdc
    img,icn,ict = image_data
    for i in xrange(len(ict)):
        if ict[i] == float:
            n = len(pdc.imgFeatureIds)
            pdc.imgFeatureIds[icn[i]] = n
        else:
            if not 'imgPropertyNames' in working_dict:
                working_dict['imgPropertyNames'] = []
            working_dict['imgPropertyNames'].append(icn[i])


    object_column_types = []

    # combine meta-information about the objects into pdc
    for k in xrange(len(object_data)):

        object_column_types.append(object_data[k][2])

        o,ocn,oct = object_data[k]
        for i in xrange(len(oct)):
            if object_column_types[k][i] == float:
                if oct[i] == float:
                    n = len(pdc.objFeatureIds)
                    pdc.objFeatureIds[objects_prefixes[k] + ocn[i]] = n
                else:
                    object_column_types[k][i] = str

    working_dict['object_column_types'] = object_column_types

    # check if we have to provide a imageId-feature for the images
    working_dict['img_image_id_feature_is_virtual'] = False
    if not pdc.imgFeatureIds.has_key(IMAGE_ID_FEATURE_NAME):
        pdc.imgFeatureIds[IMAGE_ID_FEATURE_NAME] = len(pdc.imgFeatureIds)
        working_dict['img_image_id_feature_is_virtual'] = True
    pdc.imgImageFeatureId = pdc.imgFeatureIds[IMAGE_ID_FEATURE_NAME]

    if pdc.imgFeatureIds.has_key(QUALITY_CONTROL_FEATURE_NAME):
        pdc.imgQualityControlFeatureId = -1
    else:
        print 'creating quality control feature for image'
        pdc.imgFeatureIds[QUALITY_CONTROL_FEATURE_NAME] = len(pdc.imgFeatureIds)
        pdc.imgQualityControlFeatureId = pdc.imgFeatureIds[QUALITY_CONTROL_FEATURE_NAME]

    if pdc.objFeatureIds.has_key(QUALITY_CONTROL_FEATURE_NAME):
        pdc.objQualityControlFeatureId = -1
    else:
        print 'creating quality control feature for image'
        pdc.objFeatureIds[QUALITY_CONTROL_FEATURE_NAME] = len(pdc.objFeatureIds)
        pdc.objQualityControlFeatureId = pdc.objFeatureIds[QUALITY_CONTROL_FEATURE_NAME]

    # check if we have to provide a virtual wellId-feature for the images
    working_dict['img_well_id_feature_is_virtual'] = False
    if pdc.imgFeatureIds.has_key(WELL_ID_FEATURE_NAME):
        pdc.imgWellFeatureId = pdc.imgFeatureIds[WELL_ID_FEATURE_NAME]
    elif pdc.imgFeatureIds.has_key(WELL_ID_CP2_FEATURE_NAME):
        pdc.imgWellFeatureId = pdc.imgFeatureIds[WELL_ID_CP2_FEATURE_NAME]
    else:
        print 'creating well feature for image'
        pdc.imgFeatureIds[WELL_ID_FEATURE_NAME] = len(pdc.imgFeatureIds)
        working_dict['img_well_id_feature_is_virtual'] = True
        pdc.imgWellFeatureId = pdc.imgFeatureIds[WELL_ID_FEATURE_NAME]
        if WELL_ID_FEATURE_NAME in working_dict['imgPropertyNames']:
            working_dict['img_well_id_property_name'] = WELL_ID_FEATURE_NAME
        elif WELL_ID_CP2_FEATURE_NAME in working_dict['imgPropertyNames']:
            working_dict['img_well_id_property_name'] = WELL_ID_CP2_FEATURE_NAME
        else:
            working_dict['img_well_id_property_name'] = None

    # check if we have to provide a virtual treatmentId-feature for the images
    working_dict['img_treatment_id_feature_is_virtual'] = False
    if  pdc.imgFeatureIds.has_key(TREATMENT_ID_FEATURE_NAME):
        pdc.imgTreatmentFeatureId = pdc.imgFeatureIds[TREATMENT_ID_FEATURE_NAME]
    elif pdc.imgFeatureIds.has_key(TREATMENT_ID_CP2_FEATURE_NAME):
        pdc.imgTreatmentFeatureId = pdc.imgFeatureIds[TREATMENT_ID_CP2_FEATURE_NAME]
    else:
        print 'creating treatment feature for image'
        pdc.imgFeatureIds[TREATMENT_ID_FEATURE_NAME] = len(pdc.imgFeatureIds)
        working_dict['img_treatment_id_feature_is_virtual'] = True
        pdc.imgTreatmentFeatureId = pdc.imgFeatureIds[TREATMENT_ID_FEATURE_NAME]
        if TREATMENT_ID_FEATURE_NAME in working_dict['imgPropertyNames']:
            working_dict['img_treatment_id_property_name'] = TREATMENT_ID_FEATURE_NAME
        elif TREATMENT_ID_CP2_FEATURE_NAME in working_dict['imgPropertyNames']:
            working_dict['img_treatment_id_property_name'] = TREATMENT_ID_CP2_FEATURE_NAME
        else:
            working_dict['img_treatment_id_property_name'] = None

    # check if we have to provide a virtual replicateId-feature for the images
    working_dict['img_replicate_id_feature_is_virtual'] = False
    if pdc.imgFeatureIds.has_key(REPLICATE_ID_FEATURE_NAME):
        pdc.imgReplicateFeatureId = pdc.imgFeatureIds[REPLICATE_ID_FEATURE_NAME]
    elif pdc.imgFeatureIds.has_key(REPLICATE_ID_CP2_FEATURE_NAME):
        pdc.imgReplicateFeatureId = pdc.imgFeatureIds[REPLICATE_ID_CP2_FEATURE_NAME]
    else:
        print 'creating replicate feature for image'
        pdc.imgFeatureIds[REPLICATE_ID_FEATURE_NAME] = len(pdc.imgFeatureIds)
        working_dict['img_replicate_id_feature_is_virtual'] = True
        pdc.imgReplicateFeatureId = pdc.imgFeatureIds[REPLICATE_ID_FEATURE_NAME]
        if REPLICATE_ID_FEATURE_NAME in working_dict['imgPropertyNames']:
            working_dict['img_replicate_id_property_name'] = REPLICATE_ID_FEATURE_NAME
        elif REPLICATE_ID_CP2_FEATURE_NAME in working_dict['imgPropertyNames']:
            working_dict['img_replicate_id_property_name'] = REPLICATE_ID_CP2_FEATURE_NAME
        else:
            working_dict['img_replicate_id_property_name'] = None

    # check if we have to provide a virtual objectId-feature for the objects
    working_dict['obj_object_id_feature_is_virtual'] = False
    if not pdc.objFeatureIds.has_key(OBJECT_ID_FEATURE_NAME):
        pdc.objFeatureIds[OBJECT_ID_FEATURE_NAME] = len(pdc.objFeatureIds)
        working_dict['obj_object_id_feature_is_virtual'] = True
    pdc.objObjectFeatureId = pdc.objFeatureIds[OBJECT_ID_FEATURE_NAME]

    # check if we have to provide a virtual imageId-feature for the objects
    working_dict['obj_image_id_feature_is_virtual'] = False
    if not pdc.objFeatureIds.has_key(IMAGE_ID_FEATURE_NAME):
        pdc.objFeatureIds[IMAGE_ID_FEATURE_NAME] = len(pdc.objFeatureIds)
        working_dict['obj_image_id_feature_is_virtual'] = True
    pdc.objImageFeatureId = pdc.objFeatureIds[IMAGE_ID_FEATURE_NAME]

    # check if we have to provide a virtual wellId-feature for the objects
    working_dict['obj_well_id_feature_is_virtual'] = False
    if  (not pdc.objFeatureIds.has_key(WELL_ID_FEATURE_NAME)) \
    and (not pdc.objFeatureIds.has_key(WELL_ID_CP2_FEATURE_NAME)):
        print 'creating well feature for objects'
        pdc.objFeatureIds[WELL_ID_FEATURE_NAME] = len(pdc.objFeatureIds)
        working_dict['obj_well_id_feature_is_virtual'] = True
        pdc.objWellFeatureId = pdc.objFeatureIds[WELL_ID_FEATURE_NAME]
    elif pdc.objFeatureIds.has_key(WELL_ID_CP2_FEATURE_NAME):
        pdc.objWellFeatureId = pdc.objFeatureIds[WELL_ID_CP2_FEATURE_NAME]
    else:
        pdc.objWellFeatureId = pdc.objFeatureIds[WELL_ID_FEATURE_NAME]

    # check if we have to provide a virtual treatmentId-feature for the objects
    working_dict['obj_treatment_id_feature_is_virtual'] = False
    if  (not pdc.objFeatureIds.has_key(TREATMENT_ID_FEATURE_NAME)) \
    and (not pdc.objFeatureIds.has_key(TREATMENT_ID_CP2_FEATURE_NAME)):
        print 'creating treatment feature for objects'
        pdc.objFeatureIds[TREATMENT_ID_FEATURE_NAME] = len(pdc.objFeatureIds)
        working_dict['obj_treatment_id_feature_is_virtual'] = True
        pdc.objTreatmentFeatureId = pdc.objFeatureIds[TREATMENT_ID_FEATURE_NAME]
    elif pdc.objFeatureIds.has_key(TREATMENT_ID_CP2_FEATURE_NAME):
        pdc.objTreatmentFeatureId = pdc.objFeatureIds[TREATMENT_ID_CP2_FEATURE_NAME]
    else:
        pdc.objTreatmentFeatureId = pdc.objFeatureIds[TREATMENT_ID_FEATURE_NAME]

    # check if we have to provide a virtual replicateId-feature for the objects
    working_dict['obj_replicate_id_feature_is_virtual'] = False
    if  (not pdc.objFeatureIds.has_key(REPLICATE_ID_FEATURE_NAME)) \
    and (not pdc.objFeatureIds.has_key(REPLICATE_ID_CP2_FEATURE_NAME)):
        pdc.objFeatureIds[REPLICATE_ID_FEATURE_NAME] = len(pdc.objFeatureIds)
        working_dict['obj_replicate_id_feature_is_virtual'] = True
        pdc.objReplicateFeatureId = pdc.objFeatureIds[REPLICATE_ID_FEATURE_NAME]
    elif pdc.objFeatureIds.has_key(REPLICATE_ID_CP2_FEATURE_NAME):
        pdc.objReplicateFeatureId = pdc.objFeatureIds[REPLICATE_ID_CP2_FEATURE_NAME]
    else:
        pdc.objReplicateFeatureId = pdc.objFeatureIds[REPLICATE_ID_FEATURE_NAME]



IMAGE_ARRAY_BLOCKSIZE = 8
OBJECT_ARRAY_BLOCKSIZE = 8 * 256

def update_pdc(pdc, image_data, object_data):

    # if necessary, create feature-tables

    if pdc.imgFeatures == None:
        pdc.imgFeatures = numpy.empty((0 , len(pdc.imgFeatureIds)))

    if pdc.objFeatures == None:
        pdc.objFeatures = numpy.empty((0 , len(pdc.objFeatureIds)))

    # if necessary, update size of feature-tables

    num_of_new_images = len(image_data[0])
    image_table_shape = list(pdc.imgFeatures.shape)
    if image_table_shape[0] < (len(pdc.images) + num_of_new_images):
        image_table_shape[0] += max(num_of_new_images, IMAGE_ARRAY_BLOCKSIZE)
        pdc.imgFeatures.resize(image_table_shape)

    num_of_new_objects = len(object_data[0][0])
    object_table_shape = list(pdc.objFeatures.shape)
    if object_table_shape[0] < (len(pdc.objects) + num_of_new_objects):
        object_table_shape[0] += max(num_of_new_objects, OBJECT_ARRAY_BLOCKSIZE)
        pdc.objFeatures.resize(object_table_shape)



def correct_image_data(image_data):
    special_column_names = [TREATMENT_ID_FEATURE_NAME, TREATMENT_ID_CP2_FEATURE_NAME, REPLICATE_ID_FEATURE_NAME, REPLICATE_ID_CP2_FEATURE_NAME, WELL_ID_FEATURE_NAME, WELL_ID_CP2_FEATURE_NAME]

    img,icn,ict = image_data
    for i in xrange(len(ict)):
        if ict[i] == float and icn[i] in special_column_names:
            ict[i] = str

def correct_object_data(object_data):
    special_column_names = [TREATMENT_ID_FEATURE_NAME, TREATMENT_ID_CP2_FEATURE_NAME, REPLICATE_ID_FEATURE_NAME, REPLICATE_ID_CP2_FEATURE_NAME, WELL_ID_FEATURE_NAME, WELL_ID_CP2_FEATURE_NAME]

    for k in xrange(len(object_data)):

        o,ocn,oct = object_data[k]
        for i in xrange(len(oct)):
            if oct[i] == float and ocn[i] in special_column_names:
                oct[i] = str

def fill_pdc(pdc, working_dict, image_data, object_data, image_file_postfix, object_file_postfixes,
             image_id_extractor, image_files_extractor, object_position_extractor,
             well_extractor, treatment_extractor, replicate_extractor):

    correct_image_data(image_data)
    correct_object_data(object_data)

    if len(pdc.images) <= 0:
        init_pdc(pdc, working_dict, image_data, object_data, image_file_postfix, object_file_postfixes)

    # update feature-tables
    update_pdc(pdc, image_data, object_data)

    object_column_types = working_dict['object_column_types']


    # fill yaca_data_structure

    images,image_column_names,image_column_types = image_data

    o0,ocn,oct = object_data[0]

    old_num_of_images = len(pdc.images)

    for i in xrange(len(images)):

        img = yaca_data_image()
        img.index = len(pdc.images)

        try:
            img.imageFiles = image_files_extractor(images[i], image_column_names)
        except Exception,e:
            img.import_state = 'no_image_files'
            tb = "".join(traceback.format_tb(sys.exc_info()[2]))
            pdc.errors.append(yaca_data_error(e, tb, img))
            raise

        n = 0
        for j in xrange(len(image_column_types)):
            if image_column_types[j] != float:
                img.properties[image_column_names[j]] = images[i][j]
            else:
                v1 = images[i]
                v2 = float(v1[j])
                q = pdc.imgFeatures[img.index]
                pdc.imgFeatures[img.index][n] = v2
                #pdc.imgFeatures[img.index][n] = float(images[i][j])
                n += 1

        try:

            if working_dict['img_well_id_feature_is_virtual']:
                if working_dict['img_well_id_property_name']:
                    well_name = img.properties[working_dict['img_well_id_property_name']]
                else:
                    well_name = well_extractor(images[i], image_column_names)
            else:
                well_name = str(pdc.imgFeatures[img.index, pdc.imgWellFeatureId])

            if not pdc.wellByName.has_key(well_name):
                well = yaca_data_well(well_name)
                well.index = len(pdc.wells)
                pdc.wellByName[well_name] = len(pdc.wells)
                pdc.wells.append(well)
                img.well = well
                #print 'creating well %d: %s' % (well.index, well.name)
            else:
                img.well = pdc.wells[pdc.wellByName[well_name]]
                #print 'using well %d: %s' % (img.well.index, img.well.name)

        except Exception,e:
            img.import_state = 'no_well'
            tb = "".join(traceback.format_tb(sys.exc_info()[2]))
            pdc.errors.append(yaca_data_error(e, tb, img))
            raise

        try:

            if working_dict['img_treatment_id_feature_is_virtual']:
                if working_dict['img_treatment_id_property_name']:
                    treatment_name = img.properties[working_dict['img_treatment_id_property_name']]
                else:
                    treatment_name = treatment_extractor(images[i], image_column_names)
            else:
                treatment_name = str(pdc.imgFeatures[img.index, pdc.imgTreatmentFeatureId])

            if not pdc.treatmentByName.has_key(treatment_name):
                #print 'creating treatment with name %s' % treatment_name
                treatment = yaca_data_treatment(treatment_name)
                treatment.index = len(pdc.treatments)
                pdc.treatmentByName[treatment_name] = len(pdc.treatments)
                pdc.treatments.append(treatment)
                img.treatment = treatment
                #print 'creating treatment %d: %s' % (treatment.index, treatment.name)
            else:
                img.treatment = pdc.treatments[pdc.treatmentByName[treatment_name]]

        except Exception,e:
            img.import_state = 'no_treatment'
            tb = "".join(traceback.format_tb(sys.exc_info()[2]))
            pdc.errors.append(yaca_data_error(e, tb, img))
            raise

        try:

            if working_dict['img_replicate_id_feature_is_virtual']:
                if working_dict['img_replicate_id_property_name']:
                    replicate_name = img.properties[working_dict['img_replicate_id_property_name']]
                else:
                    replicate_name = replicate_extractor(images[i], image_column_names)
            else:
                replicate_name = str(pdc.imgFeatures[img.index, pdc.imgReplicateFeatureId])

            if not pdc.replicateByName.has_key(replicate_name):
                replicate = yaca_data_replicate(replicate_name)
                replicate.index = len(pdc.replicates)
                pdc.replicateByName[replicate_name] = len(pdc.replicates)
                pdc.replicates.append(replicate)
                img.replicate = replicate
                #print 'creating replicate %d: %s' % (replicate.index, replicate.name)
            else:
                img.replicate = pdc.replicates[pdc.replicateByName[replicate_name]]

        except Exception,e:
            img.import_state = 'no_replicate'
            tb = "".join(traceback.format_tb(sys.exc_info()[2]))
            pdc.errors.append(yaca_data_error(e, tb, img))
            raise


        if working_dict['img_image_id_feature_is_virtual']:
            pdc.imgFeatures[img.index][pdc.imgImageFeatureId] = img.index

        if working_dict['img_well_id_feature_is_virtual']:
            pdc.imgFeatures[img.index][pdc.imgWellFeatureId] = img.well.index

        if working_dict['img_treatment_id_feature_is_virtual']:
            pdc.imgFeatures[img.index][pdc.imgTreatmentFeatureId] = img.treatment.index

        if working_dict['img_replicate_id_feature_is_virtual']:
            pdc.imgFeatures[img.index][pdc.imgReplicateFeatureId] = img.replicate.index

        pdc.images.append(img)

    for i in xrange(len(o0)):

        obj = yaca_data_object()
        obj.index = len(pdc.objects)

        found_img_id = False
        found_obj_position = False

        n = 0
        for k in xrange(len(object_data)):

            o,ocn,oct = object_data[k]

            try:
                #TODO: It seems that the CellProfiler output changed with revision 11429
                #image_id = image_id_extrac tor(o[i], ocn) + old_num_of_images
                # assume a single image per tuple of input files
                image_id = -1
                obj.image = pdc.images[image_id]
                found_img_id = True
            except:
                pass

            if not found_obj_position:
                try:
                    obj.position_x,obj.position_y = object_position_extractor(o[i], ocn)
                    found_obj_position = True
                except:
                    pass        

            for j in xrange(len(oct)):
                if object_column_types[k][j] == float:
                    pdc.objFeatures[obj.index][n] = float(o[i][j])
                    n += 1
                #else:
                #    obj.properties[ocn[j]] = o[i][j]

        if not found_img_id:
            obj.import_state = 'no image'
            e = Exception('Unable to extract image id')
            tb = "".join(traceback.format_tb(sys.exc_info()[2]))
            pdc.errors.append(yaca_data_error(e, tb, obj))
            raise e
        if not found_obj_position:
            obj.import_state = 'no_position'
            e = Exception('Unable to extract object position')
            tb = "".join(traceback.format_tb(sys.exc_info()[2]))
            pdc.errors.append(yaca_data_error(e, tb, img))
            raise e


        if working_dict['obj_object_id_feature_is_virtual']:
            pdc.objFeatures[obj.index , pdc.objObjectFeatureId] = obj.index

        if working_dict['obj_image_id_feature_is_virtual']:
            pdc.objFeatures[obj.index , pdc.objImageFeatureId] = obj.image.index

        if working_dict['obj_well_id_feature_is_virtual']:
            pdc.objFeatures[obj.index , pdc.objWellFeatureId] = obj.image.well.index

        if working_dict['obj_treatment_id_feature_is_virtual']:
            pdc.objFeatures[obj.index , pdc.objTreatmentFeatureId] = obj.image.treatment.index

        if working_dict['obj_replicate_id_feature_is_virtual']:
            pdc.objFeatures[obj.index , pdc.objReplicateFeatureId] = obj.image.replicate.index

        pdc.objects.append(obj)



def import_cp2_csv_results_recursive(path, pdc, working_dict, image_file_postfix, object_file_postfixes,
                                     csv_delimiter, csv_extension,
                                     image_id_extractor=default_image_id_extractor,
                                     image_files_extractor=default_image_files_extractor,
                                     object_position_extractor=default_object_position_extractor,
                                     well_extractor=default_well_extractor,
                                     treatment_extractor=default_treatment_extractor,
                                     replicate_extractor=default_replicate_extractor):

    print 'entering %s ...' % path

    current_num_of_images = 0
    current_num_of_objects = 0

    files = os.listdir(path)
    files.sort()

    for file in files:

        tmp_file = os.path.join(path, file)

        if os.path.isdir(tmp_file):
            #print 'recursing into %s' % file
            tmp_path = os.path.join(path, file)
            num_of_images,num_of_objects = import_cp2_csv_results_recursive(
                                              tmp_file, pdc, working_dict,
                                              image_file_postfix, object_file_postfixes,
                                              csv_delimiter, csv_extension,
                                              image_id_extractor,
                                              image_files_extractor,
                                              object_position_extractor,
                                              well_extractor,
                                              treatment_extractor,
                                              replicate_extractor
           )
            current_num_of_images += num_of_images
            current_num_of_objects += num_of_objects

        elif os.path.isfile(tmp_file):

            file_base, file_ext = os.path.splitext(file)
            if file_ext == csv_extension:

                if file_base.endswith(image_file_postfix):

                    file_base_without_postfix = file_base[: -len(image_file_postfix)]

                    is_valid_csv_file = True
                    for object_file_postfix in object_file_postfixes:
                        filename = os.path.join(path, file_base_without_postfix + object_file_postfix + file_ext)
                        if not os.path.isfile(filename):
                            is_valid_csv_file = False
                            break

                    if is_valid_csv_file:

                        #print 'importing image file %s...' % tmp_file
                        image_data = read_cp2_csv_file(tmp_file, csv_delimiter)
                        (images,img_column_names,img_column_types) = image_data

                        object_data = []
                        for object_file_postfix in object_file_postfixes:
                            #print 'importing object file %s...' % object_file_postfix
                            object_file = os.path.join(path, file_base_without_postfix + object_file_postfix + file_ext)
                            object_data.append(read_cp2_csv_file(object_file))

                        # some consistency control
                        o0,ocn0,oct0 = object_data[0]
                        for o,ocn,oct in object_data:
                            if (len(o) != len(o0)) or (len(ocn) != len(oct)):
                                raise Exception('invalid objects input files')

                        fill_pdc(
                                pdc, working_dict,
                                image_data, object_data,
                                image_file_postfix, object_file_postfixes,
                                image_id_extractor,
                                image_files_extractor,
                                object_position_extractor,
                                well_extractor,
                                treatment_extractor,
                                replicate_extractor
                       )
                        #all_image_data.append((images,img_column_names,img_column_types))
                        #all_object_data.append(object_data)

                        current_num_of_images += len(images)
                        current_num_of_objects += len(o0)

    return current_num_of_images, current_num_of_objects

# Import results (CSV-files) as exported from CellProfiler2.
# Returns an yaca_data_container.
# Input parameters:
#   path: path in which to look for .csv files
#   images_file_postfix: postfix of the .csv files describing the images
#   object_file_postfixs: postfixes of the .csv files describing the objects

def import_cp2_csv_results(path, image_file_postfix, object_file_postfixes,
                           csv_delimiter=',', csv_extension='csv',
                           image_id_extractor=default_image_id_extractor,
                           image_files_extractor=default_image_files_extractor,
                           object_position_extractor=default_object_position_extractor,
                           well_extractor=default_well_extractor,
                           treatment_extractor=default_treatment_extractor,
                           replicate_extractor=default_replicate_extractor):

    print 'importing results'

    # create data container
    pdc = yaca_data_container()

    # recurse into all subfolders

    working_dict = {}

    num_of_images,num_of_objects = import_cp2_csv_results_recursive(path, pdc, working_dict,
                                                                     image_file_postfix, object_file_postfixes,
                                                                     csv_delimiter, csv_extension,
                                                                     image_id_extractor,
                                                                     image_files_extractor,
                                                                     object_position_extractor,
                                                                     well_extractor,
                                                                     treatment_extractor,
                                                                     replicate_extractor)

    del working_dict

    if (len(pdc.images) != num_of_images) or (len(pdc.objects) != num_of_objects):
        raise Exception('Something went wrong when importing the data')


    image_table_shape = list(pdc.imgFeatures.shape)
    if image_table_shape[0] > num_of_images:
        image_table_shape[0] = num_of_images
        pdc.imgFeatures.resize(image_table_shape)

    object_table_shape = list(pdc.objFeatures.shape)
    if object_table_shape[0] > num_of_objects:
        object_table_shape[0] = num_of_objects
        pdc.objFeatures.resize(object_table_shape)


    print 'files imported'

    print 'finished importing'

    return pdc





# Reads a CSV-file as exported from CellProfiler2.
# Returns (rows,column_names,column_types):
#   rows is the raw data from the CSV-file (excluding the header)
#   column_names has a name for each column
#   column_types has a type for each column (float or str)
# Input parameters:
#   file: an open file-descriptor or a filename
def read_cp2_csv_file(file, delimiter=','):

    close_file = False

    # open file if a filename was passed
    if type(file) == str:
        try:
            filename = file
            file = open(file,'rb')
            close_file = True
        except IOError, e:
            print 'ERROR: unable to open file %s: %s' % (filename, e)
            raise

    # use the CSV module to read the input file
    reader = csv.reader(file, delimiter=delimiter)

    try:

        # first we read the data from the file
        
        entities = []
        column_names = []
        found_header = False
        # we will use this to identify the type of each column
        column_types = []

        # this is used to keep the image name for several rows (in case it's not repeated)
        image = ''
        
        # read file
        for row in reader:

            if not found_header:
                # we haven't found the row with the column-descriptions yet
                if len(row[0].strip()) > 0:
                    # found it, write column-descriptions into columnIds
                    found_header = True
                    for name in row:
                        column_names.append(name)
                        column_types.append(float)
            
            else:
                # check types of columns
                for i in xrange(len(column_types)):
                    if column_types[i] == float:
                        try:
                            float(row[i])
                        except:
                            column_types[i] = str
                entities.append(row)

        return (entities,column_names,column_types)


    except csv.Error, e:
        print 'ERROR: file %s, line %d: %s' % (file.name, reader.line_num, e)
        raise

    # some cleanup
    finally:
        if close_file:
            file.close()
