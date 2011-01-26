import csv
import numpy
import sys
import os
import traceback

from data_container import *



OBJECT_IMAGE_ID_IDENTIFIER = 'ImageNumber'
def default_image_id_extractor( row, column_names ):
    i = column_names.index(OBJECT_IMAGE_ID_IDENTIFIER)
    if i < 0:
        raise Exception( 'Unable to extract image id' )
    return int(row[i]) - 1

IMAGE_FILENAME_IDENTIFIER = 'FileName_'
IMAGE_PATHNAME_IDENTIFIER = 'PathName_'
#IMAGE_CHANNELS = ( ('imNucleus','R'), ('imProtein','G'), ('imCellMask','B'), ('SegC03','O1'), ('SegC01','O2') )
#IMAGE_CHANNELS = ( ('imNucleus','R'), ('imProtein','G'), ('imCellMask','B'), ('CellsObjects','O1'), ('NucleiObjects','O2') )
def default_image_files_extractor( row, column_names ):

    filenames = {}
    paths = {}

    for i in xrange( len( column_names ) ):
        if column_names[i].startswith( IMAGE_FILENAME_IDENTIFIER ):
            entity_name = column_names[i][ len( IMAGE_FILENAME_IDENTIFIER ) : ]
            filenames[ entity_name ] = row[i]
        elif column_names[i].startswith( IMAGE_PATHNAME_IDENTIFIER ):
            entity_name = column_names[i][ len( IMAGE_PATHNAME_IDENTIFIER ) : ]
            paths[ entity_name ] = row[i]

    imageFiles = []

    for entity_name,filename in filenames.iteritems():

        if entity_name in paths:

            path = paths[ entity_name ]
            full_path = os.path.join( path, filename )

            imageFiles.append( ( entity_name, full_path ) )

    return imageFiles

OBJECT_POSITION_X_IDENTIFIER = 'Location_Center_X'
OBJECT_POSITION_Y_IDENTIFIER = 'Location_Center_Y'
def default_object_position_extractor( row, column_names ):
    xi = column_names.index(OBJECT_POSITION_X_IDENTIFIER)
    yi = column_names.index(OBJECT_POSITION_Y_IDENTIFIER)
    if xi < 0 or yi < 0:
        raise Exception( 'Unable to extract object position' )
    return float(row[xi]),float(row[yi])

TREATMENT_FEATURE_IDENTIFIER = 'Metadata_RelativeImagePath'
def default_treatment_extractor( row, column_names ):
    treatment_name = None
    for i in xrange( len( column_names ) ):
        name = column_names[i]
        if name == TREATMENT_FEATURE_IDENTIFIER:
            # make sure we are windows-compatible
            v = row[i]
            v = v.replace('\\','/')
            #treatment_name = os.path.split( os.path.split(v)[0] )[-1]
            treatment_name = os.path.split( v )[-1]
            break
    if treatment_name == None:
        raise Exception( 'Unable to extract treatment identifier' )
    return treatment_name



def init_pdc(pdc, working_dict, image_data, object_data, image_file_postfix, object_file_postfixes):

    # create object_prefixes
    objects_prefixes = list( object_file_postfixes )
    for i in xrange( len( objects_prefixes ) ):
        s = objects_prefixes[ i ]
        if s.startswith( '_' ):
            s = s[ 1 : ]
        if not s.endswith( '_' ):
            s = s + '_'
        objects_prefixes[ i ] = s

    # write meta information for images into pdc
    img,icn,ict = image_data
    for i in xrange( len( ict ) ):
        if ict[i] == float:
            n = len( pdc.imgFeatureIds )
            pdc.imgFeatureIds[ icn[i] ] = n


    object_column_types = []

    # combine meta-information about the objects into pdc
    for k in xrange( len( object_data ) ):

        object_column_types.append( object_data[ k ][2] )

        o,ocn,oct = object_data[ k ]
        for i in xrange( len(oct) ):
            if object_column_types[ k ][ i ] == float:
                if oct[i] == float:
                    n = len( pdc.objFeatureIds )
                    pdc.objFeatureIds[ objects_prefixes[k] + ocn[i] ] = n
                else:
                    object_column_types[ k ][i] = str

    working_dict[ 'object_column_types' ] = object_column_types

    # check if we have to provide a imageId-feature for the images
    working_dict[ 'img_image_id_feature_is_virtual' ] = False
    if not pdc.imgFeatureIds.has_key( IMAGE_ID_FEATURE_NAME ):
        pdc.imgFeatureIds[ IMAGE_ID_FEATURE_NAME ] = len( pdc.imgFeatureIds )
        pdc.imgImageFeatureId = pdc.imgFeatureIds[ IMAGE_ID_FEATURE_NAME ]
        working_dict[ 'img_image_id_feature_is_virtual' ] = True

    # check if we have to provide a virtual treatmentId-feature for the images
    working_dict[ 'img_treatment_id_feature_is_virtual' ] = False
    if not pdc.imgFeatureIds.has_key( TREATMENT_ID_FEATURE_NAME ):
        pdc.imgFeatureIds[ TREATMENT_ID_FEATURE_NAME ] = len( pdc.imgFeatureIds )
        pdc.imgTreatmentFeatureId = pdc.imgFeatureIds[ TREATMENT_ID_FEATURE_NAME ]
        working_dict[ 'img_treatment_id_feature_is_virtual' ] = True

    # check if we have to provide a virtual objectId-feature for the objects
    working_dict[ 'obj_object_id_feature_is_virtual' ] = False
    if not pdc.objFeatureIds.has_key( OBJECT_ID_FEATURE_NAME ):
        pdc.objFeatureIds[ OBJECT_ID_FEATURE_NAME ] = len( pdc.objFeatureIds )
        pdc.objObjectFeatureId = pdc.objFeatureIds[ OBJECT_ID_FEATURE_NAME ]
        working_dict[ 'obj_object_id_feature_is_virtual' ] = True

    # check if we have to provide a virtual imageId-feature for the objects
    working_dict[ 'obj_image_id_feature_is_virtual' ] = False
    if not pdc.objFeatureIds.has_key( IMAGE_ID_FEATURE_NAME ):
        pdc.objFeatureIds[ IMAGE_ID_FEATURE_NAME ] = len( pdc.objFeatureIds )
        pdc.objImageFeatureId = pdc.objFeatureIds[ IMAGE_ID_FEATURE_NAME ]
        working_dict[ 'obj_image_id_feature_is_virtual' ] = True

    # check if we have to provide a virtual treatmentId-feature for the objects
    working_dict[ 'obj_treatment_id_feature_is_virtual' ] = False
    if not pdc.objFeatureIds.has_key( TREATMENT_ID_FEATURE_NAME ):
        pdc.objFeatureIds[ TREATMENT_ID_FEATURE_NAME ] = len( pdc.objFeatureIds )
        pdc.objTreatmentFeatureId = pdc.objFeatureIds[ TREATMENT_ID_FEATURE_NAME ]
        working_dict[ 'obj_treatment_id_feature_is_virtual' ] = True



IMAGE_ARRAY_BLOCKSIZE = 8
OBJECT_ARRAY_BLOCKSIZE = 8 * 256

def update_pdc(pdc, image_data, object_data):

    # if necessary, create feature-tables

    if pdc.imgFeatures == None:
        pdc.imgFeatures = numpy.empty( ( 0 , len( pdc.imgFeatureIds ) ) )

    if pdc.objFeatures == None:
        pdc.objFeatures = numpy.empty( ( 0 , len( pdc.objFeatureIds ) ) )

    # if necessary, update size of feature-tables

    num_of_new_images = len( image_data[0] )
    image_table_shape = list( pdc.imgFeatures.shape )
    if image_table_shape[0] < ( len( pdc.images ) + num_of_new_images ):
        image_table_shape[0] += max( num_of_new_images, IMAGE_ARRAY_BLOCKSIZE )
        pdc.imgFeatures.resize( image_table_shape )

    num_of_new_objects = len( object_data[0][0] )
    object_table_shape = list( pdc.objFeatures.shape )
    if object_table_shape[0] < ( len( pdc.objects ) + num_of_new_objects ):
        object_table_shape[0] += max( num_of_new_objects, OBJECT_ARRAY_BLOCKSIZE )
        pdc.objFeatures.resize( object_table_shape )



def fill_pdc(pdc, working_dict, image_data, object_data, image_file_postfix, object_file_postfixes,
             image_id_extractor, image_files_extractor, object_position_extractor, treatment_extractor):

    if len( pdc.images ) <= 0:
        init_pdc( pdc, working_dict, image_data, object_data, image_file_postfix, object_file_postfixes )

    # update feature-tables
    update_pdc( pdc, image_data, object_data )

    object_column_types = working_dict[ 'object_column_types' ]


    # fill phenonice_data_structure

    images,image_column_names,image_column_types = image_data

    o0,ocn,oct = object_data[0]

    old_num_of_images = len( pdc.images )

    for i in xrange( len( images ) ):

        img = phenonice_data_image()
        img.rowId = len( pdc.images )

        try:
            treatment_name = treatment_extractor( images[i], image_column_names )
            if not pdc.treatmentByName.has_key(treatment_name):
                treatment = phenonice_data_treatment( treatment_name )
                treatment.rowId = len( pdc.treatments )
                pdc.treatmentByName[treatment_name] = len( pdc.treatments )
                pdc.treatments.append( treatment )
                img.treatment = treatment
            else:
                img.treatment = pdc.treatments[ pdc.treatmentByName[ treatment_name ] ]
        except Exception,e:
            img.state = 'no_treatment'
            tb = "".join( traceback.format_tb( sys.exc_info()[2] ) )
            pdc.errors.append( phenonice_data_error( e, tb, img ) )
            raise

        try:
            img.imageFiles = image_files_extractor( images[i], image_column_names )
        except Exception,e:
            img.state = 'no_image_files'
            tb = "".join( traceback.format_tb( sys.exc_info()[2] ) )
            pdc.errors.append( phenonice_data_error( e, tb, img ) )
            raise

        n = 0
        for j in xrange( len( image_column_types ) ):
            if image_column_types[j] != float:
                img.properties[ image_column_names[j] ] = images[i][j]
            else:
                v1 = images[i]
                v2 = float( v1[j] )
                q = pdc.imgFeatures[img.rowId]
                pdc.imgFeatures[img.rowId][n] = v2
                #pdc.imgFeatures[img.rowId][n] = float( images[i][j] )
                n += 1

        if working_dict[ 'img_image_id_feature_is_virtual' ]:
            pdc.imgFeatures[img.rowId][ pdc.imgImageFeatureId ] = img.rowId

        if working_dict[ 'img_treatment_id_feature_is_virtual' ]:
            pdc.imgFeatures[img.rowId][ pdc.imgTreatmentFeatureId ] = img.treatment.rowId

        pdc.images.append(img)

    for i in xrange( len(o0) ):

        obj = phenonice_data_object()
        obj.rowId = len( pdc.objects )

        found_img_id = False
        found_obj_position = False

        n = 0
        for k in xrange( len( object_data ) ):

            o,ocn,oct = object_data[k]

            try:
                image_id = image_id_extractor( o[i], ocn ) + old_num_of_images
                obj.image = pdc.images[ image_id ]
                found_img_id = True
            except:
                pass

            try:
                obj.position_x,obj.position_y = object_position_extractor( o[i], ocn )
                found_obj_position = True
            except:
                pass

            for j in xrange( len( oct ) ):
                if object_column_types[ k ][ j ] == float:
                    pdc.objFeatures[obj.rowId][n] = float( o[i][j] )
                    n += 1
                #else:
                #    obj.properties[ ocn[j] ] = o[i][j]

        if not found_img_id:
            obj.state = 'no image'
            e = Exception( 'Unable to extract image id' )
            tb = "".join( traceback.format_tb( sys.exc_info()[2] ) )
            pdc.errors.append( phenonice_data_error( e, tb, obj ) )
            raise e
        if not found_obj_position:
            obj.state = 'no_position'
            e = Exception( 'Unable to extract object position' )
            tb = "".join( traceback.format_tb( sys.exc_info()[2] ) )
            pdc.errors.append( phenonice_data_error( e, tb, img ) )
            raise e


        if working_dict[ 'obj_object_id_feature_is_virtual' ]:
            pdc.objFeatures[ obj.rowId , pdc.objObjectFeatureId ] = obj.rowId

        if working_dict[ 'obj_image_id_feature_is_virtual' ]:
            pdc.objFeatures[ obj.rowId , pdc.objImageFeatureId ] = obj.image.rowId

        if working_dict[ 'obj_treatment_id_feature_is_virtual' ]:
            pdc.objFeatures[ obj.rowId , pdc.objTreatmentFeatureId ] = obj.image.treatment.rowId

        pdc.objects.append(obj)



def import_cp2_csv_results_recursive(path, pdc, working_dict, image_file_postfix, object_file_postfixes,
                                     csv_delimiter, csv_extension,
                                     image_id_extractor=default_image_id_extractor,
                                     image_files_extractor=default_image_files_extractor,
                                     object_position_extractor=default_object_position_extractor,
                                     treatment_extractor=default_treatment_extractor):

    print 'entering %s ...' % path

    current_num_of_images = 0
    current_num_of_objects = 0

    files = os.listdir( path )
    files.sort()

    for file in files:

        tmp_file = os.path.join( path, file )

        if os.path.isdir( tmp_file ):
            #print 'recursing into %s' % file
            tmp_path = os.path.join( path, file )
            num_of_images,num_of_objects = import_cp2_csv_results_recursive(
                                              tmp_file, pdc, working_dict,
                                              image_file_postfix, object_file_postfixes,
                                              csv_delimiter, csv_extension,
                                              image_id_extractor,
                                              image_files_extractor,
                                              object_position_extractor,
                                              treatment_extractor
            )
            current_num_of_images += num_of_images
            current_num_of_objects += num_of_objects

        elif os.path.isfile( tmp_file ):

            file_base, file_ext = os.path.splitext( file )
            if file_ext == csv_extension:

                if file_base.endswith( image_file_postfix ):

                    file_base_without_postfix = file_base[ : -len( image_file_postfix )]

                    is_valid_csv_file = True
                    for object_file_postfix in object_file_postfixes:
                        filename = os.path.join( path, file_base_without_postfix + object_file_postfix + file_ext )
                        if not os.path.isfile( filename ):
                            is_valid_csv_file = False
                            break

                    if is_valid_csv_file:

                        #print 'importing image file %s...' % tmp_file
                        image_data = read_cp2_csv_file( tmp_file, csv_delimiter )
                        (images,img_column_names,img_column_types) = image_data

                        object_data = []
                        for object_file_postfix in object_file_postfixes:
                            #print 'importing object file %s...' % object_file_postfix
                            object_file = os.path.join( path, file_base_without_postfix + object_file_postfix + file_ext )
                            object_data.append( read_cp2_csv_file( object_file ) )

                        # some consistency control
                        o0,ocn0,oct0 = object_data[0]
                        for o,ocn,oct in object_data:
                            if ( len(o) != len(o0) ) or ( len(ocn) != len(oct) ):
                                raise Exception( 'invalid objects input files' )

                        fill_pdc(
                                pdc, working_dict,
                                image_data, object_data,
                                image_file_postfix, object_file_postfixes,
                                image_id_extractor,
                                image_files_extractor,
                                object_position_extractor,
                                treatment_extractor
                        )
                        #all_image_data.append( ( images,img_column_names,img_column_types ) )
                        #all_object_data.append( object_data )

                        current_num_of_images += len( images )
                        current_num_of_objects += len( o0 )

    return current_num_of_images, current_num_of_objects

# Import results (CSV-files) as exported from CellProfiler2.
# Returns an phenonice_data_container.
# Input parameters:
#   path: path in which to look for .csv files
#   images_file_postfix: postfix of the .csv files describing the images
#   object_file_postfixs: postfixes of the .csv files describing the objects

def import_cp2_csv_results(path, image_file_postfix, object_file_postfixes,
                           csv_delimiter=',', csv_extension='csv',
                           image_id_extractor=default_image_id_extractor,
                           image_files_extractor=default_image_files_extractor,
                           object_position_extractor=default_object_position_extractor,
                           treatment_extractor=default_treatment_extractor):

    print 'importing results'

    # create data container
    pdc = phenonice_data_container()

    # recurse into all subfolders

    working_dict = {}

    num_of_images,num_of_objects = import_cp2_csv_results_recursive( path, pdc, working_dict,
                                                                     image_file_postfix, object_file_postfixes,
                                                                     csv_delimiter, csv_extension,
                                                                     image_id_extractor,
                                                                     image_files_extractor,
                                                                     object_position_extractor,
                                                                     treatment_extractor )

    del working_dict

    if ( len( pdc.images ) != num_of_images ) or ( len( pdc.objects ) != num_of_objects ):
        raise Exception( 'Something went wrong when importing the data' )


    image_table_shape = list( pdc.imgFeatures.shape )
    if image_table_shape[0] > num_of_images:
        image_table_shape[0] = num_of_images
        pdc.imgFeatures.resize( image_table_shape )

    object_table_shape = list( pdc.objFeatures.shape )
    if object_table_shape[0] > num_of_objects:
        object_table_shape[0] = num_of_objects
        pdc.objFeatures.resize( object_table_shape )


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
