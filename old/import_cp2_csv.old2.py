import csv
import numpy
import sys
import os
import traceback

from input_container import *



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



def import_cp2_csv_results_recursive(path, image_data, all_object_data,
                                     image_file_postfix, object_file_postfixes,
                                     csv_delimiter, csv_extension,
                                     image_id_extractor=default_image_id_extractor,
                                     image_files_extractor=default_image_files_extractor,
                                     object_position_extractor=default_object_position_extractor,
                                     treatment_extractor=default_treatment_extractor):

    print 'entering %s ...' % path

    total_num_of_images = 0
    total_num_of_objects = 0

    files = os.listdir( path )
    files.sort()

    for file in files:

        tmp_file = os.path.join( path, file )

        if os.path.isdir( tmp_file ):
            #print 'recursing into %s' % file
            tmp_path = os.path.join( path, file )
            num_of_images,num_of_objects = import_cp2_csv_results_recursive( tmp_file,
                                              image_data, all_object_data,
                                              image_file_postfix, object_file_postfixes,
                                              csv_delimiter, csv_extension,
                                              image_id_extractor,
                                              image_files_extractor,
                                              object_position_extractor,
                                              treatment_extractor)
            total_num_of_images += num_of_images
            total_num_of_objects += num_of_objects

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
                        (images,img_column_names,img_column_types) = read_cp2_csv_file( tmp_file, csv_delimiter )

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

                        image_data.append( ( images,img_column_names,img_column_types ) )
                        all_object_data.append( object_data )

                        total_num_of_images += len( images )
                        total_num_of_objects += len( o0 )

    return total_num_of_images,total_num_of_objects

# Import results (CSV-files) as exported from CellProfiler2.
# Returns an apc_data_container.
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

    # recurse into all subfolders

    image_data = []
    all_object_data = []

    num_of_images,num_of_objects = import_cp2_csv_results_recursive( path, image_data, all_object_data,
                                                                     image_file_postfix, object_file_postfixes,
                                                                     csv_delimiter, csv_extension,
                                                                     image_id_extractor,
                                                                     image_files_extractor,
                                                                     object_position_extractor,
                                                                     treatment_extractor )

    print 'files imported'

    # create data container
    adc = apc_data_container()

    # create object_prefixes
    objects_prefixes = list( object_file_postfixes )
    for i in xrange( len( objects_prefixes ) ):
        s = objects_prefixes[ i ]
        if s.startswith( '_' ):
            s = s[ 1 : ]
        if not s.endswith( '_' ):
            s = s + '_'
        objects_prefixes[ i ] = s

    # write meta information for images into adc
    img,icn,ict = image_data[0]
    for i in xrange( len( ict ) ):
        if ict[i] == float:
            n = len( adc.imgFeatureIds )
            adc.imgFeatureIds[ icn[i] ] = n


    object_column_types = []

    # combine meta-information about the objects into adc
    for k in xrange( len( all_object_data[0] ) ):

        object_column_types.append( all_object_data[0][ k ][2] )

        o,ocn,oct = all_object_data[0][ k ]
        for i in xrange( len(oct) ):
            if object_column_types[ k ][ i ] == float:
                if oct[i] == float:
                    n = len( adc.objFeatureIds )
                    adc.objFeatureIds[ objects_prefixes[k] + ocn[i] ] = n
                else:
                    object_column_types[ k ][i] = str

    # check if we have to provide a imageId-feature for the images
    img_image_id_feature_is_virtual = False
    if not adc.imgFeatureIds.has_key( IMAGE_ID_FEATURE_NAME ):
        adc.imgFeatureIds[ IMAGE_ID_FEATURE_NAME ] = len( adc.imgFeatureIds )
        adc.imgImageFeatureId = adc.imgFeatureIds[ IMAGE_ID_FEATURE_NAME ]
        img_image_id_feature_is_virtual = True

    # check if we have to provide a virtual treatmentId-feature for the images
    img_treatment_id_feature_is_virtual = False
    if not adc.imgFeatureIds.has_key( TREATMENT_ID_FEATURE_NAME ):
        adc.imgFeatureIds[ TREATMENT_ID_FEATURE_NAME ] = len( adc.imgFeatureIds )
        adc.imgTreatmentFeatureId = adc.imgFeatureIds[ TREATMENT_ID_FEATURE_NAME ]
        img_treatment_id_feature_is_virtual = True

    # check if we have to provide a virtual objectId-feature for the objects
    obj_object_id_feature_is_virtual = False
    if not adc.objFeatureIds.has_key( OBJECT_ID_FEATURE_NAME ):
        adc.objFeatureIds[ OBJECT_ID_FEATURE_NAME ] = len( adc.objFeatureIds )
        adc.objObjectFeatureId = adc.objFeatureIds[ OBJECT_ID_FEATURE_NAME ]
        obj_object_id_feature_is_virtual = True

    # check if we have to provide a virtual imageId-feature for the objects
    obj_image_id_feature_is_virtual = False
    if not adc.objFeatureIds.has_key( IMAGE_ID_FEATURE_NAME ):
        adc.objFeatureIds[ IMAGE_ID_FEATURE_NAME ] = len( adc.objFeatureIds )
        adc.objImageFeatureId = adc.objFeatureIds[ IMAGE_ID_FEATURE_NAME ]
        obj_image_id_feature_is_virtual = True

    # check if we have to provide a virtual treatmentId-feature for the objects
    obj_treatment_id_feature_is_virtual = False
    if not adc.objFeatureIds.has_key( TREATMENT_ID_FEATURE_NAME ):
        adc.objFeatureIds[ TREATMENT_ID_FEATURE_NAME ] = len( adc.objFeatureIds )
        adc.objTreatmentFeatureId = adc.objFeatureIds[ TREATMENT_ID_FEATURE_NAME ]
        obj_treatment_id_feature_is_virtual = True


    # create feature-tables
    adc.imgFeatures = numpy.empty( ( num_of_images, len( adc.imgFeatureIds ) ) )
    adc.objFeatures = numpy.empty( ( num_of_objects, len( adc.objFeatureIds ) ) )


    # fill apc_data_structure

    obj_rowId = 0
    img_rowId = 0
    
    for l in xrange( len( image_data ) ):

        images,image_column_names,image_column_types = image_data[ l ]
        object_data = all_object_data[ l ]
        o0,ocn,oct = object_data[0]

        for i in xrange( len( images ) ):
    
            img = apc_data_image()
            img.rowId = img_rowId
    
            try:
                treatment_name = treatment_extractor( images[i], image_column_names )
                if not adc.treatmentByName.has_key(treatment_name):
                    treatment = apc_data_treatment( treatment_name )
                    treatment.rowId = len( adc.treatments )
                    adc.treatmentByName[treatment_name] = len( adc.treatments )
                    adc.treatments.append( treatment )
                    img.treatment = treatment
                else:
                    img.treatment = adc.treatments[ adc.treatmentByName[ treatment_name ] ]
            except Exception,e:
                img.state = 'no_treatment'
                tb = "".join( traceback.format_tb( sys.exc_info()[2] ) )
                adc.errors.append( apc_data_error( e, tb, img ) )
                raise
    
            try:
                img.imageFiles = image_files_extractor( images[i], image_column_names )
            except Exception,e:
                img.state = 'no_image_files'
                tb = "".join( traceback.format_tb( sys.exc_info()[2] ) )
                adc.errors.append( apc_data_error( e, tb, img ) )
                raise
    
            n = 0
            for j in xrange( len( image_column_types ) ):
                if image_column_types[j] != float:
                    img.properties[ image_column_names[j] ] = images[i][j]
                else:
                    adc.imgFeatures[img.rowId][n] = float( images[i][j] )
                    n += 1
    
            if img_image_id_feature_is_virtual:
                adc.imgFeatures[img.rowId][ adc.imgImageFeatureId ] = img.rowId
    
            if img_treatment_id_feature_is_virtual:
                adc.imgFeatures[img.rowId][ adc.imgTreatmentFeatureId ] = img.treatment.rowId
    
            adc.images.append(img)
    
    
            for i in xrange( len(o0) ):
        
                obj = apc_data_object()
                obj.rowId = obj_rowId
        
                found_obj_position = False
        
                n = 0
                for k in xrange( len( object_data ) ):
        
                    o,ocn,oct = object_data[k]
        
                    try:
                        obj.position_x,obj.position_y = object_position_extractor( o[i], ocn )
                        found_obj_position = True
                    except:
                        pass
            
                    obj.image = adc.images[ img_rowId ]
        
                    for j in xrange( len( oct ) ):
                        if oct[j] == float:
                            adc.objFeatures[obj.rowId][n] = float( o[i][j] )
                            n += 1
                        #else:
                        #    obj.properties[ ocn[j] ] = o[i][j]
        
                if not found_obj_position:
                    obj.state = 'no_position'
                    e = Exception( 'Unable to extract object position' )
                    tb = "".join( traceback.format_tb( sys.exc_info()[2] ) )
                    adc.errors.append( apc_data_error( e, tb, img ) )
                    raise e
        
        
                if obj_object_id_feature_is_virtual:
                    adc.objFeatures[ obj.rowId , adc.objObjectFeatureId ] = obj.rowId
        
                if obj_image_id_feature_is_virtual:
                    adc.objFeatures[ obj.rowId , adc.objImageFeatureId ] = obj.image.rowId
        
                if obj_treatment_id_feature_is_virtual:
                    adc.objFeatures[ obj.rowId , adc.objTreatmentFeatureId ] = obj.image.treatment.rowId
        
                adc.objects.append(obj)
        
                obj_rowId += 1
    
            img_rowId += 1

    del image_data,all_object_data

    print 'finished importing'

    return adc





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
