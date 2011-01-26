import csv
import numpy
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
IMAGE_CHANNELS = ( ('imNucleus','R'), ('imProtein','G'), ('imCellMask','B'), ('SegC03','O1'), ('SegC01','O2') )
#IMAGE_CHANNELS = ( ('imNucleus','R'), ('imProtein','G'), ('imCellMask','B'), ('CellsObjects','O1'), ('NucleiObjects','O2') )
def default_image_files_extractor( row, column_names ):
    imageFiles = []
    for postfix,channel in IMAGE_CHANNELS:
        filename = None
        pathname = None
        for i in xrange( len( column_names ) ):
            if column_names[i] == (IMAGE_FILENAME_IDENTIFIER + postfix):
                filename = row[i]
                if pathname:
                    break
            elif column_names[i] == (IMAGE_PATHNAME_IDENTIFIER + postfix):
                pathname = row[i]
                if filename:
                    break
        if ( filename == None ) or ( pathname == None ):
            raise Exception( 'Unable to extract image filenames' )
        path = os.path.join(pathname, filename)
        imageFiles.append( ( path, channel ) )
    return imageFiles

OBJECT_POSITION_X_IDENTIFIER = 'Location_Center_X'
OBJECT_POSITION_Y_IDENTIFIER = 'Location_Center_Y'
def default_object_position_extractor( row, column_names ):
    xi = column_names.index(OBJECT_POSITION_X_IDENTIFIER)
    yi = column_names.index(OBJECT_POSITION_Y_IDENTIFIER)
    if xi < 0 or yi < 0:
        raise Exception( 'Unable to extract object position' )
    return float(row[xi]),float(row[yi])

TREATMENT_FEATURE_IDENTIFIER = 'PathName_'
def default_treatment_extractor( row, column_names ):
    treatment_name = None
    for i in xrange( len( column_names ) ):
        name = column_names[i]
        if name.startswith(TREATMENT_FEATURE_IDENTIFIER):
            # make sure we are windows-compatible
            v = row[i]
            v = v.replace('\\','/')
            #treatment_name = os.path.split( os.path.split(v)[0] )[-1]
            treatment_name = os.path.split( v )[-1]
            break
    if treatment_name == None:
        raise Exception( 'Unable to extract treatment identifier' )
    return treatment_name




# Import results (CSV-files) as exported from CellProfiler2.
# Returns an apc_data_container.
# Input parameters:
#   images_file: CSV-file describing the images
#   main_objects_file: CSV-file describing the objects (i.e. their coordinates)
#   objects_files: list of 2-tuples (prefix,file) of CSV-files describing the
#                   the objects. The first file is used as the main file
def import_cp2_csv_results(images_file, objects_files = [], delimiter=',',
                           image_id_extractor=default_image_id_extractor,
                           image_files_extractor=default_image_files_extractor,
                           object_position_extractor=default_object_position_extractor,
                           treatment_extractor=default_treatment_extractor):

    print 'importing results'

    # import files
    print 'importing image file...'
    (images,img_column_names,img_column_types) = read_cp2_csv_file(images_file, delimiter)
    objects_data = []
    objects_prefixes = []
    for prefix,filename in objects_files:
        print 'importing object file (prefix=%s)...' % prefix
        objects_prefixes.append(prefix)
        objects_data.append( read_cp2_csv_file(filename, delimiter) )

    # some consistency control
    o0,ocn0,oct0 = objects_data[0]
    for o,ocn,oct in objects_data:
        if ( len(o) != len(o0) ) or ( len(ocn) != len(oct) ):
            raise Exception( 'invalid objects input files' )

    # create data container
    adc = apc_data_container()

    # fill apc_data_structure

    print 'parsing image information...'

    # first the information about the images
    for i in xrange(len(img_column_types)):
        if img_column_types[i] == float:
            n = len(adc.imgFeatureIds)
            adc.imgFeatureIds[ img_column_names[i] ] = n

    image_id_feature_is_virtual = False
    if not adc.imgFeatureIds.has_key( IMAGE_ID_FEATURE_NAME ):
        adc.imgFeatureIds[ IMAGE_ID_FEATURE_NAME ] = len( adc.imgFeatureIds )
        adc.imgImageFeatureId = adc.imgFeatureIds[ IMAGE_ID_FEATURE_NAME ]
        image_id_feature_is_virtual = True

    treatment_id_feature_is_virtual = False
    if not adc.imgFeatureIds.has_key( TREATMENT_ID_FEATURE_NAME ):
        adc.imgFeatureIds[ TREATMENT_ID_FEATURE_NAME ] = len( adc.imgFeatureIds )
        adc.imgTreatmentFeatureId = adc.imgFeatureIds[ TREATMENT_ID_FEATURE_NAME ]
        treatment_id_feature_is_virtual = True

    adc.imgFeatures = numpy.empty( ( len(images), len(adc.imgFeatureIds) ) )

    print 'reading image information...'

    for i in xrange(len(images)):

        img = apc_data_image()
        img.rowId = i

        try:
            treatment_name = treatment_extractor( images[i], img_column_names )
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
            img.imageFiles = image_files_extractor( images[i], img_column_names )
        except Exception,e:
            img.state = 'no_image_files'
            adc.errors.append( apc_data_error( e, img ) )
            raise

        n = 0
        for j in xrange( len( img_column_types ) ):
            if img_column_types[j] != float:
                img.properties[ img_column_names[j] ] = images[i][j]
            else:
                adc.imgFeatures[i][n] = float( images[i][j] )
                n += 1

        if image_id_feature_is_virtual:
            adc.imgFeatures[i][ adc.imgImageFeatureId ] = img.rowId

        if treatment_id_feature_is_virtual:
            adc.imgFeatures[i][ adc.imgTreatmentFeatureId ] = img.treatment.rowId

        adc.images.append(img)


    """
    image_obj_ids = []
    for i in xrange( len( adc.images ) ):
        image_obj_ids.append( [] )
    """


    print 'parsing object information...'

    # then the information about the objects
    for k in xrange( len( objects_data ) ):
        o,ocn,oct = objects_data[k]
        for i in xrange( len(oct) ):
            if oct[i] == float:
                n = len( adc.objFeatureIds )
                adc.objFeatureIds[ objects_prefixes[k] + ocn[i] ] = n

    object_id_feature_is_virtual = False
    if not adc.objFeatureIds.has_key( OBJECT_ID_FEATURE_NAME ):
        adc.objFeatureIds[ OBJECT_ID_FEATURE_NAME ] = len( adc.objFeatureIds )
        adc.objObjectFeatureId = adc.objFeatureIds[ OBJECT_ID_FEATURE_NAME ]
        object_id_feature_is_virtual = True

    image_id_feature_is_virtual = False
    if not adc.objFeatureIds.has_key( IMAGE_ID_FEATURE_NAME ):
        adc.objFeatureIds[ IMAGE_ID_FEATURE_NAME ] = len( adc.objFeatureIds )
        adc.objImageFeatureId = adc.objFeatureIds[ IMAGE_ID_FEATURE_NAME ]
        image_id_feature_is_virtual = True

    treatment_id_feature_is_virtual = False
    if not adc.objFeatureIds.has_key( TREATMENT_ID_FEATURE_NAME ):
        adc.objFeatureIds[ TREATMENT_ID_FEATURE_NAME ] = len( adc.objFeatureIds )
        adc.objTreatmentFeatureId = adc.objFeatureIds[ TREATMENT_ID_FEATURE_NAME ]
        treatment_id_feature_is_virtual = True

    adc.objFeatures = numpy.empty( ( len(o0), len(adc.objFeatureIds) ) )

    print 'reading object information...'

    for i in xrange( len(o0) ):

        obj = apc_data_object()
        obj.rowId = i

        try:
            obj.position_x,obj.position_y = object_position_extractor( o0[i], ocn0 )
        except Exception,e:
            obj.state = 'no_position'
            adc.errors.append( apc_data_error( e, img ) )
            raise

        try:
            image_id = image_id_extractor( o0[i], ocn0 )
            obj.image = adc.images[image_id]
#            image_obj_ids[image_id].append(obj.rowId)
        except Exception,e:
            obj.state = 'no_image'
            adc.errors.append( apc_data_error( e, obj ) )
            raise

        n = 0
        for k in xrange( len( objects_data ) ):

            o,ocn,oct = objects_data[k]
            for j in xrange( len( oct ) ):
                if oct[j] == float:
                    adc.objFeatures[i][n] = float( o[i][j] )
                    n += 1
                #else:
                #    obj.properties[ ocn[j] ] = o[i][j]

        if object_id_feature_is_virtual:
            adc.objFeatures[ i , adc.objObjectFeatureId ] = obj.rowId

        if image_id_feature_is_virtual:
            adc.objFeatures[ i , adc.objImageFeatureId ] = obj.image.rowId

        if treatment_id_feature_is_virtual:
            adc.objFeatures[ i , adc.objTreatmentFeatureId ] = obj.image.treatment.rowId

        adc.objects.append(obj)


    """
    for i in xrange( len( adc.images ) ):
        adc.images[i].objIds = numpy.array( image_obj_ids[i] )
    """

    print 'creating ID lists...'

    """
    treatment_img_ids = []
    treatment_img_obj_ids = []

    for i in xrange( len( adc.treatments ) ):
        treatment_img_ids.append( [] )
        treatment_img_obj_ids.append( [] )

    for i in xrange( len( adc.images ) ):
        treatmentId = adc.images[i].treatment.rowId
        treatment_img_ids[treatmentId].append( i )
        treatment_img_obj_ids[treatmentId].extend( adc.images[i].objIds )

    for i in xrange( len( adc.treatments ) ):
        adc.treatments[i].imgIds = numpy.array( treatment_img_ids[i] )
        adc.treatments[i].objIds = numpy.array( treatment_img_obj_ids[i] )

    """

#    del image_obj_ids, treatment_img_ids, treatment_img_obj_ids,
    del images, img_column_names, img_column_types, objects_data, objects_prefixes

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

        print 'imported file %s' % file
    
        return (entities,column_names,column_types)


    except csv.Error, e:
        print 'ERROR: file %s, line %d: %s' % (file.name, reader.line_num, e)
        raise

    # some cleanup
    finally:
        if close_file:
            file.close()
