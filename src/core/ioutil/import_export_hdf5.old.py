#
# !!!!!!!!!!!!!!!!!! THIS IS NOT WORKING ATM !!!!!!!!!!!!!!!!!!!!!!
#

import numpy
import h5py


from input_container import *




def import_hdf5_results(file):

    try:

        f = h5py.File(file, 'r')

        print 'importing results from HDF5 file'
        root = f['APC_input']

        adc = apc_data_container()

        print 'importing treatment metadata...'
        treatments_dataset = root['treatments']
        for i in treatments_dataset.shape[0]:

            tr = apc_data_treatment( '' )
            tr.rowId = i
            tr.imgIds = []
            tr.objIds = []

            for k,id in treatments_dataset.attrs.iteritems():

                if k == 'name':
                    tr.name = treatments_dataset[i][id]
                elif k == 'state':
                    tr.state = treatments_dataset[i][id]

            tr.rowId = rowId
            if 'imgIds' in tr_group:
                tr.imgIds = numpy.empty( tr_group[ 'imgIds' ].shape )
                tr_group[ 'imgIds' ].read_direct( tr.imgIds )
            else:
                tr.imgIds = numpy.empty( (0,) )
            if 'objIds' in tr_group:
                tr.objIds = numpy.empty( tr_group[ 'objIds' ].shape )
                tr_group[ 'objIds' ].read_direct( tr.objIds )
            else:
                tr.objIds = numpy.empty( (0,) )
            adc.treatments.append( tr )

        print 'importing image metadata...'
        images_dataset = root['images']
        for i in images_dataset.shape[0]:

            img = apc_data_image()
            img.rowId = i
            img.objIds = []

            for k,id in images_dataset.attrs.iteritems():

                elif k == 'state':
                    img.state = images_dataset[i][id]

            tr.rowId = rowId
            if 'imgIds' in tr_group:
                tr.imgIds = numpy.empty( tr_group[ 'imgIds' ].shape )
                tr_group[ 'imgIds' ].read_direct( tr.imgIds )
            else:
                tr.imgIds = numpy.empty( (0,) )
            if 'objIds' in tr_group:
                tr.objIds = numpy.empty( tr_group[ 'objIds' ].shape )
                tr_group[ 'objIds' ].read_direct( tr.objIds )
            else:
                tr.objIds = numpy.empty( (0,) )
            adc.treatments.append( tr )


        for k,img_group in images_group.iteritems():
            rowId = int(k)
            attrs = img_group.attrs
            img = apc_data_image()
            img.rowId = rowId
            img.state = attrs['state']
            treatment = attrs['treatment']
            img.treatment = adc.treatments[ treatment ]
            if 'objIds' in img_group:
                tr.objIds = numpy.empty( tr_group[ 'objIds' ].shape )
                tr_group[ 'objIds' ].read_direct( tr.objIds )
            else:
                img.objIds = numpy.empty( (0,) )
            imagefiles_group = img_group['imageFiles']
            for filename,c in imagefiles_group.attrs.iteritems():
                img.imageFiles.append( ( filename, c ) )
            property_group = img_group['properties']
            for k,v in property_group.iteritems():
                img.properties[k] = v
            adc.images.append( img )

        print 'importing object metadata...'
        objects_group = root['objects']
        for k,obj_group in objects_group.iteritems():
            rowId = int(k)
            attrs = obj_group.attrs
            obj = apc_data_object()
            obj.rowId = rowId
            obj.state = attrs['state']
            image_id = attrs['image']
            obj.image = adc.images[ image_id ]
            obj.position_x = attrs[ 'position_x' ]
            obj.position_x = attrs[ 'position_y' ]
            property_group = obj_group['properties']
            for k,v in property_group.iteritems():
                obj.properties[k] = v
            adc.objects.append( obj )

        print 'importing error metadata...'
        errors_group = root['errors']
        for k,err_group in errors_group.iteritems():
            attrs = err_group.attrs
            err = apc_data_error(None, None, None)
            err.exception = Exception( attrs['message'] )
            err.traceback = attrs['traceback']
            if 'entityRowId' in attrs:
                entityRowId = attrs['entityRowId']
                if attrs['entityType'] == str( type( apc_data_image ) ):
                    err.entity = adc.images[ entityRowId ]
                elif attrs['entityType'] == str( type( apc_data_object ) ):
                    err.entity = adc.objects[ entityRowId ]
            adc.errors.append( err )

        print 'importing image features...'
        imgFeature_dataset = root['imgFeatures']
        adc.imgFeatures = numpy.empty( imgFeature_dataset.shape )
        imgFeature_dataset.read_direct( adc.imgFeatures )

        print 'importing image feature IDs...'
        attrs = imgFeature_dataset.attrs
        for k,v in imgFeature_dataset.attrs.iteritems():
            adc.imgFeatureIds[k] = v

        print 'importing object features...'
        objFeature_dataset = root['objFeatures']
        adc.objFeatures = numpy.empty( objFeature_dataset.shape )
        objFeature_dataset.read_direct( adc.objFeatures )

        print 'importing object feature IDs...'
        attrs = objFeature_dataset.attrs
        for k,v in objFeature_dataset.attrs.iteritems():
            adc.objFeatureIds[k] = v

        return adc

    finally:
        f.close()



def import_hdf5_results(file):

    try:

        f = h5py.File(file, 'r')

        print 'importing results from HDF5 file'
        root = f['APC_input']

        adc = apc_data_container()

        print 'importing treatment metadata...'
        treatments_group = root['treatments']
        for k,tr_group in treatments_group.iteritems():
            rowId = int(k)
            attrs =  tr_group.attrs
            tr = apc_data_treatment( attrs['name'] )
            tr.rowId = rowId
            if 'imgIds' in tr_group:
                tr.imgIds = numpy.empty( tr_group[ 'imgIds' ].shape )
                tr_group[ 'imgIds' ].read_direct( tr.imgIds )
            else:
                tr.imgIds = numpy.empty( (0,) )
            if 'objIds' in tr_group:
                tr.objIds = numpy.empty( tr_group[ 'objIds' ].shape )
                tr_group[ 'objIds' ].read_direct( tr.objIds )
            else:
                tr.objIds = numpy.empty( (0,) )
            adc.treatments.append( tr )

        objFeature_dataset = root['objFeatures']
        adc.objFeatures = numpy.empty( objFeature_dataset.shape )
        objFeature_dataset.read_direct( adc.objFeatures )

        print 'importing image metadata...'
        images_group = root['images']
        for k,img_group in images_group.iteritems():
            rowId = int(k)
            attrs = img_group.attrs
            img = apc_data_image()
            img.rowId = rowId
            img.state = attrs['state']
            treatment = attrs['treatment']
            img.treatment = adc.treatments[ treatment ]
            if 'objIds' in img_group:
                tr.objIds = numpy.empty( tr_group[ 'objIds' ].shape )
                tr_group[ 'objIds' ].read_direct( tr.objIds )
            else:
                img.objIds = numpy.empty( (0,) )
            imagefiles_group = img_group['imageFiles']
            for filename,c in imagefiles_group.attrs.iteritems():
                img.imageFiles.append( ( filename, c ) )
            property_group = img_group['properties']
            for k,v in property_group.iteritems():
                img.properties[k] = v
            adc.images.append( img )

        print 'importing object metadata...'
        objects_group = root['objects']
        for k,obj_group in objects_group.iteritems():
            rowId = int(k)
            attrs = obj_group.attrs
            obj = apc_data_object()
            obj.rowId = rowId
            obj.state = attrs['state']
            image_id = attrs['image']
            obj.image = adc.images[ image_id ]
            obj.position_x = attrs[ 'position_x' ]
            obj.position_x = attrs[ 'position_y' ]
            property_group = obj_group['properties']
            for k,v in property_group.iteritems():
                obj.properties[k] = v
            adc.objects.append( obj )

        print 'importing error metadata...'
        errors_group = root['errors']
        for k,err_group in errors_group.iteritems():
            attrs = err_group.attrs
            err = apc_data_error(None, None, None)
            err.exception = Exception( attrs['message'] )
            err.traceback = attrs['traceback']
            if 'entityRowId' in attrs:
                entityRowId = attrs['entityRowId']
                if attrs['entityType'] == str( type( apc_data_image ) ):
                    err.entity = adc.images[ entityRowId ]
                elif attrs['entityType'] == str( type( apc_data_object ) ):
                    err.entity = adc.objects[ entityRowId ]
            adc.errors.append( err )

        print 'importing image features...'
        imgFeature_dataset = root['imgFeatures']
        adc.imgFeatures = numpy.empty( imgFeature_dataset.shape )
        imgFeature_dataset.read_direct( adc.imgFeatures )

        print 'importing image feature IDs...'
        attrs = imgFeature_dataset.attrs
        for k,v in imgFeature_dataset.attrs.iteritems():
            adc.imgFeatureIds[k] = v

        print 'importing object features...'
        objFeature_dataset = root['objFeatures']
        adc.objFeatures = numpy.empty( objFeature_dataset.shape )
        objFeature_dataset.read_direct( adc.objFeatures )

        print 'importing object feature IDs...'
        attrs = objFeature_dataset.attrs
        for k,v in objFeature_dataset.attrs.iteritems():
            adc.objFeatureIds[k] = v

        return adc

    finally:
        f.close()




def export_hdf5_results(file, adc):

    f = None
    try:

        f = h5py.File(file, 'w')

        print 'exporting results to HDF5 file'
        root = f.create_group('APC_input')

        print 'exporting image metadata...'
        NATIVE_PROPERTY_IDS = { 'state':0, 'treatment':1 }
        dt = h5py.new_vlen( str )
        images_dataset = root.create_dataset(
            'images',
            ( len(adc.objects), len(NATIVE_PROPERTY_IDS) + len(adc.images[0].properties) ),
            dtype=dt
        )
        for i in xrange( len( adc.images ) ):
            img = adc.images[i]
            images_dataset[i][0] = img.state
            images_dataset[i][1] = img.treatment.rowId
            n = len(NATIVE_PROPERTY_IDS)
            for k,v in adc.images[i].properties.iteritems():
                images_dataset[i][n] = v
                n += 1
        attrs = images_dataset.attrs
        n = 0
        for k,v in NATIVE_PROPERTY_IDS.iteritems():
            attrs[k] = n
            n += 1
        for k,v in adc.images[0].properties.iteritems():
            attrs[k] = n
            n += 1

        ##print 'exporting image-object links...'
        ##images_group = root.create_group('images')

        """print 'exporting image metadata...'
        images_group = root.create_group('images')
        for i in xrange( len( adc.images ) ):
            img_group = images_group.create_group( '%d' % adc.images[i].rowId )
            attrs = img_group.attrs
            attrs['state'] = adc.images[i].state
            attrs['treatment'] = adc.images[i].treatment.rowId
            if len( adc.images[i].objIds ) > 0:
                img_group.create_dataset( 'objIds', data=adc.images[i].objIds )
            imagefiles_group = img_group.create_group('imageFiles')
            for filename,c in adc.images[i].imageFiles:
                imagefiles_group.attrs[filename] = c
            property_group = img_group.create_group('properties')
            for k,v in adc.images[i].properties.iteritems():
                property_group.attrs[k] = v"""

        print 'exporting object metadata...'
        OBJECT_PROPERTY_OFFSET = 4
        NATIVE_PROPERTY_IDS = { 'state':0, 'image':1, 'position_x':2, 'position_y':3 }
        dt = h5py.new_vlen( str )
        objects_dataset = root.create_dataset(
            'objects',
            ( len(adc.objects), len(NATIVE_PROPERTY_IDS) + len(adc.objects[0].properties) ),
            dtype=dt
        )
        for i in xrange( len( adc.objects ) ):
            obj = adc.objects[i]
            objects_dataset[i][0] = obj.state
            objects_dataset[i][1] = obj.image.rowId
            objects_dataset[i][2] = obj.position_x
            objects_dataset[i][3] = obj.position_y
            n = len(NATIVE_PROPERTY_IDS)
            for k,v in adc.objects[i].properties.iteritems():
                objects_dataset[i][n] = v
                n += 1
        attrs = objects_dataset.attrs
        n = 0
        for k,v in NATIVE_PROPERTY_IDS.iteritems():
            attrs[k] = n
            n += 1
        for k,v in adc.objects[0].properties.iteritems():
            attrs[k] = n
            n += 1

        """objects_group = root.create_group('objects')
        for i in xrange( len( adc.objects ) ):
            obj_group = objects_group.create_group( '%d' % adc.objects[i].rowId )
            attrs = obj_group.attrs
            attrs['state'] = adc.objects[i].state
            attrs['image'] = adc.objects[i].image.rowId
            attrs['position_x'] = adc.objects[i].position_x
            attrs['position_y'] = adc.objects[i].position_y
            property_group = obj_group.create_group('properties')
            for k,v in adc.objects[i].properties.iteritems():
                property_group.attrs[k] = v"""

        print 'exporting treatment metadata...'
        NATIVE_PROPERTY_IDS = { 'name':0 }
        dt = h5py.new_vlen( str )
        treatments_dataset = root.create_dataset(
            'treatments',
            ( len(adc.objects), len(NATIVE_PROPERTY_IDS) ),
            dtype=dt
        )
        for i in xrange( len( adc.treatments ) ):
            tr = adc.treatments[i]
            treatments_dataset[i][0] = tr.name
        attrs = treatments_dataset.attrs
        n = 0
        for k,v in NATIVE_PROPERTY_IDS.iteritems():
            attrs[k] = n
            n += 1

        """print 'exporting treatment metadata...'
        treatments_group = root.create_group('treatments')
        for i in xrange( len( adc.treatments ) ):
            tr_group = treatments_group.create_group( '%d' % adc.treatments[i].rowId )
            attrs = tr_group.attrs
            attrs['name'] = adc.treatments[i].name
            if len( adc.treatments[i].imgIds ) > 0:
                tr_group.create_dataset( 'imgIds', data=adc.treatments[i].imgIds )
            if len( adc.treatments[i].objIds ) > 0:
                tr_group.create_dataset( 'objIds', data=adc.treatments[i].objIds )"""

        print 'exporting error metadata...'
        NATIVE_PROPERTY_IDS = { 'message':0, 'traceback':1, 'entityType':2, 'entityRowId':3 }
        dt = h5py.new_vlen( str )
        errors_dataset = root.create_dataset(
            'errors',
            ( len(adc.objects), len(NATIVE_PROPERTY_IDS) ),
            dtype=dt
        )
        for i in xrange( len( adc.errors ) ):
            err = adc.treatments[i]
            errors_dataset[i][0] = str( err.exception )
            errors_dataset[i][1] = str( err.traceback )
            errors_dataset[i][2] = str( type( err.entity ) )
            try:
                attrs[3] = err.rowId
            except:
                attrs[3] = ''
        attrs = errors_dataset.attrs
        n = 0
        for k,v in NATIVE_PROPERTY_IDS.iteritems():
            attrs[k] = n
            n += 1

        """print 'exporting error metadata...'
        errors_group = root.create_group('errors')
        for i in xrange( len( adc.errors ) ):
            err_group = treatments_group.create_group( '%d' % i )
            attrs = err_group.attrs
            attrs['message'] = str( adc.errors[i].exception )
            attrs['traceback'] = str( adc.errors[i].traceback )
            attrs['entityType'] = str( type( adc.errors[i].entity ) )
            try:
                attrs['entityRowId'] = adc.errors[i].rowId
            except:
                pass"""

        print 'exporting image features...'
        imgFeature_dataset = root.create_dataset('imgFeatures', data=adc.imgFeatures)
        print 'exporting image feature IDs...'
        for k,v in adc.imgFeatureIds.iteritems():
            imgFeature_dataset.attrs[k] = v

        print 'exporting object features...'
        objFeature_dataset = root.create_dataset('objFeatures', data=adc.objFeatures)
        print 'exporting object feature IDs...'
        for k,v in adc.objFeatureIds.iteritems():
            objFeature_dataset.attrs[k] = v

    finally:
        if f:
            f.close()
