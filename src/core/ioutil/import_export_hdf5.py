# -*- coding: utf-8 -*-

"""
import_export_hdf5.py -- Importer and exporter for HDF5 files.
"""

# This software is distributed under the FreeBSD License.
# See the accompanying file LICENSE for details.
# 
# Copyright 2011 Benjamin Hepp

import numpy
import struct
import cPickle
import cStringIO
import h5py
import scipy.io

from data_container import *

def import_hdf5_results(file, further_files=[]):

    print 'importing results from HDF5 file...'

    pdcs = []

    all_files = [ file ]
    all_files.extend( further_files )

    files = []
    for filename in all_files:

        print 'opening hdf5 file:', filename

        f = None
        try:

            f = h5py.File( filename, mode='r' )
            #print 'f:', f, list( f )
            root = f[ 'YACA_input' ]
            #print 'root:', root, list( root )

            #print 'attrs:', root.attrs, list( root.attrs )

            version = root.attrs[ '__version__' ]

            print 'YACA HDF5 version:', version

            #print 'reading pickle stream ...'
            pickleStream_dataset = root[ 'pickleStream' ]
            dt = numpy.uint8
            stream_len = pickleStream_dataset.shape[0]
            pickleStr = ''.join(
                struct.unpack( '%dc' % stream_len, pickleStream_dataset[:] )
            )

            sio = cStringIO.StringIO( pickleStr )
            up = cPickle.Unpickler( sio )
            print 'unpickling object...'
            pdc_container = up.load()
            pdc = yaca_data_container.import_container( pdc_container )
            #pdc = up.load()
            sio.close()

            print 'reading image features table...'
            imgFeature_dataset = root['imgFeatures']
            pdc.imgFeatures = numpy.empty( imgFeature_dataset.shape )
            imgFeature_dataset.read_direct( pdc.imgFeatures )
            print 'imgFeatures.shape:', pdc.imgFeatures.shape

            print 'reading object features table...'
            objFeature_dataset = root['objFeatures']
            pdc.objFeatures = numpy.empty( objFeature_dataset.shape )
            objFeature_dataset.read_direct( pdc.objFeatures )
            print 'objFeatures.shape:', pdc.objFeatures.shape

            """
            print 'reading mat...'
            d = scipy.io.loadmat(file + '.mat')
            pdc.objFeatures = d['objFeatures']
            pdc.imgFeatures = d['imgFeatures']
            """

            pdcs.append( pdc )

        finally:
            if f:
                print 'closing hdf5 file'
                files.append( f )
                #f.close()

    for f in files:
        f.close()

    print 'imported data from HDF5 file'

    if len( pdcs ) > 1:

        num_of_images = 0
        num_of_objects = 0

        for pdc in pdcs:
            num_of_images += len( pdc.images )
            num_of_objects += len( pdc.objects )

        pdc = pdcs[ 0 ]

        imgFeatures = numpy.empty( ( num_of_images, pdc.imgFeatures.shape[1] ), dtype=float )
        objFeatures = numpy.empty( ( num_of_objects, pdc.objFeatures.shape[1] ), dtype=float )

        images = []
        objects = []
        wells = []
        treatments = []
        replicates = []
        errors = []

        wellIndexToPdcMapping = []
        treatmentIndexToPdcMapping = []
        replicateIndexToPdcMapping = []

        imgOffset = 0
        objOffset = 0
        wellOffset = 0
        treatmentOffset = 0
        replicateOffset = 0

        imgFeatureNames = []
        for i in xrange( pdcs[0].imgFeatures.shape[1] ):
            imgFeatureNames.append( None )
        for fname,fid in pdcs[0].imgFeatureIds.iteritems():
            imgFeatureNames[ fid ] = fname

        objFeatureNames = []
        for i in xrange( pdcs[0].objFeatures.shape[1] ):
            objFeatureNames.append( None )
        for fname,fid in pdcs[0].objFeatureIds.iteritems():
            objFeatureNames[ fid ] = fname

        for pdcIndex in xrange( len( pdcs ) ):

            pdc = pdcs[ pdcIndex ]

            tmp = []
            for i in xrange( pdc.imgFeatures.shape[1] ):
                tmp.append( None )
            for fname,fid in pdc.imgFeatureIds.iteritems():
                tmp[ fid ] = fname
            tmp.sort()

            imgFeatureMapping = []
            for i in xrange( len( imgFeatureNames ) ):
                imgFeatureMapping.append( pdc.imgFeatureIds[ imgFeatureNames[ i ] ] )
            for img in pdc.images:
                img.index += imgOffset
                images.append( img )
            imgFeatures[ imgOffset : imgOffset + pdc.imgFeatures.shape[0] ] = pdc.imgFeatures[ : , imgFeatureMapping ]
            objFeatureMapping = []
            for i in xrange( len( objFeatureNames ) ):
                objFeatureMapping.append( pdc.objFeatureIds[ objFeatureNames[ i ] ] )
            for obj in pdc.objects:
                obj.index += objOffset
                objects.append( obj )
            objFeatures[ objOffset : objOffset + pdc.objFeatures.shape[0] ] = pdc.objFeatures[ : , objFeatureMapping ]
            for well in pdc.wells:
                well.index += wellOffset
                wells.append( well )
                wellIndexToPdcMapping.append( pdcIndex )
            objFeatures[ objOffset : objOffset + pdc.objFeatures.shape[0], pdc.objWellFeatureId ] += wellOffset
            imgFeatures[ imgOffset : imgOffset + pdc.imgFeatures.shape[0], pdc.imgWellFeatureId ] += wellOffset
            for treatment in pdc.treatments:
                if treatment.name.startswith("wt"):
                    treatment.name = treatment.name.replace("wt", "mock")
                treatment.index += treatmentOffset
                treatments.append( treatment )
                treatmentIndexToPdcMapping.append( pdcIndex )
            objFeatures[ objOffset : objOffset + pdc.objFeatures.shape[0], pdc.objTreatmentFeatureId ] += treatmentOffset
            imgFeatures[ imgOffset : imgOffset + pdc.imgFeatures.shape[0], pdc.imgTreatmentFeatureId ] += treatmentOffset
            for replicate in pdc.replicates:
                replicate.index += replicateOffset
                if replicate.name == '':
                    replicate.name = '%d' % pdcIndex
                replicates.append( replicate )
                replicateIndexToPdcMapping.append( pdcIndex )
            objFeatures[ objOffset : objOffset + pdc.objFeatures.shape[0], pdc.objReplicateFeatureId ] += replicateOffset
            imgFeatures[ imgOffset : imgOffset + pdc.imgFeatures.shape[0], pdc.imgReplicateFeatureId ] += replicateOffset
            imgOffset += len( pdc.images )
            objOffset += len( pdc.objects )
            wellOffset += len( pdc.wells )
            treatmentOffset += len( pdc.treatments )
            replicateOffset += len( pdc.replicates )

        #for replicate in replicates:
        #    print 'replicate %d,%d: %s' % (replicate.index, replicateIndexToPdcMapping[replicate.index], replicate.name)

        """renameWellIndices = []
        for i in xrange( len( wells ) ):
            well1 = wells[ i ]
            rename = False
            for j in xrange( i+1, len( wells ) ):
                well2 = wells[ j ]
                if well2.name == well1.name:
                    well2.name = well2.name + '_%d' % wellIndexToPdcMapping[ j ]
                    rename = True
            if rename:
                well1.name += '_%d' % wellIndexToPdcMapping[ i ]"""
        #for i in xrange( len( wells ) ):
        #    wells[ i ].name += '_%d' % wellIndexToPdcMapping[ i ]

        """removeTreatmentIndices = []
        for i in xrange( len( treatments ) ):
            treatment1 = treatments[ i ]
            rename = False
            for j in xrange( i+1, len( treatments ) ):
                treatment2 = treatments[ j ]
                if treatment2.name == treatment1.name:
                    treatment2.name = treatment2.name + '_%d' % treatmentIndexToPdcMapping[ j ]
                    rename = True
            if rename:
                treatment1.name += '_%d' % treatmentIndexToPdcMapping[ i ]"""
        #for i in xrange( len( treatments ) ):
        #    treatments[ i ].name += '_%d' % treatmentIndexToPdcMapping[ i ]

        """removeReplicateIndices = []
        for i in xrange( len( replicates ) ):
            replicate1 = replicates[ i ]
            rename = False
            for j in xrange( i+1, len( replicates ) ):
                replicate2 = replicates[ j ]
                if replicate2.name == replicate1.name:
                    replicate2.name = replicate2.name + '_%d' % replicateIndexToPdcMapping[ j ]
                    rename = True
            if rename:
                replicate1.name += '_%d' % replicateIndexToPdcMapping[ i ]"""
        #for i in xrange( len( replicates ) ):
        #    replicates[ i ].name += '_%d' % replicateIndexToPdcMapping[ i ]

        pdc = pdcs[ 0 ]

        for obj in objects:
            objFeatures[ obj.index, pdc.objObjectFeatureId ] = obj.index
            objFeatures[ obj.index, pdc.objImageFeatureId ] = obj.image.index
            objFeatures[ obj.index, pdc.objWellFeatureId ] = obj.image.well.index
            objFeatures[ obj.index, pdc.objTreatmentFeatureId ] = obj.image.treatment.index
            objFeatures[ obj.index, pdc.objReplicateFeatureId ] = obj.image.replicate.index

        for img in images:
            imgFeatures[ img.index, pdc.imgImageFeatureId ] = img.index
            imgFeatures[ img.index, pdc.imgWellFeatureId ] = img.well.index
            imgFeatures[ img.index, pdc.imgTreatmentFeatureId ] = img.treatment.index
            imgFeatures[ img.index, pdc.imgReplicateFeatureId ] = img.replicate.index

        # remove duplicate wells
        for i in xrange( len(wells)-1, -1, -1 ):
            well1 = wells[i]
            for j in xrange( i-1, -1, -1 ):
                well2 = wells[j]
                if well1.name == well2.name:
                    # remove well1
                    tmp = imgFeatures[:,pdc.imgWellFeatureId]
                    img_mask = tmp == well1.index
                    tmp[img_mask] = well2.index
                    imgFeatures[:,pdc.imgWellFeatureId] = tmp
                    tmp = objFeatures[:,pdc.objWellFeatureId]
                    obj_mask = tmp == well1.index
                    tmp[obj_mask] = well2.index
                    objFeatures[:,pdc.objWellFeatureId] = tmp
                    del wells[i]
                    break
        # reassign well indexes
        for i,well in enumerate(wells):
            tmp = imgFeatures[:,pdc.imgWellFeatureId]
            img_mask = tmp == well.index
            tmp[img_mask] = i
            imgFeatures[:,pdc.imgWellFeatureId] = tmp
            tmp = objFeatures[:,pdc.objWellFeatureId]
            obj_mask = tmp == well.index
            tmp[obj_mask] = i
            objFeatures[:,pdc.objWellFeatureId] = tmp
            well.index = i
        # remove duplicate treatments
        for i in xrange( len(treatments)-1, -1, -1 ):
            treatment1 = treatments[i]
            for j in xrange( i-1, -1, -1 ):
                treatment2 = treatments[j]
                if treatment1.name == treatment2.name:
                    # remove treatment1
                    tmp = imgFeatures[:,pdc.imgTreatmentFeatureId]
                    img_mask = tmp == treatment1.index
                    tmp[img_mask] = treatment2.index
                    imgFeatures[:,pdc.imgTreatmentFeatureId] = tmp
                    tmp = objFeatures[:,pdc.objTreatmentFeatureId]
                    obj_mask = tmp == treatment1.index
                    tmp[obj_mask] = treatment2.index
                    objFeatures[:,pdc.objTreatmentFeatureId] = tmp
                    del treatments[i]
                    break
        # reassign treatment indexes
        for i,treatment in enumerate(treatments):
            tmp = imgFeatures[:,pdc.imgTreatmentFeatureId]
            img_mask = tmp == treatment.index
            tmp[img_mask] = i
            imgFeatures[:,pdc.imgTreatmentFeatureId] = tmp
            tmp = objFeatures[:,pdc.objTreatmentFeatureId]
            obj_mask = tmp == treatment.index
            tmp[obj_mask] = i
            objFeatures[:,pdc.objTreatmentFeatureId] = tmp
            treatment.index = i
        # remove duplicate replicates
        for i in xrange( len(replicates)-1, -1, -1 ):
            replicate1 = replicates[i]
            for j in xrange( i-1, -1, -1 ):
                replicate2 = replicates[j]
                if replicate1.name == replicate2.name:
                    # remove replicate1
                    tmp = imgFeatures[:,pdc.imgReplicateFeatureId]
                    img_mask = tmp == replicate1.index
                    tmp[img_mask] = replicate2.index
                    imgFeatures[:,pdc.imgReplicateFeatureId] = tmp
                    tmp = objFeatures[:,pdc.objReplicateFeatureId]
                    obj_mask = tmp == replicate1.index
                    tmp[obj_mask] = replicate2.index
                    objFeatures[:,pdc.objReplicateFeatureId] = tmp
                    del replicates[i]
                    break
        # reassign replicate indexes
        for i,replicate in enumerate(replicates):
            tmp = imgFeatures[:,pdc.imgReplicateFeatureId]
            img_mask = tmp == replicate.index
            tmp[img_mask] = i
            imgFeatures[:,pdc.imgReplicateFeatureId] = tmp
            tmp = objFeatures[:,pdc.objReplicateFeatureId]
            obj_mask = tmp == replicate.index
            tmp[obj_mask] = i
            objFeatures[:,pdc.objReplicateFeatureId] = tmp
            replicate.index = i

        pdc.imgFeatures = imgFeatures
        pdc.objFeatures = objFeatures
        pdc.images = images
        pdc.objects = objects
        pdc.wells = wells
        pdc.treatments = treatments
        pdc.replicates = replicates
        pdc.errors = errors

        pdc.wellByName = {}
        for well in pdc.wells:
            pdc.wellByName[ well.name ] = well.index

        pdc.treatmentByName = {}
        for treatment in pdc.treatments:
            pdc.treatmentByName[ treatment.name ] = treatment.index

        pdc.replicateByName = {}
        for replicate in pdc.replicates:
            pdc.replicateByName[ replicate.name ] = replicate.index

    else:
        pdc.treatmentByName = {}
        for treatment in pdc.treatments:
            if treatment.name.startswith("wt"):
                treatment.name = treatment.name.replace("wt", "mock")
            pdc.treatmentByName[treatment.name] = treatment.index

    return pdc



def export_hdf5_results(file, pdc):

    print 'exporting results to HDF5 file...'

    f = None
    try:
        print 'writing hdf5...'
        f = h5py.File( file, mode='w' )
        root = f.create_group( 'YACA_input' )

        root.attrs[ '__version__' ] = pdc.__version__

        print 'pickling object...'
        sio = cStringIO.StringIO()
        p = cPickle.Pickler( sio )

        #imgFeatures = pdc.imgFeatures
        #objFeatures = pdc.objFeatures
        #pdc.imgFeatures = None
        #pdc.objFeatures = None

        pdc_container = pdc.export_container()
        p.dump( pdc_container )

        #pdc.objFeatures = objFeatures
        #pdc.imgFeatures = imgFeatures

        pickleStr = sio.getvalue()
        sio.close()

        print 'writing pickle stream...'
        dt = numpy.uint8
        pickleData = struct.unpack( '%dB' % len( pickleStr ), pickleStr )
        pickleStream_dataset = root.create_dataset('pickleStream', dtype=numpy.uint8, data=pickleData)

        print 'writing image feature table...'
        imgFeature_dataset = root.create_dataset('imgFeatures', data=pdc.imgFeatures)

        print 'writing object feature table...'
        objFeature_dataset = root.create_dataset('objFeatures', data=pdc.objFeatures)

        """
        print 'writing mat...'
        d = {}
        d['imgFeatures'] = pdc.imgFeatures
        d['objFeatures'] = pdc.objFeatures
        scipy.io.savemat(file, d)
        """

    finally:
        if f:
            f.close()

    print 'exported data to HDF5 file'


"""
import numpy
import cPickle
import h5py
import StringIO
import sys
import os


from input_container import *


class pdc_hybrid_format(object):

    SHM_TEMP_DIRECTORY = '/dev/shm'
    if not os.path.isdir( SHM_TEMP_DIRECTORY ):
        SHM_TEMP_DIRECTORY = '/tmp/shm'
        if not os.path.isdir( SHM_TEMP_DIRECTORY ):
            os.mkdir( SHM_TEMP_DIRECTORY )

    SHM_TEMP_FILENAME = SHM_TEMP_DIRECTORY + '/yaca-hybrid-tmp-file-%d'

    def __init__(self, pdc):
        self.pdc = pdc
        self.hdf5 = None
        fileUsed = True
        while fileUsed:
            rndId = numpy.random.randint(sys.maxint)
            self.tempfile = self.SHM_TEMP_FILENAME % rndId
            fileUsed = os.path.isfile(self.tempfile)

    def make_hybrid(self, remove_tmp_file = True):

        OBJECT_PROPERTY_IDS = self.pdc.objects[0].properties.keys()

        dt = h5py.new_vlen( str )
        objectProperties_dataset = root.create_dataset(
            'objectProperties',
            ( len(pdc.objects), len(OBJECT_PROPERTY_IDS) ),
            dtype=dt
        )
        for i in xrange( len( pdc.objects ) ):
            obj = pdc.objects[i]
            objectProperties_dataset[i] = obj.properties.values()

        f = None
        try:

            f = h5py.File( self.tempfile, mode='w' )
    
            print 'exporting results to HDF5 file'
            root = f.create_group('YACA_input')
    
            print 'exporting image features...'
            #imgFeature_dataset = root.create_dataset('imgFeatures', data=self.pdc.imgFeatures)
            self.pdc.imgFeatures = None
    
            print 'exporting object features...'
            #objFeature_dataset = root.create_dataset('objFeatures', data=self.pdc.objFeatures)
            self.pdc.objFeatures = None

            f.close()

        except:
            if f:
                f.close()
            os.remove( self.tempfile )
            raise

        f = None
        try:

            f = open( self.tempfile, mode='rb' )
            self.hdf5 = f.read()

        finally:
            if f:
                f.close()
            if remove_tmp_file:
                os.remove( self.tempfile )

    def restore(self, create_tmp_file = True):

        if create_tmp_file:

            f = None
            try:

                f = open( self.tempfile, mode='wb' )
                f.write( self.hdf5 )
                f.close()

            except:
                if f:
                    f.close()
                os.remove( self.tempfile )
                raise

        f = None
        try:

            f = h5py.File( self.tempfile, mode='r' )

            print 'importing results from HDF5 file'
            root = f[ 'YACA_input' ]

            print 'importing image features...'
            #imgFeature_dataset = root['imgFeatures']
            #self.pdc.imgFeatures = numpy.empty( imgFeature_dataset.shape )
            #imgFeature_dataset.read_direct( self.pdc.imgFeatures )
    
            print 'importing object features...'
            #objFeature_dataset = root['objFeatures']
            #self.pdc.objFeatures = numpy.empty( objFeature_dataset.shape )
            #objFeature_dataset.read_direct( self.pdc.objFeatures )

        finally:
            if f:
                f.close()
            self.hdf5 = None
            os.remove( self.tempfile )




def import_hybrid_results(file):

    print 'importing results from HYBRID file...'
    f = None

    try:
        f = open(file, 'r')
        up = cPickle.Unpickler(f)
        print 'unpickling object...'
        hybrid = up.load()
        hybrid.restore()
    finally:
        if f:
            f.close()
    return hybrid.pdc




def export_hybrid_results(file, pdc):

    print 'exporting results to HYBRID file...'

    objFeatures = pdc.objFeatures
    imgFeatures = pdc.imgFeatures
    f = None

    try:
        hybrid = pdc_hybrid_format(pdc)
        hybrid.make_hybrid( False )
        f = open(file, 'w')
        p = cPickle.Pickler(f)
        print 'pickling object...'
        p.dump(hybrid)
        hybrid.restore( False )
    finally:
        if f:
            f.close()
        pdc.objFeatures = objFeatures
        pdc.imgFeatures = imgFeatures
"""
