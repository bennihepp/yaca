import numpy
import struct
import cPickle
import cStringIO
import h5py
import scipy.io


from data_container import *



def import_hdf5_results(file):

    print 'importing results from HDF5 file...'

    f = None
    try:
        f = h5py.File( file, mode='r' )
        root = f[ 'PhenoNice_input' ]

        print 'reading pickle stream ...'
        pickleStream_dataset = root[ 'pickleStream' ]
        dt = numpy.uint8
        stream_len = pickleStream_dataset.shape[0]
        pickleStr = ''.join(
            struct.unpack( '%dc' % stream_len, pickleStream_dataset[:] )
        )

        sio = cStringIO.StringIO( pickleStr )
        up = cPickle.Unpickler( sio )
        print 'unpickling object...'
        pdc = up.load()
        sio.close()

        print 'reading image features table...'
        imgFeature_dataset = root['imgFeatures']
        pdc.imgFeatures = numpy.empty( imgFeature_dataset.shape )
        imgFeature_dataset.read_direct( pdc.imgFeatures )

        print 'reading object features table...'
        objFeature_dataset = root['objFeatures']
        pdc.objFeatures = numpy.empty( objFeature_dataset.shape )
        objFeature_dataset.read_direct( pdc.objFeatures )

        """
        print 'reading mat...'
        d = scipy.io.loadmat(file + '.mat')
        pdc.objFeatures = d['objFeatures']
        pdc.imgFeatures = d['imgFeatures']
        """

    finally:
        if f:
            f.close()

    print 'imported data from HDF5 file'

    return pdc



def export_hdf5_results(file, pdc):

    print 'exporting results to HDF5 file...'

    f = None
    try:
        print 'writing hdf5...'
        f = h5py.File( file, mode='w' )
        root = f.create_group( 'PhenoNice_input' )

        print 'pickling object...'
        sio = cStringIO.StringIO()
        p = cPickle.Pickler( sio )

        imgFeatures = pdc.imgFeatures
        objFeatures = pdc.objFeatures
        pdc.imgFeatures = None
        pdc.objFeatures = None
        p.dump( pdc )
        pdc.objFeatures = objFeatures
        pdc.imgFeatures = imgFeatures

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

    SHM_TEMP_FILENAME = SHM_TEMP_DIRECTORY + '/phenonice-hybrid-tmp-file-%d'

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
            root = f.create_group('PhenoNice_input')
    
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
            root = f[ 'PhenoNice_input' ]

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
