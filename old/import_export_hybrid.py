import numpy
import cPickle
import h5py
import scipy.io


from input_container import *



def import_hybrid_results(file):

    print 'importing results from HYBRID file...'

    f = None
    try:
        f = open(file, 'r')
        up = cPickle.Unpickler(f)
        print 'unpickling object...'
        adc = up.load()

    finally:
        if f:
            f.close()

    f = None
    try:
        f = h5py.File( file + '.hdf5', mode='r' )
        root = f[ 'APC_input' ]

        print 'reading imgFeatures...'
        imgFeature_dataset = root['imgFeatures']
        adc.imgFeatures = numpy.empty( imgFeature_dataset.shape )
        imgFeature_dataset.read_direct( adc.imgFeatures )

        print 'reading objFeatures...'
        objFeature_dataset = root['objFeatures']
        adc.objFeatures = numpy.empty( objFeature_dataset.shape )
        objFeature_dataset.read_direct( adc.objFeatures )

        """
        print 'reading mat...'
        d = scipy.io.loadmat(file + '.mat')
        adc.objFeatures = d['objFeatures']
        adc.imgFeatures = d['imgFeatures']
        """

    finally:
        if f:
            f.close()

    return adc



def export_hybrid_results(file, adc):

    print 'exporting results to HYBRID file...'

    imgFeatures = adc.imgFeatures
    objFeatures = adc.objFeatures
    adc.imgFeatures = None
    adc.objFeatures = None

    f = None
    try:
        f = open(file, 'w')
        p = cPickle.Pickler(f)

        print 'pickling object...'
        p.dump(adc)

    finally:
        if f:
            f.close()

        adc.objFeatures = objFeatures
        adc.imgFeatures = imgFeatures

    f = None
    try:
        print 'writing hdf5...'
        f = h5py.File( file + '.hdf5', mode='w' )
        root = f.create_group( 'APC_input' )
        imgFeature_dataset = root.create_dataset('imgFeatures', data=adc.imgFeatures)
        objFeature_dataset = root.create_dataset('objFeatures', data=adc.objFeatures)

        """
        print 'writing mat...'
        d = {}
        d['imgFeatures'] = adc.imgFeatures
        d['objFeatures'] = adc.objFeatures
        scipy.io.savemat(file, d)
        """

    finally:
        if f:
            f.close()


"""
import numpy
import cPickle
import h5py
import StringIO
import sys
import os


from input_container import *


class adc_hybrid_format(object):

    SHM_TEMP_FILENAME = '/dev/shm/apc-hybrid-tmp-file-%d'

    def __init__(self, adc):
        self.adc = adc
        self.hdf5 = None
        fileUsed = True
        while fileUsed:
            rndId = numpy.random.randint(sys.maxint)
            self.tempfile = self.SHM_TEMP_FILENAME % rndId
            fileUsed = os.path.isfile(self.tempfile)

    def make_hybrid(self, remove_tmp_file = True):

        OBJECT_PROPERTY_IDS = self.adc.objects[0].properties.keys()

        dt = h5py.new_vlen( str )
        objectProperties_dataset = root.create_dataset(
            'objectProperties',
            ( len(adc.objects), len(OBJECT_PROPERTY_IDS) ),
            dtype=dt
        )
        for i in xrange( len( adc.objects ) ):
            obj = adc.objects[i]
            objectProperties_dataset[i] = obj.properties.values()

        f = None
        try:

            f = h5py.File( self.tempfile, mode='w' )
    
            print 'exporting results to HDF5 file'
            root = f.create_group('APC_input')
    
            print 'exporting image features...'
            #imgFeature_dataset = root.create_dataset('imgFeatures', data=self.adc.imgFeatures)
            self.adc.imgFeatures = None
    
            print 'exporting object features...'
            #objFeature_dataset = root.create_dataset('objFeatures', data=self.adc.objFeatures)
            self.adc.objFeatures = None

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
            root = f[ 'APC_input' ]

            print 'importing image features...'
            #imgFeature_dataset = root['imgFeatures']
            #self.adc.imgFeatures = numpy.empty( imgFeature_dataset.shape )
            #imgFeature_dataset.read_direct( self.adc.imgFeatures )
    
            print 'importing object features...'
            #objFeature_dataset = root['objFeatures']
            #self.adc.objFeatures = numpy.empty( objFeature_dataset.shape )
            #objFeature_dataset.read_direct( self.adc.objFeatures )

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
    return hybrid.adc




def export_hybrid_results(file, adc):

    print 'exporting results to HYBRID file...'

    objFeatures = adc.objFeatures
    imgFeatures = adc.imgFeatures
    f = None

    try:
        hybrid = adc_hybrid_format(adc)
        hybrid.make_hybrid( False )
        f = open(file, 'w')
        p = cPickle.Pickler(f)
        print 'pickling object...'
        p.dump(hybrid)
        hybrid.restore( False )
    finally:
        if f:
            f.close()
        adc.objFeatures = objFeatures
        adc.imgFeatures = imgFeatures
"""
