import numpy



OBJECT_ID_FEATURE_NAME = 'OBJECT_ID'
TREATMENT_ID_FEATURE_NAME = 'TREATMENT_ID'
IMAGE_ID_FEATURE_NAME = 'IMAGE_ID'



# definition of the data structure used for the CellProfiler2 output
class apc_data_container(object):

    def __init__(self):

        self.images = []
        self.objects = []

        self.treatments = []
        self.treatmentByName = {}

        self.imgFeatureIds = {}
        self.imgFeatures = None #numpy.array(0.0)

        self.imgImageFeatureId = -1
        self.imgTreatmentFeatureId = -1

        self.objFeatureIds = {}
        self.objFeatures = None #numpy.array(0.0)

        self.objObjectFeatureId = -1
        self.objImageFeatureId = -1
        self.objTreatmentFeatureId = -1

        self.errors = []


class apc_data_image(object):

    def __init__(self):
        self.rowId = -1
        self.state = 'ok'
        self.treatment = None
        self.imageFiles = []
        self.properties = {}
#        self.objIds = None #numpy.array(0)


class apc_data_object(object):

    def __init__(self):
        self.rowId = -1
        self.state = 'ok'
        self.image = None
        self.position_x = -1.0
        self.position_y = -1.0
#        self.properties = {}


class apc_data_treatment(object):

    def __init__(self, name):
        self.rowId = -1
        self.name = name
#        self.imgIds = None #numpy.array(0)
#        self.objIds = None #numpy.array(0)


class apc_data_error(object):

    def __init__(self, exception, traceback, entity):
        self.exception = exception
        self.traceback = traceback
        self.entity = entity
