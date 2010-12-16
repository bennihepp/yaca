import csv
import numpy
import os

# definition of the data structure used for the CellProfiler output
class apc_data(object):
    def __init__(self):
        self.images = []
        self.imagesByName = {}
        self.cells = []
        self.nuclei = []
        self.treatments = {}
        self.imagePropertyIds = {}
        self.cellFeatureIds = {}
        self.nucleusFeatureIds = {}
        self.imageProperties = numpy.array(0.0)
        self.objFeatures = numpy.array(0.0)
        #self.nucleusFeatures = numpy.array(0.0)
class apc_data_image(object):
    def __init__(self, name):
        self.name = name
        self.state = 'OK'
        self.rowId = -1
        self.treatment = None
        self.properties = {}
        self.cells = []
        self.nuclei = []
        self.objFeatures = numpy.array(0.0)
        #self.nucleusFeatures = numpy.array(0.0)
class apc_data_cell(object):
    def __init__(self):
        self.state = 'OK'
        self.rowId = -1
        self.image = None
        self.nucleus = None
        self.features = {}
class apc_data_nucleus(object):
    def __init__(self):
        self.state = 'OK'
        self.rowId = -1
        self.image = None
        self.cell = None
        self.features = {}
class apc_data_treatment(object):
    def __init__(self, name):
        self.name = name
        self.images = []
        self.cells = []
        self.nuclei = []
        self.imageProperties = numpy.array(0.0)
        self.objFeatures = numpy.array(0.0)
        #self.nucleusFeatures = numpy.array(0.0)



# Import CSV-files as exported from CellProfiler.
# Returns a list of python dictionaries d:
#   d['image'] is the name of the image.
#   All the other fields are the columns of the CSV-file indexed
#   with the corresponding headers.
def import_cp1_csv_files(file_images, file_cells, file_nuclei,
                         close_file_images = False,
                         close_file_cells = False,
                         close_file_nuclei = False):

    # import files
    (imageObjs,imageFeatures,imageColIds) = \
            read_cp1_csv_file(file_images,close_file_images)
    (cellObjs,cellFeatures,cellColIds) = \
            read_cp1_csv_file(file_cells,close_file_cells)
    (nucleusObjs,nucleusFeatures,nucleusColIds) = \
            read_cp1_csv_file(file_nuclei,close_file_nuclei)

    # some consistency control
    if cellFeatures.shape[0] != nucleusFeatures.shape[0]:
        raise Exception('invalid input files: %s,%s,%s' % (file_images,file_cells,file_nuclei))

    # create data container
    data = apc_data()

    # copy numerical data-arrays into the apc_data structure
    data.imageProperties = imageFeatures
    # merge cells and nuclei into 'objects'
    data.objFeatures = numpy.array(0.0)
    data.objFeatures.resize(cellFeatures.shape[0],len(cellColIds)+len(nucleusColIds))
    data.objFeatures[:,:len(cellColIds)] = cellFeatures
    data.objFeatures[:,len(cellColIds):] = nucleusFeatures
    for k,v in nucleusColIds.iteritems():
        nucleusColIds[k] = v + len(cellColIds)
    # copy column descriptions into the apc_data structure
    data.imagePropertyIds = imageColIds
    data.cellFeatureIds = cellColIds
    data.nucleusFeatureIds = nucleusColIds
    
    # copy image data into the apc_data structure
    for obj in imageObjs:
        name = obj['image_name']
        image = apc_data_image(name)
        image.rowId = int(obj['row_id'])
        image.properties = obj
        data.images.append(image)
        data.imagesByName[name] = image
        
    # copy cell data into the apc_data structure
    for obj in cellObjs:
        cell = apc_data_cell()
        cell.rowId = int(obj['row_id'])
        cell.features = obj
        data.cells.append(cell)
        
    # copy nucleus data into the apc_data structure
    for obj in nucleusObjs:
        nucleus = apc_data_nucleus()
        nucleus.rowId = int(obj['row_id'])
        nucleus.features = obj
        data.nuclei.append(nucleus)

    # link cells with corresponding image and fill cells-list
    # for each apc_data_image
    image = data.images[0]
    for cell in data.cells:
        # what image does the cell belong to?
        image_name = cell.features['image_name']
        # do we already have the corresponding image?
        if image_name != image.name:
            # find corresponding image
            image = data.imagesByName[image_name]
        
        # link cell with image
        cell.image = image
        # fill cells-list of the apc_data_image
        image.cells.append(cell)

    # link nuclei with corresponding image and fill nuclei-list
    # for each apc_data_image
    image = data.images[0]
    for nucleus in data.nuclei:
        # what image does the nucleus belong to?
        image_name = nucleus.features['image_name']
        # do we already have the corresponding image?
        if image_name != image.name:
            # find corresponding image
            image = data.imagesByName[image_name]
        
        # link nucleus with image
        nucleus.image = image
        # fill nuclei-list of the apc_data_image
        image.nuclei.append(nucleus)

    # fill objFeatures of apc_data_image
    for im in data.images:
        im.objFeatures.resize(( len(im.cells), data.objFeatures.shape[1] ))
        for i in xrange(len(im.cells)):
            cell = im.cells[i]
            im.objFeatures[i] = data.objFeatures[cell.rowId]
        #im.nucleusFeatures.resize(( len(im.nuclei), len(data.nucleusFeatureIds) ))
        #for i in xrange(len(im.nuclei)):
        #    nucleus = im.nuclei[i]
        #    im.nucleusFeatures[i] = data.nucleusFeatures[nucleus.rowId]

    # link nuclei with corresponding cells and vice versa
    for image in data.images:
    
        # just make sure the data is consistent
        if len(image.cells) != len (image.nuclei):
            raise Exception(
                'apc_data is inconsistent: different number of cells (%d)' \
                'and nuclei (%d) for the same image (%s)' \
                % (len(image.cells),len(image.nuclei),image.name)
            )
        
        for i in xrange(len(image.cells)):
            cell = image.cells[i]
            # just in case CellProfiler didn't write out cells and nuclei
            # in the same order use the information from the CSV-files
            parent_nucleus_index = int(cell.features['Parent_nucleus'])
            assert parent_nucleus_index != i, 'inconsistent data files'
            parent_nucleus = image.nuclei[parent_nucleus_index-1]
            # make the actual linkage
            cell.nucleus = parent_nucleus
            parent_nucleus.cell = cell
    
    # buildup treatments-dictionary of apc_data
    treatment = None
    for image in data.images:
    
        treatment_name = retrieve_treatment_name(image)
        # do we already have the corresponding treatment?
        if treatment == None or treatment.name != treatment_name:
            # did we already create a corresponding apc_data_treatment?
            try:
                treatment = data.treatments[treatment_name]
            except:
                # we need to create a new treatment
                treatment = apc_data_treatment(treatment_name)
                data.treatments[treatment_name] = treatment
                
        # link image with treatment
        image.treatment = treatment
        # fill images-list of the apc_data_treatment
        treatment.images.append(image)
        # fill cells-list of the apc_data_treatment
        treatment.cells.extend(image.cells)
        # fill nuclei-list of the apc_data_treatment
        treatment.nuclei.extend(image.nuclei)

    # fill imageProperties and objFeatures of apc_data_treatment
    for (name,tr) in data.treatments.iteritems():
        tr.imageProperties.resize(( len(tr.images), len(data.imagePropertyIds) ))
        for i in xrange(len(tr.images)):
            tr.imageProperties[i] = data.imageProperties[tr.images[i].rowId]
        tr.objFeatures.resize(( len(tr.cells), data.objFeatures.shape[1] ))
        for i in xrange(len(tr.cells)):
            tr.objFeatures[i] = data.objFeatures[tr.cells[i].rowId]
        #tr.nucleusFeatures.resize(( len(tr.nuclei), len(data.nucleusFeatureIds) ))
        #for i in xrange(len(tr.nuclei)):
        #    tr.nucleusFeatures[i] = data.nucleusFeatures[tr.nuclei[i].rowId]
            

    return data



# retrieve the treatment used in the given image
TREATMENT_FEATURE_IDENTIFIER = 'PathName_'
def retrieve_treatment_name(image):
    treatment_name = ''
    for (k,v) in image.properties.iteritems():
        if k.startswith(TREATMENT_FEATURE_IDENTIFIER):
            # make sure we are windows-compatible
            v = v.replace('\\','/')
            treatment_name = os.path.split(os.path.split(v)[0])[-1]
            break
    return treatment_name



# Reads a CSV-file as exported from CellProfiler.
# Returns (columnIds,data,numericalColumnIds,numericalData):
#   data is a numpy-array with all data values as strings
#   numericalData is a numpy-array with numerical values
#   columnIds and numericalColumnIds are dictionaries describing the columns
#   of data and numericalData respectivly
#   columnIds[columnName] = columnId
#   The columns correspond to the columns of the CSV-file indexed
#   with the corresponding headers.
def read_cp1_csv_file(file, close_file = False):

    # open file if a filename was passed
    if type(file) == str:
        close_file = True
        filename = file
        try:
            file = open(file,'rb')
        except IOError, e:
            print 'ERROR: unable to open file %s: %s' % (filename, e)
            raise

    # use the CSV module to read the input file
    reader = csv.reader(file, delimiter='\t')

    try:

        # first we read the data from the file
        
        rows = []
        # we will use this dict to link column names to the indices
        # in data and numericalData
        columnIds = {'image_name':0}
        columnNames = ['image_name']
        found_headers = False
        # we will use this to identify the type of each column
        columnTypes = [float]

        # this is used to keep the image name for several rows (in case it's not repeated)
        image = ''
        
        for row in reader:

            if not found_headers:
                # we haven't found the row with the column-descriptions yet
                if len(row) >= 2 and len(row[1].strip()) > 0:
                    # found it, write column-descriptions into columnIds
                    found_headers = True
                    for i in xrange(1,len(row)):
                        columnIds[row[i]] = i
                        columnNames.append(row[i])
                        columnTypes.append(float)
            
            else:
                # read dataset
                # if the row has an image name, we use it,
                # otherwise we take the image name of the previous row(s)
                if len(row[0]) > 0:
                    image = row[0]
                else:
                    row[0] = image
                # add dataset to the data-container
                rows.append(numpy.array(row))
                # check types of columns
                for i in xrange(len(columnIds)):
                    if columnTypes[i] == float:
                        try:
                            float(row[i])
                        except:
                            columnTypes[i] = str

        # this will keep all the data as dictionaries
        objects = []
        # and all numerical data is also written into this numpy-array
        features = numpy.array(0.0)
        # determine columns with numerical data
        floatIndices = []
        for i in xrange(len(columnTypes)):
            if columnTypes[i] == float:
                floatIndices.append(i)
        # adjust the size of features to the number of numerical columns
        features.resize(( len(rows), len(floatIndices) ))
        
        for i in xrange(len(rows)):
            row = rows[i]
            obj = {}
            obj['row_id'] = i
            for (name,id) in columnIds.iteritems():
                obj[name] = row[id]
            objects.append(obj)
            features[i] = row[floatIndices]
        
        columnIds = {}
        for i in xrange(len(floatIndices)):
            columnIds[columnNames[floatIndices[i]]] = i
                
        return (objects,features,columnIds)
        
        
    except csv.Error, e:
        print 'ERROR: file %s, line %d: %s' % (file.name, reader.line_num, e)
        raise

    # some cleanup
    finally:
        if close_file:
            file.close()


"""

# Import CSV-files as exported from CellProfiler.
# Returns a list of python dictionaries d:
#   d['image'] is the name of the image.
#   All the other fields are the columns of the CSV-file indexed
#   with the corresponding headers.
def import_mat_results_file(file, close_file = False):

    # import files
    (imageObjs,imageFeatures,imageColIds) = \
            read_cp1_csv_file(file_images,close_file_images)
    (cellObjs,cellFeatures,cellColIds) = \
            read_cp1_csv_file(file_cells,close_file_cells)
    (nucleusObjs,nucleusFeatures,nucleusColIds) = \
            read_cp1_csv_file(file_nuclei,close_file_nuclei)

    # some consistency control
    if cellFeatures.shape[0] != nucleusFeatures.shape[0]:
        raise Exception('invalid input files: %s,%s,%s' % (file_images,file_cells,file_nuclei))

    # create data container
    data = apc_data()

    # copy numerical data-arrays into the apc_data structure
    data.imageProperties = imageFeatures
    # merge cells and nuclei into 'objects'
    data.objFeatures = numpy.array(0.0)
    data.objFeatures.resize(cellFeatures.shape[0],len(cellColIds)+len(nucleusColIds))
    data.objFeatures[:,:len(cellColIds)] = cellFeatures
    data.objFeatures[:,len(cellColIds):] = nucleusFeatures
    for k,v in nucleusColIds.iteritems():
        nucleusColIds[k] = v + len(cellColIds)
    # copy column descriptions into the apc_data structure
    data.imagePropertyIds = imageColIds
    data.cellFeatureIds = cellColIds
    data.nucleusFeatureIds = nucleusColIds
    
    # copy image data into the apc_data structure
    for obj in imageObjs:
        name = obj['image_name']
        image = apc_data_image(name)
        image.rowId = int(obj['row_id'])
        image.properties = obj
        data.images.append(image)
        data.imagesByName[name] = image
        
    # copy cell data into the apc_data structure
    for obj in cellObjs:
        cell = apc_data_cell()
        cell.rowId = int(obj['row_id'])
        cell.features = obj
        data.cells.append(cell)
        
    # copy nucleus data into the apc_data structure
    for obj in nucleusObjs:
        nucleus = apc_data_nucleus()
        nucleus.rowId = int(obj['row_id'])
        nucleus.features = obj
        data.nuclei.append(nucleus)

    # link cells with corresponding image and fill cells-list
    # for each apc_data_image
    image = data.images[0]
    for cell in data.cells:
        # what image does the cell belong to?
        image_name = cell.features['image_name']
        # do we already have the corresponding image?
        if image_name != image.name:
            # find corresponding image
            image = data.imagesByName[image_name]
        
        # link cell with image
        cell.image = image
        # fill cells-list of the apc_data_image
        image.cells.append(cell)

    # link nuclei with corresponding image and fill nuclei-list
    # for each apc_data_image
    image = data.images[0]
    for nucleus in data.nuclei:
        # what image does the nucleus belong to?
        image_name = nucleus.features['image_name']
        # do we already have the corresponding image?
        if image_name != image.name:
            # find corresponding image
            image = data.imagesByName[image_name]
        
        # link nucleus with image
        nucleus.image = image
        # fill nuclei-list of the apc_data_image
        image.nuclei.append(nucleus)

    # fill objFeatures of apc_data_image
    for im in data.images:
        im.objFeatures.resize(( len(im.cells), data.objFeatures.shape[1] ))
        for i in xrange(len(im.cells)):
            cell = im.cells[i]
            im.objFeatures[i] = data.objFeatures[cell.rowId]
        #im.nucleusFeatures.resize(( len(im.nuclei), len(data.nucleusFeatureIds) ))
        #for i in xrange(len(im.nuclei)):
        #    nucleus = im.nuclei[i]
        #    im.nucleusFeatures[i] = data.nucleusFeatures[nucleus.rowId]

    # link nuclei with corresponding cells and vice versa
    for image in data.images:
    
        # just make sure the data is consistent
        if len(image.cells) != len (image.nuclei):
            raise Exception(
                'apc_data is inconsistent: different number of cells (%d)' \
                'and nuclei (%d) for the same image (%s)' \
                % (len(image.cells),len(image.nuclei),image.name)
            )
        
        for i in xrange(len(image.cells)):
            cell = image.cells[i]
            # just in case CellProfiler didn't write out cells and nuclei
            # in the same order use the information from the CSV-files
            parent_nucleus_index = int(cell.features['Parent_nucleus'])
            assert parent_nucleus_index != i, 'inconsistent data files'
            parent_nucleus = image.nuclei[parent_nucleus_index-1]
            # make the actual linkage
            cell.nucleus = parent_nucleus
            parent_nucleus.cell = cell
    
    # buildup treatments-dictionary of apc_data
    treatment = None
    for image in data.images:
    
        treatment_name = retrieve_treatment_name(image)
        # do we already have the corresponding treatment?
        if treatment == None or treatment.name != treatment_name:
            # did we already create a corresponding apc_data_treatment?
            try:
                treatment = data.treatments[treatment_name]
            except:
                # we need to create a new treatment
                treatment = apc_data_treatment(treatment_name)
                data.treatments[treatment_name] = treatment
                
        # link image with treatment
        image.treatment = treatment
        # fill images-list of the apc_data_treatment
        treatment.images.append(image)
        # fill cells-list of the apc_data_treatment
        treatment.cells.extend(image.cells)
        # fill nuclei-list of the apc_data_treatment
        treatment.nuclei.extend(image.nuclei)

    # fill imageProperties and objFeatures of apc_data_treatment
    for (name,tr) in data.treatments.iteritems():
        tr.imageProperties.resize(( len(tr.images), len(data.imagePropertyIds) ))
        for i in xrange(len(tr.images)):
            tr.imageProperties[i] = data.imageProperties[tr.images[i].rowId]
        tr.objFeatures.resize(( len(tr.cells), data.objFeatures.shape[1] ))
        for i in xrange(len(tr.cells)):
            tr.objFeatures[i] = data.objFeatures[tr.cells[i].rowId]
        #tr.nucleusFeatures.resize(( len(tr.nuclei), len(data.nucleusFeatureIds) ))
        #for i in xrange(len(tr.nuclei)):
        #    tr.nucleusFeatures[i] = data.nucleusFeatures[tr.nuclei[i].rowId]
            

    return data



# retrieve the treatment used in the given image
TREATMENT_FEATURE_IDENTIFIER = 'PathName_'
def retrieve_treatment_name2(image):
    treatment_name = ''
    for (k,v) in image.properties.iteritems():
        if k.startswith(TREATMENT_FEATURE_IDENTIFIER):
            # make sure we are windows-compatible
            v = v.replace('\\','/')
            treatment_name = os.path.split(os.path.split(v)[0])[-1]
            break
    return treatment_name



# Reads a CSV-file as exported from CellProfiler.
# Returns (columnIds,data,numericalColumnIds,numericalData):
#   data is a numpy-array with all data values as strings
#   numericalData is a numpy-array with numerical values
#   columnIds and numericalColumnIds are dictionaries describing the columns
#   of data and numericalData respectivly
#   columnIds[columnName] = columnId
#   The columns correspond to the columns of the CSV-file indexed
#   with the corresponding headers.
def read_mat_results_file(file, close_file = False):

    # open file if a filename was passed
    if type(file) == str:
        close_file = True
        filename = file
        try:
            file = open(file,'rb')
        except IOError, e:
            print 'ERROR: unable to open file %s: %s' % (filename, e)
            raise

    # use the CSV module to read the input file
    reader = csv.reader(file, delimiter='\t')

    try:

        # first we read the data from the file
        
        rows = []
        # we will use this dict to link column names to the indices
        # in data and numericalData
        imageColumnIds = {'image_name':0}
        columnNames = ['image_name']
        found_headers = False
        # we will use this to identify the type of each column
        columnTypes = [float]

        for row in reader:

            if not found_headers:
                # we haven't found the row with the column-descriptions yet
                if len(row) >= 2 and len(row[1].strip()) > 0:
                    # found it, write column-descriptions into columnIds
                    found_headers = True
                    for i in xrange(1,len(row)):
                        imageColumnIds[row[i]] = i
                        columnNames.append(row[i])
                        columnTypes.append(float)
            
            else:
                # read dataset
                if len(row[0]) == 0:
                	break
                # add dataset to the data-container
                rows.append(numpy.array(row))
                # check types of columns
                for i in xrange(len(imageColumnIds)):
                    if columnTypes[i] == float:
                        try:
                            float(row[i])
                        except:
                            columnTypes[i] = str

        # this will keep all the image-data as dictionaries
        imageObjects = []
        # and all numerical data is also written into this numpy-array
        imageFeatures = numpy.array(0.0)
        # determine columns with numerical data
        floatIndices = []
        for i in xrange(len(columnTypes)):
            if columnTypes[i] == float:
                floatIndices.append(i)
        # adjust the size of features to the number of numerical columns
        imageFeatures.resize(( len(rows), len(floatIndices) ))
        
        for i in xrange(len(rows)):
            row = rows[i]
            obj = {}
            obj['row_id'] = i
            for (name,id) in imageColumnIds.iteritems():
                obj[name] = row[id]
            imageObjects.append(obj)
            imageFeatures[i] = row[floatIndices]
        
        imageColumnIds = {}
        for i in xrange(len(floatIndices)):
            imageColumnIds[columnNames[floatIndices[i]]] = i
                
        
        found_cell_based_values = False
        rows = []
        # we will use this dict to link column names to the indices
        # in data and numericalData
        cellColumnIds = {'image_name':0}
        columnNames = ['image_name']
        found_cell_number = False
        # we will use this to identify the type of each column
        columnTypes = [float]
        state = 'invalid'
        treatment = ''
        feature_name = ''
        cellFeatures = {}

        for row in reader:

			if not found_cell_based_values:
				if row[0].startswith('Cell based values'):
					found_cell_based_values = True
					state = 'looking_for_treatment'
				continue
			if len(row[0]) == 0:
				continue
			
			if state == 'looking_for_treatment':
				treatment = row[0]
				state = 'looking_for_feature_name'
			elif state == 'looking_for_feature_name':
				feature_name = row[0]
				state == 'looking_for_feature_values'
			elif state == 'looking_for_feature_values':
				if not found_cell_number:
					found_cell_number = True
					if not cellFeatures.has_key(feature_name):
						cellFeatures[feature_name] = numpy.array(0.0)
						cellFeatures[feature_name].resize(len(row))
						i = 0
					else:
						i = cellFeatures[feature_name].shape[0]
						cellFeatures[feature_name].resize(len(row)+i)
					for j in xrange(len(row)):
						cellFeatures[feature_name][i+j] = float(row[j])
												
				for col in row:
            if not found_cell_number:
            	cell_numer = len(row)
                # we haven't found the row with the column-descriptions yet
                if len(row) >= 2 and len(row[1].strip()) > 0:
                    # found it, write column-descriptions into columnIds
                    found_headers = True
                    for i in xrange(1,len(row)):
                        imageColumnIds[row[i]] = i
                        columnNames.append(row[i])
                        columnTypes.append(float)
            
            else:
                # read dataset
                if len(row[0]) == 0:
                	break
                # add dataset to the data-container
                rows.append(numpy.array(row))
                # check types of columns
                for i in xrange(len(imageColumnIds)):
                    if columnTypes[i] == float:
                        try:
                            float(row[i])
                        except:
                            columnTypes[i] = str

        # this will keep all the image-data as dictionaries
        cellObjects = []
        # and all numerical data is also written into this numpy-array
        cellFeatures = numpy.array(0.0)
        # determine columns with numerical data
        floatIndices = []
        for i in xrange(len(columnTypes)):
            if columnTypes[i] == float:
                floatIndices.append(i)
        # adjust the size of features to the number of numerical columns
        cellFeatures.resize(( len(rows), len(floatIndices) ))
        
        for i in xrange(len(rows)):
            row = rows[i]
            obj = {}
            obj['row_id'] = i
            for (name,id) in cellColumnIds.iteritems():
                obj[name] = row[id]
            cellObjects.append(obj)
            cellFeatures[i] = row[floatIndices]
        
        cellColumnIds = {}
        for i in xrange(len(floatIndices)):
            cellColumnIds[columnNames[floatIndices[i]]] = i
                
                
        return (objects,features,columnIds)
        
        
    except csv.Error, e:
        print 'ERROR: file %s, line %d: %s' % (file.name, reader.line_num, e)
        raise

    # some cleanup
    finally:
        if close_file:
            file.close()
"""
