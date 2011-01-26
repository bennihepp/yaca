import csv
import numpy
import scipy

import mahal_dist


class QualityControl:
	pass


project_name = 'miriam'


# make some configuration depending on the project
if project_name == 'fabian':
	objects = ['cells','nucleus']
	path = '\\\\almf\\almf\\group\\ALMFstuff\\Tischi\\Fabian_Heiko\\IF_NPC1_results\\'
	#data = 'MergedOUT_101103.mat''
	file_images = '../data/CP-output--Fabian_Image.xls'
	file_cells = '../data/CP-output--Fabian_cells.xls'
	file_nuclei = '../data/CP-output--Fabian_nucleus.xls'
	channels = []
	channels.append('Nucleus')
	channels.append('NPC1')
	channels.append('Vesicle')
	channelsOrig = []
	channelsOrig.append('Dapi')
	channelsOrig.append('GFP')
	channelsOrig.append('Cy3')
	channelsSeg = []
	channelsSeg.append('Segmentation')
	channelsSeg.append('VesicleSeg')
	color = '0.0.255-0.255.0-255.0.0-255.255.255-255.255.255-255.255.255'
	treatmentControl = 'A_Lamp1_scram'
	qc = QualityControl()
	qc.computeMahalanobis = True
	qc.makeMedianMontage = False
	qc.minAreaBG = 50000 # pixels
elif project_name == 'miriam':
	path = '\\\\almf\\almf\\group\\ALMFstuff\\Tischi\\Miriam\\VSVG_release_morphology_results\\'
	file_images = '../data/CP-output--Fabian_Image.xls'
	file_cells = '../data/CP-output--Fabian_cells.xls'
	file_nuclei = '../data/CP-output--Fabian_nucleus.xls'
	channels = []
	channels.append('Nucleus')
	channels.append('Protein')
	channels.append('Vesicle')
	channelsOrig = []
	channelsOrig.append('dapi')
	channelsOrig.append('yfp')
	channelsOrig.append('cy5')
	channelsSeg = []
	channelsSeg.append('Segmentation')
	channelsSeg.append('ProteinSeg')
	color = '0.0.255-0.255.0-255.0.0-255.255.255-255.255.255-255.255.255'
	treatmentControl = 'wt_00'
	evaluationChannel = 'Protein'
	qc = QualityControl()
	qc.computeMahalanobis = True
	qc.makeMedianMontage = False
	qc.minAreaBG = 50000 # pixels
	
qc.minDist_X_left = 100 #300
qc.minDist_X_right = 100
qc.minDist_Y_top = 100
qc.minDist_Y_bottom = 100 # todo: check where top and bottom is

qc.imageX = 1344
qc.imageY = 1024
qc.imageSize = qc.imageX * qc.imageY

qc.minCytoNucAreaFrac = 1.5
qc.maxCytoNucAreaFrac = 15

qc.minCells = 1
qc.maxCells = 300




# deletes listed cells and nuclei from the passed feature array
# returns the new feature array new_features
def delete_objects_from_feature_array(ids, features):
	new_features = numpy.array(0.0)
	new_features.resize(
				features.shape[0] - len(ids), features.shape[1]
	)
	i = 0
	for j in xrange(features.shape[0]):
		if not j in ids:
			new_features[i] = features[j]
			i += 1
	return new_features



# deletes listed cells and nuclei from a apc_data structure
def delete_objects(qc, ids, apc_data):

	qc.new_rowId_to_old_rowId = {}
	qc.old_rowId_to_new_rowId = {}
	rowIdCorrection = 0
	for i in xrange(len(data.cells)):
		cell = data.cells[i]
		nucleus = data.nuclei[i]
		old_rowId = cell.rowId
		cell.rowId += rowIdCorrection
		nucleus.rowId += rowIdCorrection
		if i in ids:
			cell.state = 'invalid'
			nucleus.state = 'invalid'
			rowIdCorrection -= 1
			qc.old_rowId_to_new_rowId[old_rowId] = -1
		else:
			qc.new_rowId_to_old_rowId[cell.rowId] = old_rowId
			qc.old_rowId_to_new_rowId[old_rowId] = cell.rowId
	ids.sort(reverse=True)
	for i in ids:
		del data.cells[i]
		del data.nuclei[i]

	new_objFeatures = delete_objects_from_feature_array(ids, data.objFeatures)
	del data.objFeatures
	data.objFeatures = new_objFeatures
	
	for im in data.images:
		ids2 = []
		for i in xrange(len(im.cells)):
			if im.cells[i].state == 'invalid':
				ids2.append(i)
		ids2.reverse()
		for j in ids2:
			del im.cells[j]
			del im.nuclei[j]
		new_objFeatures = delete_objects_from_feature_array(ids2, im.objFeatures)
		del im.objFeatures
		im.objFeatures = new_objFeatures
		
	for tr in data.treatments.values():
		ids2 = []
		for i in xrange(len(tr.cells)):
			if tr.cells[i].state == 'invalid':
				ids2.append(i)
		ids2.reverse()
		for j in ids2:
			del tr.cells[j]
			del tr.nuclei[j]
		new_objFeatures = delete_objects_from_feature_array(ids2, tr.objFeatures)
		del tr.objFeatures
		tr.objFeatures = new_objFeatures


# quality control of the data
def quality_control(qc, data):
	    delete_ids = []
	#for image in data.images:
	
		# maximum number of cells
		#if (len(image.cells) > qc.maxCells):
		#	print 'too_many_cells'
		#	image.state = 'too_many_cells'
		#	#sys.exit(-1)
		
		# minimal number of background pixels
		#featureId = data.cellFeatureIds['AreaShape_Area']
		#areaOccupiedByCells = sum(image.objFeatures[:,featureId])
		#if (qc.imageSize - areaOccupiedByCells < qc.minAreaBG):
		#	print 'not_enough_bg_Pixels for image(%d): %s' % (image.rowId,image.name)
		#	image.state = 'not_enough_bg_pixels'
		#	#sys.exit(-1)

		# cells are at image periphery
		featureId = columnIds['Location_Center_X']
		cellsOk = (features[:,featureId] > qc.minDist_X_left)
		cellsOK = numpy.logical_and(cellsOk,
					(features[:,featureId] < qc.imageX-qc.minDist_X_right)
		)
		featureId = columnIds['Location_Center_Y']
		cellsOk = numpy.logical_and(cellsOk,
					(image.objFeatures[:,featureId] > qc.minDist_Y_top)
		)
		cellsOK = numpy.logical_and(cellsOk,
					(features[:,featureId] < qc.imageY-qc.minDist_Y_bottom)
		)
		
		# minimum nucleus area
		#featureId = data.nucleusFeatureIds['AreaShape_Area']
		#thresholdNucleusArea = numpy.median(image.objFeatures[:,featureId] \
		#              - 2 * numpy.mean( numpy.abs(
		#                      image.objFeatures[:,featureId]
		#                      - numpy.mean(image.objFeatures[:,featureId])
		#              ) )
		#)
		#cellsOk = numpy.logical_and(cellsOk,
		#			(image.objFeatures[:,featureId] > thresholdNucleusArea)
		#)

		# minimum relative cytoplasm area
		#cellFeatureId = columnIds['AreaShape_Area']
		#cellsOk = numpy.logical_and(cellsOk,
		#			(features[:,cellFeatureId]
		#)
		#					  > qc.minCytoNucAreaFrac * features[:,featureId]
		#)

		# maximum relative cytoplasm area
		#cellsOk = numpy.logical_and(cellsOk,
		#			(image.objFeatures[:,cellFeatureId]
		#			  < qc.maxCytoNucAreaFrac * image.objFeatures[:,featureId])
		#)

		# check wether there are enough Ok cells left
		#numberOfOkCells = numpy.sum(cellsOk)
		#if (numberOfOkCells < qc.minCells):
		#	print 'not_enough_OKcells'
		#	image.state = 'not_enough_OK_cells'
			#sys.exit(-1)

		# add bad cells and nuclei to the cleanup list
		for i in xrange(len(cellsOk)):
			if not cellsOk[i]:
				delete_ids.append(i + image.cells[0].rowId)

	# finally cleanup all bad objects
	delete_objects(qc, delete_ids, data)


# do some validation of the data
quality_control(qc, data)



from parser.import_cp1_csv import read_cp1_csv_file
read_cp1_csv_file(file, close_file = False):

filename = 'Merged_cells.xls'
objects,features,columnIds = read_cp1_csv_file(filename)


