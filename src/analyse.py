import sys
import os

if len(sys.argv) < 2:
	print \
"""Usage: ./%s <output_file> <coordinate_file>""" % (sys.argv[0])
	sys.exit(1)

import numpy
import scipy
import scipy.linalg
from ioutil.import_cp1_csv import import_cp1_csv_files
from ioutil.import_cp2_csv import import_cp2_csv_results
from ioutil.import_export_hybrid import *
import mahal_dist
import cluster
import mds
from gui import gui_window


class QualityControl:
	pass


#if __name__ == "__main__":

qc = QualityControl()

project_name = 'HCM_Course_GolgiMorpho' #'miriam'

# make some configuration depending on the project\
if project_name == 'HCM_Course_GolgiMorpho':

    objects = ['cells','nucleus'] # todo
    delimiter = ','
    path = '/home/benjamin/Desktop/images/HCM_Course_GolgiMorpho'
    #data = 'MergedOUT_101103.mat''
    file_images = os.path.join(path,'results-cp/DefaultOUT__11_Image.csv')
    files_objects = [ ( 'nucleus_', os.path.join(path,'results-cp/DefaultOUT__11_Nuclei.csv') ),
                      ( 'cell_', os.path.join(path,'results-cp/DefaultOUT__11_Cells.csv') ) ]
    hybrid_file = os.path.join( path,'results-cp/results.hybrid' )

    channelDescription = {}
    channelDescription['R'] = 'Nucleus staining (A568)'
    channelDescription['G'] = 'Protein staining (A488)'
    channelDescription['B'] = 'Cell staining (DAPI)'
    channelDescription['O1'] = 'Cell segmentation'
    channelDescription['O2'] = 'Nucleus segmentation'

    control_treatment_names = ['sc','wt']

elif project_name == 'fabian':
	objects = ['cells','nucleus']
	path = '\\\\almf\\almf\\group\\ALMFstuff\\Tischi\\Fabian_Heiko\\IF_NPC1_results\\'
	#data = 'MergedOUT_101103.mat''
	file_images = '../data/Merged_clng_new_Image.xls'
	file_cells = '../data/Merged_clng_new_cells.xls'
	file_nuclei = '../data/Merged_clng_new_nucleus.xls'
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
	treatmentControl = 'IF_NPC1_Lamp1_scram1'
	qc = QualityControl()
	qc.computeMahalanobis = True
	qc.makeMedianMontage = False
	qc.minAreaBG = 50000 # pixels
	qc.minNucSize = 5000
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



qc.minAreaBG = 50000 # pixels
qc.minNucSize = 1000 # pixels

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


if len( sys.argv ) > 1:
    output_file = sys.argv[ 1 ]
if len( sys.argv ) > 2:
    mahal_file = sys.argv[ 2 ]
if len( sys.argv ) > 3:
    sorted_output_file = sys.argv[ 3 ]
if len( sys.argv ) > 4:
    clustering_file = sys.argv[ 4 ]
if len( sys.argv ) > 5:
    mds_file = sys.argv[ 5 ]


# import data from files

# see if results are already available in native format and are up to date
max_results_file_mtime = 0
tmp = [ ( 'dummy', file_images ) ]
tmp.extend(files_objects)
for prefix,file in tmp:
    if os.path.getmtime(file) > max_results_file_mtime:
        max_results_file_mtime = os.path.getmtime(file)

if os.path.isfile(hybrid_file) and ( os.path.getmtime(hybrid_file) > max_results_file_mtime ):
    adc = import_hybrid_results(hybrid_file)
else:
    adc = import_cp2_csv_results(file_images, files_objects, delimiter)
    export_hybrid_results(hybrid_file, adc)



print 'number of images: %d' % len( adc.images )
print 'number of objects: %d' % len( adc.objects )


print 'doing quality control'

# do some validation of the data
validImageMask, validCellMask = quality_control( qc, adc )

imgFeatures = adc.imgFeatures[ validImageMask ]
objFeatures = adc.objFeatures[ validCellMask ]

print 'number of images: %d' % len( adc.images )
print 'number of cells: %d' % len( adc.objects )
print 'number of valid images: %d' % validImageMask.sum()
print 'number of valid cells: %d' % validCellMask.sum()


# select features

featureList = [ 'AreaShape', 'Granularity', 'Texture', 'Intensity']
prefix_list = [ 'cell', 'nucleus' ]

featureIds = []

for prefix in prefix_list:
    for feature in featureList:
        for n,i in adc.objFeatureIds.iteritems():
            if n.startswith( '%s_%s' % (prefix,feature) ):
                featureIds.append( int( i ) )

"""featureIds = []
for feature in ['AreaShape_Area','AreaShape_Eccentricity','AreaShape_Extent','AreaShape_FormFactor','AreaShape_MajorAxisLength','AreaShape_MinorAxisLength','AreaShape_Perimeter','AreaShape_Solidity']:
    featureIds.append( adc.objFeatureIds[ 'nucleus_%s' % feature ] )

#featureIds.append(adc.nucleusFeatureIds['AreaShape_Perimeter'])

for feature in ['AreaShape_Area','AreaShape_Eccentricity','AreaShape_Extent','AreaShape_FormFactor','AreaShape_MajorAxisLength','AreaShape_MinorAxisLength','AreaShape_Perimeter','AreaShape_Solidity']:
    featureIds.append( adc.objFeatureIds[ 'cell_%s' % feature ] )
"""

#featureIds.append(adc.cellFeatureIds['AreaShape_Perimeter'])
#featureIds.append(adc.objFeatureIds['cell_MQ_number_SegSmall_SegVesicle_Vesicle'])
#featureIds.append(adc.objFeatureIds['cell_MQ_area_SegSmall_mean_SegVesicle_Vesicle'])
#featureIds.append(adc.objFeatureIds['cell_MQ_intensMean_SegSmall_Foreground_mean_SegVesicle_Vesicle'])


qc.computeMahalanobis = True
if qc.computeMahalanobis:

    control_treatment_mask = numpy.empty( (adc.objFeatures.shape[0],) )
    control_treatment_mask[ : ] = False
    for name in control_treatment_names:
        control_treatment = adc.treatments[ adc.treatmentByName[ name ] ]
        control_treatment_mask = numpy.logical_or( control_treatment_mask,
                                                   adc.objFeatures[ : , adc.objTreatmentFeatureId ] == control_treatment.rowId )
    control_treatment_mask = numpy.logical_and( validCellMask, control_treatment_mask )

    refMask = control_treatment_mask
    ref = adc.objFeatures[ refMask ]

    mahalFeatureIds = mahal_dist.select_features( ref, featureIds )

    tmp_list = list( mahalFeatureIds )
    mahalFeatureNames = list( mahalFeatureIds )
    for k,v in adc.objFeatureIds.iteritems():
        if v in tmp_list:
            i = tmp_list.index( v )
            if i >= 0:
                mahalFeatureNames[ i ] = k

    #print 'Using the following features for control cutoff:'
    #print '\n'.join(mahalFeatureNames)
    print 'using %d features for control cutoff' % len(mahalFeatureIds)

    ref = adc.objFeatures[ refMask ][ : , mahalFeatureIds ]
    test = adc.objFeatures[ : , mahalFeatureIds ]

    fraction = 0.8
    #fraction = 1.0
    d_mahal,controlMean,controlIds = mahal_dist.mahalanobis_distance(ref, test, fraction)

    """print 'calculating dist_median_by_treatment'
    treatment_id_to_name = []
    dist_median_by_treatment = numpy.empty( len( adc.treatments ) )
    for tr in adc.treatments:
        mask = adc.objFeatures[ : , adc.objTreatmentFeatureId ] == tr.rowId
        dist_median_by_treatment[ tr.rowId ] = numpy.median( d_mahal[ mask ] )
        treatment_id_to_name.append( tr.name )"""

    """# calculate the transformation matrix for the mahalanobis space
    print 'calculating mahalanobis transformation matrix...'
    ref = adc.objFeatures[ refMask ][ controlIds ][ : , mahalFeatureIds ]
    cov = mahal_dist.covariance_matrix( ref )
    eigenvalues, eigenvectors = scipy.linalg.eigh( cov )
    diag_m = numpy.diag( 1.0 / scipy.sqrt( eigenvalues ) )
    trans_m = numpy.dot( numpy.dot( eigenvectors , diag_m ) , eigenvectors.transpose() )
    #trans_m = multiply_m(eigenvectors,diag_m)
    #trans_m = multiply_m(trans_m,eigenvectors.transpose())
    # transformate data points into mahalanobis space
    trans_features = adc.objFeatures[ : , mahalFeatureIds ] - controlMean
    trans_features = numpy.dot( trans_m , trans_features.transpose() ).transpose()"""

    # select 'non-normal' cells
    cutoff_threshold = 1.0
    max_mahal_dist = numpy.median( d_mahal[ refMask ] )
    cutoffCellMask = d_mahal[ : ] > max_mahal_dist

    cellMask = numpy.logical_and( validCellMask , cutoffCellMask )
    objFeatures = adc.objFeatures[ cellMask ]

    print 'valid cells: %d' % numpy.sum(validCellMask)
    print 'non-normal cells: %d' % numpy.sum(cellMask)

    mahalFeatureIds = mahal_dist.select_features( objFeatures, mahalFeatureIds )

    objFeatures = adc.objFeatures[ cellMask ]

    tmp_list = list( mahalFeatureIds )
    mahalFeatureNames = list( mahalFeatureIds )
    for k,v in adc.objFeatureIds.iteritems():
        if v in tmp_list:
            i = tmp_list.index( v )
            if i >= 0:
                mahalFeatureNames[ i ] = k

    #print 'Using the following features:'
    #print '\n'.join(mahalFeatureNames)
    print 'using %d features' % len(mahalFeatureIds)

    # calculate the transformation matrix for the mahalanobis space
    print 'calculating mahalanobis transformation matrix...'
    mahalFeatures = adc.objFeatures[ cellMask ][ : , mahalFeatureIds ]
    cov = mahal_dist.covariance_matrix( mahalFeatures )
    eigenvalues, eigenvectors = scipy.linalg.eigh( cov )
    diag_m = numpy.diag( 1.0 / scipy.sqrt( eigenvalues ) )
    trans_m = numpy.dot( numpy.dot( eigenvectors , diag_m ) , eigenvectors.transpose() )
    #trans_m = multiply_m(eigenvectors,diag_m)
    #trans_m = multiply_m(trans_m,eigenvectors.transpose())
    # transformate data points into mahalanobis space
    mahalPoints = adc.objFeatures[ cellMask ][ : , mahalFeatureIds ]
    mahalMean = numpy.mean( mahalFeatures, 0 )
    mahalPoints = mahalPoints - mahalMean
    mahalPoints = numpy.dot( trans_m , mahalPoints.transpose() ).transpose()


    print 'calculating mahalanobis score...'
    #for i in xrange(trans_features.shape[0]):
    #	dist = numpy.sum(trans_features[i,:]**2)
    #	d_mahal[i] = dist
    """tmp_array = d_mahal[ refMask ][ controlIds ]

    medianMahalCtrl = numpy.median( tmp_array )
    meanMahalCtrl = numpy.mean( tmp_array )
    madMahalCtrl = numpy.mean( numpy.abs( tmp_array - meanMahalCtrl ) )
    mahalScore1 = ( d_mahal - medianMahalCtrl ) / madMahalCtrl
    dist = numpy.sum( mahalPoints[ : , : ]**2 , 1 )

    tmp_array = dist[ refMask ][ controlIds ]

    medianMahalCtrl = numpy.median( tmp_array )
    meanMahalCtrl = numpy.mean( tmp_array )
    madMahalCtrl = numpy.mean( numpy.abs( tmp_array - meanMahalCtrl ) )
    mahalScore2 = ( dist - medianMahalCtrl ) / madMahalCtrl"""

    output_file = None
    if output_file:
        print 'writing output file...'
        #for i in xrange(d_mahal.shape[0]):
        #	print 'i=%d: %f (new_i=%d), cell_area=%f, cell_perimeter=%f, nucleus_area=%f, nucleus_perimeter=%f' \
        #		% (qc.new_rowId_to_old_rowId[i],d_mahal[i],i,
        #			adc.objFeatures[i,adc.cellFeatureIds['AreaShape_Area']] * 10**(-4),
        #			adc.objFeatures[i,adc.cellFeatureIds['AreaShape_Perimeter']] * 10**(-2),
        #			adc.objFeatures[i,adc.nucleusFeatureIds['AreaShape_Area']] * 10**(-3),
        #			adc.objFeatures[i,adc.nucleusFeatureIds['AreaShape_Perimeter']] * 10**(-2))
        fout = open(output_file,'wb')
        str = 'treatment\tmedian mahalanobis distance\n'
        fout.write(str)
        for i in xrange(dist_median_by_treatment.shape[0]):
            str = '%s\t%f\n' % (treatment_id_to_name[i], dist_median_by_treatment[i])
            fout.write(str)
        str = '\n'
        fout.write(str)
        fout.write(str)
        fout.write(str)
        str = 'valid\trowId\td_mahal\tmahal_score1\tdist\tmahal_score2'
        for featureId in mahalFeatureIds:
            str += '\t%s' % featureId
        str += '\n'
        fout.write(str)
        for i in xrange(d_mahal.shape[0]):
            str = '%d\t%d\t%f\t%f\t%f\t%f' % ( validCellMask[i], i, d_mahal[i], mahalScore1[i], dist[i], mahalScore2[i])
            for featureId in mahalFeatureIds:
                str += '\t%f' % adc.objFeatures[i,featureId]
            str += '\n'
            fout.write(str)
        fout.close()

    sorted_output_file = None
    if sorted_output_file:
        print 'writing sorted output file...'
        sortIds = d_mahal.argsort()
        fout = open(sorted_output_file,'wb')
        str = 'valid\trowId\td_mahal\tx\ty'
        for featureId in mahalFeatureIds:
            str += '\t%s' % featureId
        str += '\n'
        fout.write(str)
        for j in xrange(d_mahal.shape[0]):
            i = sortIds[j]
            fout.write('%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t\n' \
                % ( validCellMask[i], i, d_mahal[i],
                    adc.objFeatures[i,adc.objFeatureIds['cell_Location_Center_X']] * 10**(-2),
                    adc.objFeatures[i,adc.objFeatureIds['cell_Location_Center_Y']] * 10**(-2),
                    adc.objFeatures[i,adc.objFeatureIds['cell_AreaShape_Area']] * 10**(-4),
                    adc.objFeatures[i,adc.objFeatureIds['cell_AreaShape_Perimeter']] * 10**(-2),
                    adc.objFeatures[i,adc.objFeatureIds['nucleus_AreaShape_Area']] * 10**(-3),
                    adc.objFeatures[i,adc.objFeatureIds['nucleus_AreaShape_Perimeter']] * 10**(-2)
                )
            )
        fout.close()


    # output data in mahalanobis space + mahalanobis distance
    mahal_file = None
    if mahal_file:
        print 'writing mahalanobis output file...'
        #for i in xrange(trans_features.shape[0]):
       # 	dist = numpy.sum(trans_features[i,:]**2)
       # 	d_mahal[i] = dist
        tmp_array = d_mahal[ refMask ][ controlIds ]
        medianMahalCtrl = numpy.median(tmp_array)
        meanMahalCtrl = numpy.mean(tmp_array)
        madMahalCtrl = numpy.mean(numpy.abs(tmp_array - meanMahalCtrl))
        mahalScore = (d_mahal - medianMahalCtrl) / madMahalCtrl
        fout = open(mahal_file,'wb')
        str = ''
        for i in xrange(mahalPoints.shape[1]):
        	str += 'feature %d\t' % i
        str += 'distance 1\tdistance 2\tmahalanobis_score\n'
        fout.write(str)
        for i in xrange(mahalPoints.shape[0]):
        	str = ''
        	for j in xrange(mahalPoints.shape[1]):
        		str += '%f\t' % mahalPoints[i,j]
        	dist = numpy.sum(mahalPoints[i,:]**2)
        	str += '%f\t%f\t%f\n' % (dist,d_mahal[i],mahalScore[i])
        	fout.write(str)
        fout.close()

    print 'performing clustering...'

    # cluster data points in mahalanobis space
    partition,clusters = cluster.cluster_by_dist( mahalPoints, len( adc.treatments ), 0 )
    """partition = numpy.empty( (points.shape[0],) , dtype='int64' )
    partition[ : ] = features[ : , adc.objTreatmentFeatureId ]
    clusters = numpy.empty( ( len(adc.treatments) , points.shape[1] ) )
    for i in xrange( clusters.shape[0] ):
        mask = features[ adc.objTreatmentFeatureId ] == adc.treatments[ i ].rowId
        clusters[i] = numpy.mean( points[ mask ] , 0 )"""

    cluster_count = numpy.zeros( clusters.shape[ 0 ] )
    for i in xrange( partition.shape[ 0 ] ):
        cluster_count[ partition[ i ] ] += 1

    # output clustering
    clustering_file = None
    if clustering_file:
        print 'writing clustering file...'
        fout = open(clustering_file,'wb')
        fout.write('clusters\n')
        for i in xrange(clusters.shape[0]):
            str = ''
            for j in xrange(clusters.shape[1]):
                str += '%f\t' % clusters[i,j]
            str = str[:-1] + '\n'
            fout.write(str)
        fout.write('\n')
        fout.write('cluster_count\n')
        for i in xrange(cluster_count.shape[0]):
            str = '%d\n' % cluster_count[i]
            fout.write(str)
        fout.write('\n')
        fout.write('partitioning (rowId -> clusterId)\n')
        str = 'rowId\tclusterId\t'
        for i in xrange(mahalPoints.shape[1]):
            str += 'feature %d\t' % i
        str += 'distance\n'
        fout.write(str)
        for i in xrange(partition.shape[0]):
            str = '%d\t%d\t' % (i,partition[i])
            for j in xrange(mahalPoints.shape[1]):
                str += '%f\t' % mahalPoints[i,j]
            dist = 0.0
            for j in xrange(mahalPoints.shape[1]):
                dist += (mahalPoints[i,j] - clusters[partition[i],j])**2
            dist = numpy.sqrt(dist)
            str += '%f\n' % dist
            fout.write(str)
        fout.close()


    #print 'calculating distance matrix...'

    # calculate the mahalanobis-distance matrix for all samples
    #d_mahal_m = mahal_dist.euclidian_distance_between_all_points(points)

    """print 'performing mds...'

    X,fitness = mds.simulated_annealing_mds(
                            points, 2,
                            10,
                            1000,
                            1000.0,
                            100,
                            1.0,
                            50000
    )
    print 'fitness=%f' % fitness


    # output MDS
    if mds_file:
        print 'writing mds output file...'
        fout = open(mds_file,'wb')
        fout.write('MDS by simulated annealing (fitness=%f)\n' % fitness)
        fout.write('x\ty\n')
        for i in xrange(X.shape[0]):
            str = '%f\t%f\n' % (X[i,0], X[i,1])
            fout.write(str)
        fout.close()"""


#===============================================================================
#     def mds(dist_m, M):
#     
#         A_m = - 0.5 * dist_m
#         A_row = numpy.mean(A_m,1)
#         A_col = numpy.mean(A_m,0)
#         A_scalar = numpy.mean(A_row)
#         
#         B_m = A_m + A_scalar
#         B_m -= A_col
#         B_m = B_m.transpose()
#         B_m -= A_row
#         B_m = B_m.transpose()
#         
#         eigvals,eigvecs = numpy.linalg.eig(B_m)
#         eigvecs_len = numpy.sum(eigvecs**2,0)
#         
#         eigvecs = eigvecs * numpy.sqrt(eigvals / eigvecs_len)
#         
#         sorting = numpy.argsort(eigvals)
#         X = eigvecs[:,sorting[-1-M:-1]] * numpy.sqrt(eigvals[sorting[-1-M:-1]])
#         
#         return X
#         
#     X = mds(d_mahal_m, 2)
#===============================================================================



    sorting = numpy.argsort(partition)
    inverse_sorting = numpy.empty( sorting.shape, dtype='int64')
    for i in xrange(sorting.shape[0]):
        inverse_sorting[sorting[i]] = i
    sorting_by_cluster = []
    j = 0
    for i in xrange(clusters.shape[0]):
        tmp = sorting[ j:j+cluster_count[i] ]
        j += cluster_count[i]
        sorting_by_cluster.append(tmp)

    #points = trans_features
    #X_by_clusters = []
    #points_by_clusters= []
    #j = 0
    #for i in xrange(clusters.shape[0]):
    #    X_by_clusters.append( X[ sorting[ j:j+cluster_count[i] ],: ] )
    #    points_by_clusters.append( points[ sorting[ j:j+cluster_count[i] ],: ] )
    #    j += cluster_count[i]
#===============================================================================
# 
#     from matplotlib import pyplot
#     #symbols = ['bo','gp','rs','ch','mD','y1']
#     symbols = ['bs','gs','rs','cs','ms','ys','s']
#     for i in xrange(len(X_by_clusters)):
#         pyplot.plot(
#                 X_by_clusters[i][ :,0 ],
#                 X_by_clusters[i][ :,1 ],
#                 symbols[ numpy.min( i, len(symbols)-1 ) ]
#         )
#     #pyplot.plot(X[:,0],X[:,1],'o')
#     pyplot.show()
#===============================================================================


#X = numpy.array( points[ : , (0,1) ] )

g = gui_window.GUI(adc, objFeatures, mahalPoints, mahalFeatureNames, channelDescription, partition, sorting, inverse_sorting, clusters, cluster_count, sys.argv)
