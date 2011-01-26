import numpy
import numpy.linalg
import scipy.spatial
import scipy.linalg



def minkowski_dist(a, b, minkowski_p=2):

	if len( a.shape ) == 1:
		a = numpy.array( [ a ] )
	if len( b.shape ) == 1:
		b = numpy.array( [ b ] )

	if minkowski_p == 2:
	    return scipy.spatial.distance.cdist( a, b, 'euclidean' )
	elif minkowski_p == 1:
	    return scipy.spatial.distance.cdist( a, b, 'cityblock' )
	else:
	    return scipy.spatial.distance.cdist( a, b, 'minkowski', minkowski_p )

def weighted_minkowski_dist(a, b, weights, minkowski_p=2):
	wa = weights * a
	wb = weights * b

	return minkowski_dist( wa, wb, minkowski_p )

def minkowski_cdist(A, B, minkowski_p=2):
	if minkowski_p == 2:
	    return scipy.spatial.distance.cdist( A, B, 'euclidean' )
	elif minkowski_p == 1:
	    return scipy.spatial.distance.cdist( A, B, 'cityblock' )
	else:
	    return scipy.spatial.distance.cdist( A, B, 'minkowski', minkowski_p )

def weighted_minkowski_cdist(A, B, weights, minkowski_p=2):
	wA = weights * A
	wB = weights * B
	return minkowski_cdist( wA, wB, minkowski_p )


# Returns the covariance matrix of the passed observations.
# observations is an MxN array/matrix whereas M is the number of
# observations and N is the number of dimensions/features
def covariance_matrix(observations):
	# by hand
	#return numpy.dot(observations.transpose(), observations) / len(observations)
	# or with numpy (bias=1 means normalization by N instead of N-1)
	return numpy.cov(observations, rowvar=0, bias=1)



# Returns the inverse covariance matrix of the passed observations.
# observations is an MxN array/matrix whereas M is the number of
# observations and N is the number of dimensions/features
def inverse_covariance_matrix(observations):

    cov_m = covariance_matrix(observations)

    # first try Cholesky Decomposition
    try:
        L = numpy.linalg.cholesky(cov_m)
        inv_L = numpy.linalg.inv(L)
        return numpy.dot(inv_L.transpose(), inv_L)
    except:
        print 'Cholesky decomposition failed. Falling back to LU decomposition...'

    # then try it with LU Decomposition
    try:
        L,U = scipy.linalg.lu( cov_m, True )
        inv_L = numpy.linalg.inv( L )
        inv_U = numpy.linalg.inv( U )
        return numpy.dot( inv_U, inv_L )
    except:
        print 'LU decomposition failed. Falling back to standard matrix inversion...'

    # the easy way
    return numpy.linalg.inv(cov_m)



# returns the transformation matrix for the mahalanobis space
# in respect of the mean-value of the reference set
# reference is an MxL array/matrix
# whereas M is the number of observations
# and L is the number of dimensions/features
#def mahalanobis_transformation(reference_m):

	# compute the covariance matrix of the reference set
	#cov = covariance_matrix(reference_m)

	# calculate eigenvector
	#eigenvalues,eigenvectors = numpy.linalg.eigh(cov)
	# orthogonalize them
	#trans_m,R = numpy.linalg.qr(eigenvectors)
	# and denormalize them
	#trans_m = trans_m / eigenvalues

	#return trans_m



# returns the mahalanobis_distance of both sets in respect of the mean-value of
# the reference set
# reference is an MxL and test is an NxL array/matrix
# whereas M or N are the numbers of observations respectivly
# and L is the number of dimensions/features
def mahalanobis_distance(reference_m, test_m, fraction = 0.8):

    ref_ids = numpy.array(0)
    ref_ids.resize(reference_m.shape[0])
    for i in xrange(ref_ids.shape[0]):
        ref_ids[i] = i

    if fraction < 1.0:
    	dist = mahalanobis_distance(reference_m, reference_m, 1.0)
    	ref_ids = numpy.argsort(dist)
    	old_number = reference_m.shape[0]
    	number = numpy.round( reference_m.shape[0] * fraction) + 1
    	ref_ids = ref_ids[:number]
    	reference_m = reference_m[ref_ids]
    	print 'fraction=%f' % fraction
    	print 'using %d out of %d objects for mahalanobis distance reference' \
    			% (number, old_number)

    # compute the mean-value of the reference set
    ref_mean = numpy.mean(reference_m,0)

    # center reference and test set
    reference_centered_m = reference_m - ref_mean
    test_centered_m = test_m - ref_mean

    # compute the inverse covariance matrix of the reference set
    inv_cov = inverse_covariance_matrix(reference_m)

    # compute the actual distances
    #dist_m = numpy.dot(numpy.dot(test_centered_m, inv_cov), test_m.transpose())
    #dist = numpy.real(numpy.diag(dist_m))
    # a bit more efficient
    dist_m = numpy.dot(test_centered_m, inv_cov) * test_centered_m

    dist = numpy.sum(dist_m, 1)

    #return dist,ref_mean,ref_ids

    return dist



# returns the euclidian distance between all points.
# Points is an MxL array/matrix,
# whereas M are the numbers of observations respectivly
# and L is the number of dimensions/features.
# Returns a MxM matrix.
def euclidian_distance_between_all_points(points):

    return scipy.spatial.distance.cdist( points, points )



# returns the mahalanobis distance between all points in the test set using the
# coveriance matrix of the reference set
# reference is an MxL and test is an NxL array/matrix
# whereas M or N are the numbers of observations respectivly
# and L is the number of dimensions/features
def mahalanobis_distance_between_all_points(reference_m, test_m):

	# compute the mean-value of the reference set
	mean = numpy.mean(reference_m,0)

	# compute the inverse covariance matrix of the reference set
	inv_cov = inverse_covariance_matrix(reference_m)

	# this matrix will keep all the distances
	# all_dist_m[i,j] = distance between point i and point j
	all_dist_m = numpy.empty( test_m.shape[0] , test_m.shape[0] )

	# compute the actual distances
	for i in xrange(test_m.shape[0]):
		p = test_m[i]
		diff_m = test_m - p
		dist_m = numpy.dot(diff_m, inv_cov) * diff_m
		dist = numpy.transpose(numpy.sum(dist_m, 1))
		all_dist_m[i] = dist
		
	return all_dist_m



# Returns the mahalanobis distances of a set of observations to another set
# of observations.
# The inverse covariance matrix has to be provided.
# A_m is an MxL and B_m is an NxL array/matrix
# whereas M or N are the numbers of observations respectivly
# and L is the number of dimensions/features.
# inv_cov is the inverse covariance matrix.
# dist_m can be a MxN matrix. In this case, the results are stored
# in dist_m.
def mahalanobis_distance_between_sets(A_m, B_m, inv_cov, dist_m=None):

	if dist_m == None:
		dist_m = numpy.array(0.0)
		dist_m.resize(A_m.shape[0],B_m.shape[0])

	# compute the actual distances
	for i in xrange(A_m.shape[0]):
		p = A_m[i]
		diff_m = B_m - p
		dist_m = numpy.dot(diff_m, inv_cov) * diff_m
		dist = numpy.transpose(numpy.sum(dist_m, 1))
		dist_m[i] = dist

	return dist_m



def mahalanobis_transformation( features ):

    # calculate the transformation matrix for the mahalanobis space
    print 'calculating mahalanobis transformation matrix...'

    cov = covariance_matrix( features )
    eigenvalues, eigenvectors = scipy.linalg.eigh( cov )
    diag_m = numpy.diag( 1.0 / scipy.sqrt( eigenvalues ) )
    trans_m = numpy.dot( numpy.dot( eigenvectors , diag_m ) , eigenvectors.transpose() )

    return trans_m



def transform_features( features, transformation, center=True ):

    if center:
        mean = numpy.mean( features, 0 )
        transformed_features = features - mean

    # transform features according to transformation
    transformed_features = numpy.dot( transformation , transformed_features.transpose() ).transpose()

    return transformed_features
