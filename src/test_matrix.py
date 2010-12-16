import numpy
import scipy
import scipy.spatial.distance as scidist

a = numpy.array( [
(5.1, 1.0),
(2.0, 5.0),
(3.0, 3.3)
] )
print "a:"
print a

u = numpy.mean(a,0)

ac = a - u
print "ac:"
print ac

cov = numpy.cov(a, rowvar=0, bias=1)
cov2 = numpy.dot(ac.transpose(), ac) / len(ac)
print "cov:"
print cov
print "cov2:"
print cov2
print "cov1 == cov2"
print (cov == cov2).all()

inv_cov = numpy.linalg.inv(cov)
print "inv_cov:"
print inv_cov

L = numpy.linalg.cholesky(cov)
print "L:"
print L

inv_L = numpy.linalg.inv(L)
print "inv_L:"
print inv_L

inv_cov2 = numpy.dot(inv_L.transpose(), inv_L)
print "inv_cov2:"
print inv_cov2

print "inv_cov1 == inv_cov2"
print (inv_cov - inv_cov2)
print (inv_cov == inv_cov2).all()

d_mahal1 = numpy.dot(numpy.dot(a[0],inv_cov),a[0].transpose())
print "d_mahal1:"
print d_mahal1
d_mahal2 = numpy.dot(numpy.dot(a[1],inv_cov),a[1].transpose())
print "d_mahal2:"
print d_mahal2
d_mahal3 = numpy.dot(numpy.dot(a[2],inv_cov),a[2].transpose())
print "d_mahal3:"
print d_mahal3

d_mahal_m = numpy.dot(numpy.dot(a, inv_cov), a.transpose())
print "d_mahal_m:"
print d_mahal_m

d_mahal = numpy.diag(d_mahal_m)
print "d_mahal:"
print d_mahal

d_mahal_m2 = numpy.dot(a, inv_cov) * a
print "d_mahal_m2:"
print d_mahal_m2

d_mahal2 = numpy.sum(d_mahal_m2,1)
print "d_mahal2:"
print d_mahal2


eigvals,eigvecs = numpy.linalg.eigh(cov)
inv_eigvals,inv_eigvecs = numpy.linalg.eigh(inv_cov)
print "eigvals:"
print eigvals
print "eigvecs:"
print eigvecs

diag = numpy.diag(1.0/numpy.sqrt(eigvals))
inv_diag = numpy.diag(numpy.sqrt(inv_eigvals))
print "diag:"
print diag
print "inv_diag:"
print inv_diag

trans_m = numpy.dot(numpy.dot(eigvecs,diag),eigvecs.transpose())
print "trans_m:"
print trans_m

trans_a = numpy.dot(trans_m, ac.transpose()).transpose()
print "trans_a:"
print trans_a


