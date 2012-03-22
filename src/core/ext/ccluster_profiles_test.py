import unittest

import numpy as np

class ImportCheck(unittest.TestCase):
    def testImport(self):
        """importing ccluster c-extension"""
        import ext.ccluster_profiles_mp

class RunCheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global ccluster_profiles
        import ext.ccluster_profiles_mp as ccluster_profiles
        global cluster_profiles
        import cluster_profiles

    def createRandomArray(self, n1, n2):
        points = np.random.uniform(0.0, 1.0, size=(n1, n2))
        return points

    def testSimpleRun(self):
        """testing a single run with 100x50 matrices"""
        points = points = self.createRandomArray(100, 50)

        m1 = ccluster_profiles.compute_treatment_distance_map(points)
        m2 = cluster_profiles.compute_treatment_distance_map(
            points, 'quadratic_chi')
        deviation = np.sum(np.abs(m1 - m2))
        self.assertAlmostEqual(0.0, deviation)

    def testComplexRun(self):
        """testing a single run with 1000x500 matrices"""
        points = points = self.createRandomArray(1000, 50)

        m1 = ccluster_profiles.compute_treatment_distance_map(points)
        m2 = ccluster_profiles.compute_treatment_distance_map(points)
        deviation = np.sum(np.abs(m1 - m2))
        self.assertAlmostEqual(0.0, deviation)

    #def testSimpleRun(self):
        #"""testing a single run with 10 points, 2 clusters and 1 feature"""
        #masks, points, clusters = self.createEmptyParameterArrays(2, 10, 2, 1)
        #points[:5] = np.random.normal(-1.0, 0.5, (5,1))
        #points[5:] = np.random.normal(1.0, 0.5, (5,1))
        #clusters[0,0] = -10.0
        #clusters[1,0] = -5.0

        #exp_factor = 10.0
        #minkowski_p = 2
        #ccluster_profiles.compute_cluster_profiles(
            #masks, points, clusters, exp_factor, minkowski_p)

    #def testBigRun(self):
        #"""testing a single run with 50000 points, 50 clusters and 50 features"""
        #nmasks = 20
        #npoints = 50000
        #nclusters = 50
        #nfeatures = 50
        #masks, points, clusters = self.createEmptyParameterArrays(
            #nmasks, npoints, nclusters, nfeatures)
        #for mask in masks:
            #mean = np.random.uniform(-1.0, 1.0, nfeatures)
            #sigma = np.diag(np.random.uniform(0.1, 0.5, nfeatures))
            #s = np.sum(mask)
            #points[mask,:] = np.random.multivariate_normal(mean, sigma, int(np.sum(mask)))
        #l = np.random.randint(0, npoints, nclusters)
        #clusters = points[l].copy()

        #exp_factor = 10.0
        #minkowski_p = 2
        #ccluster_profiles.compute_cluster_profiles(
            #masks, points, clusters, exp_factor, minkowski_p)

    #def testComplexRun(self):
        #"""testing a single run with 1000 masks, 1000000 points, 50 clusters and 50 features"""
        #nmasks = 5000
        #npoints = 1000000
        #npoints = 50000
        #nclusters = 50
        #nfeatures = 50
        #masks, points, clusters = self.createEmptyParameterArrays(
            #nmasks, npoints, nclusters, nfeatures)
        #for mask in masks:
            #mean = np.random.uniform(-1.0, 1.0, nfeatures)
            #sigma = np.diag(np.random.uniform(0.1, 0.5, nfeatures))
            #s = np.sum(mask)
            #points[mask,:] = np.random.multivariate_normal(mean, sigma, int(np.sum(mask)))
        #l = np.random.randint(0, npoints, nclusters)
        #clusters = points[l].copy()

        #exp_factor = 10.0
        #minkowski_p = 2
        #ccluster_profiles.compute_cluster_profiles(
            #masks, points, clusters, exp_factor, minkowski_p)

    #def testMultipleRuns(self):
        #"""testing 50 runs with 10000 points, 20 clusters and 20 features"""
        #nruns = 20
        #nmasks = 10
        #npoints = 10000
        #nclusters = 20
        #nfeatures = 20
        #masks, points, clusters = self.createEmptyParameterArrays(
            #nmasks, npoints, nclusters, nfeatures)
        #for mask in masks:
            #mean = np.random.uniform(-1.0, 1.0, nfeatures)
            #sigma = np.diag(np.random.uniform(0.1, 0.5, size=nfeatures))
            #points[mask] = np.random.multivariate_normal(mean, sigma, int(np.sum(mask)))
        #l = np.random.randint(0, npoints, nclusters)
        #clusters = points[l].copy()
        #first_profiles = None

        #for i in xrange(nruns):
            #exp_factor = 10.0
            #minkowski_p = 2
            #profiles = ccluster_profiles.compute_cluster_profiles(
                #masks, points, clusters, exp_factor, minkowski_p)
            #if first_profiles is None:
                #first_profiles = profiles
            #else:
                ##profiles_equal_first_profiles = (profiles == first_profiles).all()
                ##self.assertTrue(profiles_equal_first_profiles)
                #deviation = np.sum(np.abs(profiles - first_profiles))
                #self.assertAlmostEqual(0.0, deviation)

    #def testCompareRuns(self):
        #"""testing a single run with 50000 points, 50 clusters and 50 features"""
        #import core.cluster_profiles as cp
        #nmasks = 20
        #npoints = 50000
        #nclusters = 50
        #nfeatures = 50

        ##nmasks = 2
        ##npoints = 5
        ##nclusters = 2
        ##nfeatures = 1
        #masks, points, clusters = self.createEmptyParameterArrays(
            #nmasks, npoints, nclusters, nfeatures)
        #for mask in masks:
            #mean = np.random.uniform(-1.0, 1.0, nfeatures)
            #sigma = np.diag(np.random.uniform(0.1, 0.5, nfeatures))
            #s = np.sum(mask)
            #points[mask,:] = np.random.multivariate_normal(mean, sigma, int(np.sum(mask)))
        #l = np.random.randint(0, npoints, nclusters)
        #clusters = points[l].copy()

        ##masks = (np.array([ True, False, False,  True,  True], dtype=bool), np.array([False,  True,  True, False, False], dtype=bool))
        ##points = np.array([[ 1.1403257 ],
       ##[ 0.62425984],
       ##[ 0.54179782],
       ##[ 1.86185732],
       ##[ 1.74856199]])
        ##clusters = np.array([[ 0.62425984],
       ##[ 0.62425984]])

        ##import wingdbstub

        #exp_factor = 10.0
        #minkowski_p = 2
        #profiles1 = ccluster_profiles.compute_cluster_profiles(
            #masks, points, clusters, exp_factor, minkowski_p)
        #profiles2 = cp.compute_cluster_profiles(
            #masks, points, clusters, exp_factor, minkowski_p)

        ##profiles_equal = (profiles1 == profiles2).all()
        ##self.assertTrue(profiles_equal)
        #deviation = np.sum(np.abs(profiles1 - profiles2))
        #self.assertAlmostEqual(0.0, deviation)

if __name__ == '__main__':
    unittest.main()
