import unittest

class ImportCheck(unittest.TestCase):
    def testImport(self):
        """importing ccluster c-extension"""
        import ccluster

class RunCheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global ccluster
        global np
        import ccluster
        import numpy as np

    def createEmptyParameterArrays(self, npoints, nclusters, nfeatures):
        points = np.empty((npoints, nfeatures), dtype=np.float64)
        clusters = np.empty((nclusters, nfeatures), dtype=np.float64)
        partition = np.empty((npoints,), dtype=np.int64)
        return points, clusters, partition
    def randomizeParameterArrays(self, points, clusters, features):
        points[:] = np.random.uniform(-1.0, 1.0, size=(points.shape))
        clusters[:] = np.random.uniform(-1.0, 1.0, size=(clusters.shape))
        partition[:] = np.random.randint(0, clusters.shape[0], size=(partition.shape))
    def normalRandomizeArray(self, arr, mean, sigma):
        arr[:] = np.random.normal(mean, sigma, size=(arr.shape))
    def uniformRandomizeArray(self, arr, lower=-1.0, upper=1.0):
        arr[:] = np.random.uniform(lower, upper, size=(arr.shape))
    def createRandomParameterArrays(self, npoints, nclusters, nfeatures):
        result = createEmptyParameterArrays(npoints, nclusters, nfeatures)
        randomizeParameterArrays(*result)
        return result

    def testSimpleRun(self):
        """testing a single run with 10 points, 2 clusters and 1 feature"""
        points, clusters, partition = self.createEmptyParameterArrays(10, 2, 1)
        points[:5] = np.random.normal(-1.0, 0.5, (5,1))
        points[5:] = np.random.normal( 1.0, 0.5, (5,1))
        clusters[0,0] = -10.0
        clusters[1,0] = -5.0
        partition[:] = 0

        minkowski_p = 2
        swap_threshold = 0
        ccluster.kmeans(points, clusters, partition, minkowski_p, swap_threshold)

    def testComplexRun(self):
        """testing a single run with 50000 points, 50 clusters and 50 features"""
        npoints = 50000
        nclusters = 50
        nfeatures = 50
        points, clusters, partition = self.createEmptyParameterArrays(npoints, nclusters, nfeatures)
        partition[:] = np.random.randint(0, nclusters, npoints)
        for i in xrange(nclusters):
            mask = partition == i
            mean = np.random.uniform(-1.0, 1.0, nfeatures)
            sigma = np.diag(np.random.uniform(0.1, 0.5, nfeatures))
            s = np.sum(mask)
            points[mask,:] = np.random.multivariate_normal(mean, sigma, int(np.sum(mask)))
        l = np.random.randint(0, npoints, nclusters)
        clusters = points[l].copy()
        partition[:] = np.random.randint(0, nclusters, npoints)

        minkowski_p = 2
        swap_threshold = 0
        ccluster.kmeans(points, clusters, partition, minkowski_p, swap_threshold)

    def testMultipleRuns(self):
        """testing 50 runs with 10000 points, 20 clusters and 20 features"""
        nruns = 20
        npoints = 10000
        nclusters = 20
        nfeatures = 20
        points, clusters, partition = self.createEmptyParameterArrays(npoints, nclusters, nfeatures)
        partition[:] = np.random.randint(0, nclusters, npoints)
        for i in xrange(nclusters):
            mask = partition == i
            mean = np.random.uniform(-1.0, 1.0, nfeatures)
            sigma = np.diag(np.random.uniform(0.1, 0.5, size=nfeatures))
            points[mask] = np.random.multivariate_normal(mean, sigma, int(np.sum(mask)))
        l = np.random.randint(0, npoints, nclusters)
        clusters = points[l].copy()
        partition[:] = np.random.randint(0, nclusters, npoints)
        backup_partition = partition.copy()
        backup_clusters = clusters.copy()
        first_partition = None
        first_clusters = None

        for i in xrange(nruns):
            minkowski_p = 2
            swap_threshold = 0
            ccluster.kmeans(points, clusters, partition, minkowski_p, swap_threshold)
            if first_partition is None:
                first_partition = partition.copy()
                first_clusters = clusters.copy()
            else:
                partition_equal_first_partition = (partition == first_partition).all()
                self.assertTrue(partition_equal_first_partition)
                clusters_equal_first_clusters = (clusters == first_clusters).all()
                self.assertTrue(clusters_equal_first_clusters)
            partition = backup_partition.copy()
            clusters = backup_clusters.copy()

if __name__ == '__main__':
    unittest.main()

