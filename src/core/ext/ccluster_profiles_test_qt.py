import unittest

import numpy as np

from PyQt4.QtCore import *
from PyQt4.QtGui import *

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
        points = points = self.createRandomArray(10, 50)

        m1 = ccluster_profiles.compute_treatment_distance_map(points)
        m2 = cluster_profiles.compute_treatment_distance_map(
            points, 'quadratic_chi')
        deviation = np.sum(np.abs(m1 - m2))
        self.assertAlmostEqual(0.0, deviation)

    def testComplexRun(self):
        """testing a single run with 1000x500 matrices"""
        points = points = self.createRandomArray(100, 50)

        m1 = ccluster_profiles.compute_treatment_distance_map(points)
        m2 = ccluster_profiles.compute_treatment_distance_map(points)
        deviation = np.sum(np.abs(m1 - m2))
        self.assertAlmostEqual(0.0, deviation)

if __name__ == '__main__':
    app = QApplication([])

    unittest.main()

    app.exec_()
