# -*- coding: utf-8 -*-

"""
headless_cluster_configuration.py -- Stub module for running in headless mode.
"""

# This software is distributed under the FreeBSD License.
# See the accompanying file LICENSE for details.
# 
# Copyright 2011 Benjamin Hepp

from ..core import parameter_utils as utils

class HeadlessClusterConfiguration:

    #__OBJECT_NAME = 'ClusterConfiguration'

    def __init__(self):

        self.clusterConfiguration = []

        utils.register_object('ClusterConfiguration', self)
        utils.register_attribute(self, 'clusterConfiguration', self.getClusterConfiguration, self.setClusterConfiguration)

    def getClusterConfiguration(self):
        return self.clusterConfiguration
    def setClusterConfiguration(self, configuration):
        self.clusterConfiguration = configuration
