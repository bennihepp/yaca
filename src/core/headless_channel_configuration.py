# -*- coding: utf-8 -*-

"""
headless_channel_configuration.py -- Stub module for running in headless mode.
"""

# This software is distributed under the FreeBSD License.
# See the accompanying file LICENSE for details.
# 
# Copyright 2011 Benjamin Hepp

from ..core import parameter_utils as utils

class HeadlessChannelConfiguration:

    #__OBJECT_NAME = 'ChannelDescription'

    def __init__(self):

        self.channelMapping = {}
        self.channelDescription = {}

        utils.register_object('ChannelConfiguration', self)
        utils.register_attribute(
            self, 'channelMappingAndDescription',
            self.getChannelMappingAndDescription,
            self.setChannelMappingAndDescription)

    def getChannelMappingAndDescription(self):
        return self.channelDescription, self.channelMapping
    def setChannelMappingAndDescription(self, configuration):
        self.channelDescription, self.channelMapping = configuration
