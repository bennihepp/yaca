# -*- coding: utf-8 -*-

"""
cell_mask_filter_widget.py -- Allows to define filters for the Cell Gallery.
"""

# This software is distributed under the FreeBSD License.
# See the accompanying file LICENSE for details.
# 
# Copyright 2011 Benjamin Hepp

from PyQt4.QtCore import *
from PyQt4.QtGui import *

import numpy

from ..core import quality_control


class CellMaskFilterWidget(QWidget):

    __pyqtSignals__ = ('adjust',)

    def __init__(self, pipeline, galleryWindow, parent=None):
        self.__pipeline = pipeline
        self.__pdc = self.__pipeline.pdc
        self.__galleryWindow = galleryWindow
        QWidget.__init__(self, parent)
        self.__build_widget()

    def __on_mask_text_changed(self):
        self.__updateButton.setEnabled(True)

    def __on_sort_text_changed(self):
        self.__updateButton.setEnabled(True)

    def __on_update_filter(self):
        self.__updateButton.setEnabled(False)

        partition = -numpy.ones((self.__pdc.objFeatures.shape[0],))
        partition[self.__pipeline.get_non_control_cell_mask()] = self.__pipeline.nonControlPartition

        exec_dict = globals().copy()
        exec_dict.update(locals())

        d1 = {
            'qc' : quality_control,
            'qualityControl' : self.__pdc.objFeatures[: , self.__pdc.objQualityControlFeatureId],
            'treatmentId' : self.__pdc.objFeatures[: , self.__pdc.objTreatmentFeatureId],
            'wellId' : self.__pdc.objFeatures[: , self.__pdc.objWellFeatureId],
            'replicateId' : self.__pdc.objFeatures[: , self.__pdc.objReplicateFeatureId],
            'objId' : self.__pdc.objFeatures[: , self.__pdc.objObjectFeatureId],
            'imgId' : self.__pdc.objFeatures[: , self.__pdc.objImageFeatureId],
            'features' : (lambda feature_name: self.__pdc.objFeatures[: , self.__pdc.objFeatureIds[feature_name]]),
            'imgFeatures' : (lambda feature_name: self.__pdc.imgFeatures[: , self.__pdc.imgFeatureIds[feature_name]]),
            'mahalDist' : self.__pdc.objFeatures[: , self.__pdc.objFeatureIds['Mahalanobis Distance']],
            'well' : (lambda well_name: self.__pdc.wellByName[well_name].index),
            'treatment' : (lambda treatment_name: self.__pdc.treatmentByName[treatment_name]),
            'replicate' : (lambda replicate_name: self.__pdc.replicateByName[replicate_name]),
            'pdc' : self.__pdc,
            'pl' : self.__pipeline,
            'np' : numpy,
            'mask_not' : self.__pipeline.mask_not,
            'mask_and' : self.__pipeline.mask_and,
            'mask_or' : self.__pipeline.mask_or,
            'partition' : partition,
            'clusterDist' : (lambda i: self.__pipeline.clusterDist[:,i])
        }
        keys = d1.keys()
        exec_dict.update(d1)
        d2 = {
             'VALID' : quality_control.QUALITY_CONTROL_VALID,
             'NOT_ENOUGH_VALID_CELLS' : quality_control.QUALITY_CONTROL_NOT_ENOUGH_VALID_CELLS,
             'TOO_MANY_CELLS' : quality_control.QUALITY_CONTROL_TOO_MANY_CELLS,
             'NOT_ENOUGH_BG_PIXELS' : quality_control.QUALITY_CONTROL_NOT_ENOUGH_VALID_CELLS,
             'OUT_OF_PERIPHERY' : quality_control.QUALITY_CONTROL_OUT_OF_PERIPHERY,
             'TOO_SMALL_NUCLEUS' : quality_control.QUALITY_CONTROL_TOO_SMALL_NUCLEUS,
             'TOO_BIG_NUCLEUS' : quality_control.QUALITY_CONTROL_TOO_BIG_NUCLEUS,
             'TOO_SMALL_CYTOPLASM' : quality_control.QUALITY_CONTROL_TOO_SMALL_CYTOPLASM,
             'TOO_BIG_CYTOPLASM' : quality_control.QUALITY_CONTROL_TOO_BIG_CYTOPLASM,
             'TOO_SMALL_NUCLEUS_SOLIDITY' : quality_control.QUALITY_CONTROL_NOT_ENOUGH_VALID_CELLS,
        }
        keys.extend(d2.keys())
        exec_dict.update(d2)

        maskText = self.__maskTextEdit.toPlainText()
        maskText.replace('\n', ' ')

        maskText = 'cell_mask = ' + str(maskText)
        print 'executing:', maskText
        #print 'context:'
        #keys.sort()
        #for key in keys:
        #    print '  ', key
        exec maskText in exec_dict
        cell_mask = exec_dict['cell_mask']

        selectionIds = self.__pdc.objFeatures[: , self.__pdc.objObjectFeatureId][cell_mask]


        exec_dict = globals().copy()
        exec_dict.update(locals())
        exec_dict.update(d1)
        exec_dict.update(d2)


        sortText = self.__sortTextEdit.toPlainText()
        sortText.replace('\n', ' ')

        sortText = 'sort_array = ' + str(sortText)
        print 'executing:', sortText
        exec sortText in exec_dict
        sort_array = exec_dict['sort_array']
        sort_array = sort_array[cell_mask]

        sort = numpy.argsort(sort_array)

        selectionIds = selectionIds[sort]


        self.__galleryWindow.on_selection_changed(
            self.__galleryWindow.focusId,
            selectionIds,
            self.__galleryWindow.pixmapFactory,
            self.__galleryWindow.featureFactory
       )


    def __build_widget(self):

        """
        quality_control[]
        treatmentId[]
        wellId[]
        replicateId[]
        objId[]
        imgId[]
        features(feature_name)
        imgFeatures(feature_name)
        """

        #default_mask_text = 'mask_and(pl.get_non_control_cell_mask(), pl.get_control_treatment_cell_mask())'
        default_mask_text = 'mask_and(pl.get_valid_cell_mask(), pl.get_control_treatment_cell_mask())'
        self.__maskTextEdit = QTextEdit(default_mask_text)

        default_sort_text = 'objId'
        self.__sortTextEdit = QTextEdit(default_sort_text)

        self.__updateButton = QPushButton('Update filter')

        self.connect(self.__maskTextEdit, SIGNAL('textChanged()'), self.__on_mask_text_changed)
        self.connect(self.__sortTextEdit, SIGNAL('textChanged()'), self.__on_sort_text_changed)
        self.connect(self.__updateButton, SIGNAL('clicked()'), self.__on_update_filter)

        vbox = QVBoxLayout()
        vbox.addWidget(QLabel('Mask by:'), 0)
        vbox.addWidget(self.__maskTextEdit, 1)
        vbox.addWidget(QLabel('Sort by:'), 0)
        vbox.addWidget(self.__sortTextEdit, 1)
        vbox.addWidget(self.__updateButton, 0)

        self.setLayout(vbox)
