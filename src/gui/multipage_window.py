# -*- coding: utf-8 -*-

"""
multipage_window.py -- Window for embedding multiple windows.
"""

# This software is distributed under the FreeBSD License.
# See the accompanying file LICENSE for details.
# 
# Copyright 2011 Benjamin Hepp

from PyQt4.QtCore import *
from PyQt4.QtGui import *

import functools

class MultiPageWindow(QWidget):

    def __init__(self, pageTitles=['Page 1'], title='MultiPage', parent=None):
        QWidget.__init__(self, parent)
        self.setWindowTitle(title)
        if type(pageTitles) == int:
            pageNum = pageTitles
            pageTitles = []
            for page in xrange(1, pageNum+1):
                pageTitles.append('Page %d' % page)
        self.__build_widget(pageTitles)
        self.page = 0

    @property
    def page_count(self):
        return self.__stackedWidget.count()

    @property
    def page(self):
        return self.__stackedWidget.currentIndex()
    @page.setter
    def page(self, page):
        self.__stackedWidget.setCurrentIndex(page)
        self.__pageComboBox.setCurrentIndex(page)

    def get_child_window(self, page=-1):
        if page > -1:
            return self.__stackedWidget.widget(page)
        else:
            return self.__stackedWidget.currentWidget()
    def set_child_window(self, page, widget):
        if page <= -1:
            page = self.__stackedWidget.currentIndex()
        old_widget = self.__stackedWidget.widget(page)
        self.__stackedWidget.insertWidget(page, widget)
        self.__stackedWidget.removeWidget(old_widget)
    def __set_current_child_window(self, widget):
        self.set_child_window(-1, widget)
    child_window = property(get_child_window, __set_current_child_window)

    def set_child_window_title(self, page, title):
        if page <= -1:
            page = self.__stackedWidget.currentIndex()
        self.__pageComboBox.setItemText(page, title)

    #def statusBar(self):
    #    return self.parent().statusBar()

    def __build_widget(self, pageTitles):
        self.__stackedWidget = QStackedWidget()
        for i, title in enumerate(pageTitles):
            widget = QWidget()
            self.__stackedWidget.addWidget(widget)
        self.__pageComboBox = QComboBox()
        for title in pageTitles:
            self.__pageComboBox.addItem(title)
        self.connect(self.__pageComboBox, SIGNAL('activated(int)'),
                     self.__stackedWidget, SLOT('setCurrentIndex(int)'))
        vbox = QVBoxLayout()
        vbox.addWidget(self.__pageComboBox, 0)
        vbox.addWidget(self.__stackedWidget, 1)
        self.setLayout(vbox)
