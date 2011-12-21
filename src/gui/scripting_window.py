# -*- coding: utf-8 -*-

"""
scripting_window.py -- Script editor and interactive shell.
"""

# This software is distributed under the FreeBSD License.
# See the accompanying file LICENSE for details.
# 
# Copyright 2011 Benjamin Hepp

import sys, os, random
import cStringIO

from PyQt4.QtCore import *
from PyQt4.QtGui import *

from scripting.ipython_view_qt import IPythonView
from scripting.script_file_widget import ScriptFileWidget

class ScriptingWindow(QMainWindow):
    def __init__(self, user_ns, global_ns, parent=None):
        super(ScriptingWindow, self).__init__(parent)
        self.setWindowTitle('Scripting')
        self.__user_ns = user_ns.copy()
        self.__global_ns = global_ns.copy()
        self.__interactive_session_counter = 1
        self.__build_widget()

    def __build_widget(self):
        self.__tabwidget = QTabWidget()
        self.__tabwidget.setTabsClosable(True)
        self.__tabwidget.setMovable(True)
        self.connect(self.__tabwidget, SIGNAL('tabCloseRequested(int)'), self.on_tab_close_requested)
        #self.__mdiarea.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        #self.__mdiarea.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        #interactive_python_widget = InteractivePythonWidget(self.__user_ns, self.__global_ns,
        #                                                    title='Interactive session #1')
        interactive_python_widget = IPythonView(self.__user_ns, self.__global_ns,
                                                title='Interactive session')
        #self.__interactive_session_counter += 1
        self.__tabwidget.addTab(interactive_python_widget, interactive_python_widget.windowTitle())
        #self.__mdiarea.addSubWindow(interactive_python_widget)
        interactive_python_widget.show()
        self.setCentralWidget(self.__tabwidget)
        self.__build_menu()
        interactive_python_widget.setFocus()

    def minimumSizeHint(self):
        return QSize(600, 400)

    def sizeHint(self):
        return QSize(800, 600)

    #def on_new_interactive_session(self):
    #    interactive_python_widget = IPythonView(self.__user_ns, self.__global_ns,
    #                                            title='Interactive session #%d' % self.__interactive_session_counter)
    #    self.__interactive_session_counter += 1
    #    interactive_python_widget.show()
    #    self.__mdiarea.addSubWindow(interactive_python_widget)
    #    interactive_python_widget.show()
    def on_new_script_file(self):
        script_file_widget = ScriptFileWidget()
        self.__tabwidget.addTab(script_file_widget, script_file_widget.windowTitle())
        self.__tabwidget.setCurrentWidget(script_file_widget)
        #script_file_widget.show()
    def on_load_script_file(self):
        script_file_widget = ScriptFileWidget()
        if script_file_widget.on_open_file():
            self.__tabwidget.addTab(script_file_widget, script_file_widget.windowTitle())
            self.__tabwidget.setCurrentWidget(script_file_widget)
            #script_file_widget.show()
        else:
            script_file_widget.close()
    def on_save_script_file(self):
        if isinstance(self.__tabwidget.currentWidget(), ScriptFileWidget):
            self.__tabwidget.currentWidget().on_save_file()
    def on_save_as_script_file(self):
        if isinstance(self.__tabwidget.currentWidget(), ScriptFileWidget):
            self.__tabwidget.currentWidget().on_save_as_file()
    def on_close_script_file(self):
        if isinstance(self.__tabwidget.currentWidget(), ScriptFileWidget):
            if self.__tabwidget.currentWidget().on_close_file():
                self.__tabwidget.removeTab(self.__tabwidget.currentIndex())

    def on_run_script_file(self):
        if isinstance(self.__tabwidget.currentWidget(), ScriptFileWidget):
            content = unicode(self.__tabwidget.currentWidget().text())
            codeobj = compile(content, self.__tabwidget.currentWidget().filename(), 'exec')
            exec codeobj in self.__global_ns, self.__user_ns

    def closeEvent(self, event):
        for index in xrange(self.__tabwidget.count() - 1, 0, -1):
            widget = self.__tabwidget.widget(index)
            if isinstance(widget, ScriptFileWidget):
                if widget.on_close_file():
                    self.__tabwidget.removeTab(index)
                else:
                    event.ignore()
                    return
        event.accept()

    def on_tab_close_requested(self, index):
        widget = self.__tabwidget.widget(index)
        if isinstance(widget, ScriptFileWidget):
            if widget.on_close_file():
                self.__tabwidget.removeTab(index)

    def __build_menu(self):

        self.file_menu = self.menuBar().addMenu("&File")
        self.script_menu = self.menuBar().addMenu("Script")

        #new_interactive_action = self.__make_action("&New interactive session",
        #    shortcut="Ctrl+I", slot=self.on_new_interactive_session, 
        #    tip="Create a new interactive session")
        new_action = self.__make_action("&New script file",
            shortcut="Ctrl+N", slot=self.on_new_script_file, 
            tip="Create a new script file")
        open_action = self.__make_action("&Open script file",
            shortcut="Ctrl+O", slot=self.on_load_script_file, 
            tip="Load a script file")
        save_action = self.__make_action("&Save script file",
            shortcut="Ctrl+S", slot=self.on_save_script_file, 
            tip="Save script file")
        save_as_action = self.__make_action("&Save script file as ...",
            slot=self.on_save_as_script_file, 
            tip="Save script file as ...")
        close_action = self.__make_action("&Close script file",
            shortcut="Ctrl+W", slot=self.on_close_script_file,
            tip="Close script file")
        close_window_action = self.__make_action("Close scripting window",
            shortcut="Ctrl+Q", slot=self.close,
            tip="Close scripting window")

        self.__add_actions(self.file_menu, 
            (new_action, open_action, save_action, save_as_action, close_action, None, close_window_action)
        )

        run_script_action = self.__make_action("&Run script file",
            shortcut="Ctrl+R", slot=self.on_run_script_file,
            tip="Run script file")

        self.__add_actions(self.script_menu, 
            (run_script_action,)
        )

    def __add_actions(self, target, actions):
        for action in actions:
            if action is None:
                target.addSeparator()
            else:
                target.addAction(action)

    def __make_action(  self, text, slot=None, shortcut=None, 
                        icon=None, tip=None, checkable=False, 
                        signal="triggered()"):
        action = QAction(text, self)
        if icon is not None:
            action.setIcon(QIcon(":/%s.png" % icon))
        if shortcut is not None:
            action.setShortcut(shortcut)
        if tip is not None:
            action.setToolTip(tip)
            action.setStatusTip(tip)
        if slot is not None:
            self.connect(action, SIGNAL(signal), slot)
        if checkable:
            action.setCheckable(True)
        return action
