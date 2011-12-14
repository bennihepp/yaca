# -*- coding: utf-8 -*-

"""
Provides a widget for editing python scripts.

@author: Benjamin Hepp
@copyright: Copyright (c) 2011 Benjamin Hepp
@license: BSD

All rights reserved. This program and the accompanying materials are made 
available under the terms of the FreeBSD which accompanies this distribution,
and is available at U{http://www.opensource.org/licenses/BSD-2-Clause}
"""

try:

    import qsci_script_file_widget

    ScriptFileMdiWidget = qsci_script_file_widget.QSciScriptFileMdiWidget
    ScriptFileWidget = qsci_script_file_widget.QSciScriptFileWidget

except:

    import os
    from PyQt4.QtCore import *
    from PyQt4.QtGui import *

    class ScriptFileMdiWidget(QMdiSubWindow):
        def __init__(self, parent=None):
            QMdiSubWindow.__init__(self, parent)
            self.script_file_widget = ScriptFileWidget(self)
            self.setWidget(self.script_file_widget)
            self.setAttribute(Qt.WA_DeleteOnClose)

    class ScriptFileWidget(QTextEdit):
    
        def __init__(self, parent=None, watch=True):
            QTextEdit.__init__(self, parent)
            self.__curFile = QString('')
            if watch:
                self.fileWatcher = QFileSystemWatcher()
                self.connect(self.fileWatcher, SIGNAL('fileChanged(const QString&)'), self.on_file_changed)
            else:
                self.fileWatcher = None
            #self.setWindowTitle(title)
            self.setCurrentFile(QString(''))
            self.setFontFamily('monospace')
            self.setTextInteractionFlags(Qt.TextEditorInteraction)
            self.show()
    
        def closeEvent(self, event):
            if (self.maybeSave()):
                event.accept()
            else:
                event.ignore()

        def filename(self):
            if self.__curFile:
                return os.path.basename(self.__curFile)
            else:
                return '<unnamed>'

        text = toPlainText
        #def text(self):
        #    return self.toPlainText()

        def on_open_file(self):
            file_choices = "Python script file (*.py)"
            path = QFileDialog.getOpenFileName(self, 
                                               'Open script file', '', 
                                               file_choices)
            if not path.isEmpty():
                self.loadFile(path)
                return True
            return False
    
        def on_save_file(self):
            return self.save()
    
        def on_save_as_file(self):
            return self.saveAs()
    
        def on_close_file(self):
            if self.maybeSave():
                #self.close()
                return True
            else:
                return False
    
        def on_file_changed(self, path):
            self.document().setModified()
            ret = QMessageBox.question(self, 'Script file: %s' % (self.__curFile),
                                'The file has been modified outside of this editor.\n' \
                                'Do you want to reload the file?',
                                QMessageBox.Yes | QMessageBox.No)
            # TODO: Check if file was removed or renamed
            if ret == QMessageBox.Yes:
                self.loadFile(path)
            else:
                self.fileWatcher.addPath(path)
    
        def loadFile(self, path):
            file = QFile(path)
            if not file.open(QFile.ReadOnly | QFile.Text):
                QMessageBox.warning(self, 'Script Widget',
                                    'Cannot open file %s:\n%s.' % path, file.errorString())
                return False
            sin = QTextStream(file)
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.setPlainText(sin.readAll())
            QApplication.restoreOverrideCursor()
            file.close()
            self.setCurrentFile(path)
            return True
    
        def setCurrentFile(self, path):
            self.document().setModified(False)
            self.setWindowModified(False)
            shownName = path
            if path.isEmpty():
                shownName = '<unnamed>'
            self.setWindowFilePath(shownName)
            qf = QFileInfo(path)
            self.setWindowTitle(qf.fileName())
            self.__curFile = path
            if self.fileWatcher is not None:
                for file in self.fileWatcher.files():
                    self.fileWatcher.removePath(file)
                if qf.exists() and qf.isFile() and qf.isReadable():
                    self.fileWatcher.addPath(path)
    
        def save(self):
            if self.__curFile.isEmpty():
                return self.saveAs()
            else:
                return self.saveFile(self.__curFile)
    
        def saveAs(self):
            path = QFileDialog.getSaveFileName(self)
            if path.isEmpty():
                return False
            return self.saveFile(path)
    
        def saveFile(self, path):
            file = QFile(path)
            if not file.open(QFile.WriteOnly | QFile.Text):
                QMessageBox.warning(self, 'Script file: %s' % (self.__curFile),
                                    'Cannot save file %s:\n%s.' % path, file.errorString())
                return False
            sout = QTextStream(file)
            QApplication.setOverrideCursor(Qt.WaitCursor)
            sout << self.toPlainText()
            QApplication.restoreOverrideCursor()
            file.close()
            self.setCurrentFile(path)
            return True
    
        def maybeSave(self):
            if self.document().isModified():
                ret = QMessageBox.warning(self, 'Script file: %s' % (self.__curFile),
                                    'The file has been modified.\n' \
                                    'Do you want to save your changes?',
                                    QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
                if ret == QMessageBox.Save:
                    return self.save()
                elif ret == QMessageBox.Cancel:
                    return False
            return True
    
        def keyPressEvent(self, event):
            '''
            Key press callback with some python script support.
    
            @return: True if event should not trickle.
            @rtype: boolean
            '''
            if event.key() == Qt.Key_Tab:
                self.insertPlainText(QString(4*' '))
            else:
                QTextEdit.keyPressEvent(self, event)
