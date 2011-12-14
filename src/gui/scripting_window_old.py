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

from ..core.command_interpreter import InteractiveSession


class InteractivePythonWidget(QPlainTextEdit):
    def __init__(self, user_ns, global_ns, title='Interactive session', parent=None):
        super(InteractivePythonWidget, self).__init__(parent)
        self.__banner = '>>> '
        #self.setReadOnly(False)
        self.setTabStopWidth(4)
        self.setWindowTitle(title)
        #self.__keytimer = QTimer()
        #self.connect(self.__keytimer, SIGNAL('timeout()'), self.__on_timeout)
        #self.__key = 'a'
        self.__linecache = []
        self.__cacheIndex = None
        #self.__readfd, self.__writefd = os.pipe()
        #self.__readf = os.fdopen(self.__readfd, 'r')
        #self.__writef = os.fdopen(self.__writefd, 'w')
        self.__sio = cStringIO.StringIO()
        self.__isession = InteractiveSession(user_ns, global_ns, stdout=self.__sio, stderr=self.__sio)
        self.setPlainText(self.__banner)
        self.__moveCursorToEnd()

    def __appendText(self, text):
        old_text = self.toPlainText()
        print 'old_text:', str(old_text)
        print 'appending %s to %s' % (str(text),str(old_text))
        old_text.append(text)
        self.setPlainText(old_text)
        self.moveCursor(QTextCursor.End)

    def __removeText(self, sl):
        old_text = self.toPlainText()
        print 'old_text:', str(old_text)
        start, stop = sl.start, sl.stop
        if start == None:
            start = 0
        if start < 0:
            start = old_text.length() + start
        if stop == None:
            stop = old_text.length()
        if stop < 0:
            stop = old_text.length() + stop
        #if step < 0:
        #    tmp = stop
        #    stop = start
        #    start = tmp
        text = old_text.remove(start, stop-start)
        self.setPlainText(text)
        self.__moveCursorToEnd()

    def __moveCursorToEnd(self):
        self.moveCursor(QTextCursor.End)
    def __moveCursorToNextChar(self):
        self.moveCursor(QTextCursor.NextCharacter)
    def __moveCursorToPrevChar(self):
        self.moveCursor(QTextCursor.PreviousCharacter)

    #def __on_timeout(self):
    #    self.__appendText(self.__key)

    def __replaceLastLine(self, line):
        text = self.toPlainText()
        lines = text.split('\n')
        lines.replace(len(lines) - 1, line)
        text = lines.join('\n')
        self.setPlainText(text)
        self.__moveCursorToEnd()

    def __getLastLine(self):
        text = self.toPlainText()
        lines = text.split('\n')
        line = str(lines[len(lines) - 1])
        return line

    def keyPressEvent(self, event):
        print 'keypress: %d,%s' % (event.key(), event.text())
        if event.key() == Qt.Key_Return:
            print 'ENTER'
            text = self.toPlainText()
            print 'text:', str(text)
            lines = text.split('\n')
            line = str(lines[len(lines) - 1])
            if len(line) > 0:
                if len(self.__linecache) > 0:
                    self.__linecache[0] = line
                else:
                    self.__linecache.append(line)
            #self.__linecache.insert(0, line)
            if line.startswith(self.__banner):
                line = line[len(self.__banner):]
            line = line.strip()
            self.__cacheIndex = None
            self.__sio.reset()
            self.__sio.truncate()
            self.__isession.parse(line)
            output = self.__sio.getvalue()
            self.__appendText('\n')
            self.__appendText(output)
            if len(output) > 0 and output[-1] != '\n':
                self.__appendText('\n')
            self.__appendText(self.__banner)
            self.__moveCursorToEnd()
        elif event.key() == Qt.Key_Backspace:
            print 'BACKSPACE'
            text = self.toPlainText()
            lines = text.split('\n')
            line = str(lines[len(lines) - 1])
            if line != self.__banner:
                pos = self.textCursor().position()
                if pos > 0:
                    if self.toPlainText()[pos - 1] != '\n':
                        sl = slice(pos-1, pos)
                        self.__removeText(slice(-1,None))
            self.__moveCursorToPrevChar()
        elif event.key() == Qt.Key_Delete:
            pass
        elif event.key() == Qt.Key_Tab:
            pass
        elif event.key() == Qt.Key_Up:
            if self.__cacheIndex == None:
                self.__linecache.insert(0, self.__getLastLine())
                self.__cacheIndex = 0
            self.__cacheIndex += 1
            if self.__cacheIndex >= len(self.__linecache):
                self.__cacheIndex = len(self.__linecache) - 1
            self.__replaceLastLine(self.__linecache[self.__cacheIndex])
            self.__moveCursorToEnd()
        elif event.key() == Qt.Key_Down:
            if self.__cacheIndex == None:
                self.__linecache.insert(0, self.__getLastLine())
                self.__cacheIndex = 0
            self.__cacheIndex -= 1
            if self.__cacheIndex < -1:
                self.__cacheIndex = -1
                line = ''
            else:
                line = self.__linecache[self.__cacheIndex]
            self.__replaceLastLine(line)
            self.__moveCursorToEnd()
        elif event.text().length() > 0 and QChar(event.text()[0]).isPrint():
            self.__appendText(event.text())
            if self.__cacheIndex == 0:
                self.__linecache[self.__cacheIndex] = self.__getLastLine()
            self.__moveCursorToNextChar()
        else:
            super(InteractivePythonWidget, self).keyPressEvent(event)
        #self.__key = event.text()
        #self.__keytimer.start(500)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Backspace or \
           event.key() == Qt.Key_Delete or event.key() == Qt.Key_Tab or \
           (event.text().length() > 0 and QChar(event.text()[0]).isPrint()):
            pass
        else:
            super(InteractivePythonWidget, self).keyReleaseEvent(event)
        #self.__key = event.text()
        #if self.__keytimer.isActive():
        #    self.__keytimer.stop()
        

class ScriptingWindow(QMainWindow):
    def __init__(self, user_ns, global_ns, parent=None):
        super(ScriptingWindow, self).__init__(parent)
        self.setWindowTitle('Scripting')
        self.__user_ns = user_ns
        self.__global_ns = global_ns
        self.__build_widget()

    def __build_widget(self):
        self.__mdiarea = QMdiArea()
        interactive_python_widget = InteractivePythonWidget(self.__user_ns, self.__global_ns,
                                                            title='Interactive session #1')
        self.__mdiarea.addSubWindow(interactive_python_widget)
        self.setCentralWidget(self.__mdiarea)
