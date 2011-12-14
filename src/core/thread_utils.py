# -*- coding: utf-8 -*-

"""
thread_utils.py -- Thread wrapper and utilities.

This class provides a wrapper around the Qt Thread class QThread when available and otherwise
defines a stub implementation so that it can be used in headless mode.
"""

# This software is distributed under the FreeBSD License.
# See the accompanying file LICENSE for details.
# 
# Copyright 2011 Benjamin Hepp

import sys
import traceback

import debug

class AbstractThreadBase(object):
    
    __thread_support__ = False

    @classmethod
    def thread_support(cls):
        return cls.__thread_support__

    def start_method(self, method, *args):
        """Run a method within the thread
    
        Input parameters:
          - method: The method to be called
          - args: Arbitrary parameters to be passed to the method
        """
        # set the method to be run in the thread...
        self.__thread_method = method
        # and the parameters that should be passed to it
        self.__thread_args = args

        # start the thread
        self.start()

    def wait_safe(self):
        """Wait for the thread to finish and throw and exception if an exception was raised in the thread.
        """

        self.wait()

        if self.__exception != None:
            #print 'An exception was raised in a running thread:'
            sys.stderr.write( self.__traceback )
            sys.stderr.flush()
            #print self.__traceback
            raise Exception( 'An exception was raised in a running thread:' )
            #return False
            #raise self.__exception

        return True

    def run(self):
        """This is the method that is called when the thread is started
        """

        self.__result = None

        self.__exception = None

        if self.__thread_method != None:

            try:
                self.__result = self.__thread_method( *self.__thread_args )
            except Exception,e:
                if debug.is_debugging():
                    raise
                else:
                    self.__exception = e
                    self.__traceback = traceback.format_exc()

    def get_result(self):
        """Return the result, that was returned by the last thread method
        """

        self.wait_safe()

        return self.__result



if 'PyQt4.QtCore' in sys.modules and not debug.is_debugging():
    from PyQt4.QtCore import *

    class Thread(QThread, AbstractThreadBase):

        __thread_support__ = True

        def __init__(self, *args, **kwargs):
            super(Thread, self).__init__(*args, **kwargs)
            #QThread.__init__( self )
            self.mutex = QMutex()
            self.stop_running = False

        def stop(self):
            self.mutex.lock()
            self.stop_running = True
            self.mutex.unlock()

        def has_been_stopped(self):
            self.mutex.lock()
            stop_running = self.stop_running
            self.mutex.unlock()
            return stop_running

else:

    def SIGNAL(s):
        return s

    class Thread(AbstractThreadBase):

        __thread_support__ = False

        def __init__(self, *args, **kwargs):
            super(Thread, self).__init__(*args, **kwargs)
            self.__slots = {}
            self.stop_running = False

        def start(self):
            self.emit( 'started()' )
            self.run()
            self.emit( 'finished()' )

        def stop(self):
            self.stop_running = True

        def has_been_stopped(self):
            return self.stop_running

        def wait(self):
            return True

        def connect(self, obj, signal, slot):
            if not signal in self.__slots:
                self.__slots[ signal ] = []
            self.__slots[ signal ].append( slot )

        def disconnect(self, obj, signal, slot):
            if signal in self.__slots:
                self.__slots[signal].remove(slot)

        def emit(self, signal, *args):
            if signal in self.__slots:
                for slot in self.__slots[ signal ]:
                    slot( *args )

