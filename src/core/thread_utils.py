import sys

if 'PyQt4.QtCore' in sys.modules:
    from PyQt4.QtCore import *

    class Thread(QThread):

        def __init__(self):
            QThread.__init__( self )
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

    class Thread(object):

        def __init__(self):
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

        def emit(self, signal, *args):
            if signal in self.__slots:
                for slot in self.__slots[ signal ]:
                    slot( *args )

