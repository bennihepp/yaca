# -*- coding: utf-8 -*-

"""
command_interpreter.py -- Simple interactive python shell.
"""

# This software is distributed under the FreeBSD License.
# See the accompanying file LICENSE for details.
# 
# Copyright 2011 Benjamin Hepp

import sys

from thread_utils import Thread

#except:
import cmd

class ICmd(InteractiveSession, cmd.Cmd, Thread):

    def __init__(self, local_ns, global_ns, stdout=sys.stdout, stderr=sys.stderr, *args, **kwargs):
        #super(ICmd, self).__init__(*args, **kwargs)
        cmd.Cmd.__init__(self)
        Thread.__init__(self, **kwargs)
        self.__hooks = []
        self.__local_ns = local_ns
        self.__global_ns = global_ns
        self.__stdout = stdout
        self.__stderr = stderr
        if self.thread_support():
            self.start_method(self.__loop)

    def __loop(self):
        self.cmdloop('Starting interactive python session...\n')

    default = InteractiveSession.parse

    #def add_hook(self, callback):
        #self.__hooks.append(callback)

    #def remove_hook(self, callback):
        #self.__hooks.remove(callback)

    #def default(self, line):
        #try:
            #old_stdout = sys.stdout
            #old_stderr = sys.stderr
            #sys.stdout = self.__stdout
            #sys.stderr = self.__stderr
            #try:
                #tmp = None
                #if '_' in self.__local_ns:
                    #tmp = [ self.__local_ns[ '_' ] ]
                #if '_' in self.__global_ns:
                    #tmp = [ self.__global_ns[ '_' ] ]
                #line2 = '_ = ' + line
                #exec line2 in self.__global_ns, self.__local_ns
                #print >> self.__stdout, self.__local_ns[ '_' ]
                #self.__stdout.flush()
                #if tmp is not None:
                    #self.__local_ns[ '_' ] = tmp
            #except Exception, e:
                #exec line in self.__global_ns, self.__local_ns
        #except Exception, e:
            #print >> self.__stderr, 'Exception:', e
            #self.__stderr.flush()
        #finally:
            #for hook in self.__hooks:
                #hook()
            #sys.stdout = old_stdout
            #sys.stderr = old_stderr
