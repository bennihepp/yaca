# -*- coding: utf-8 -*-

"""
debug.py -- For debugging purposes.
"""

# This software is distributed under the FreeBSD License.
# See the accompanying file LICENSE for details.
# 
# Copyright 2011 Benjamin Hepp

__DEBUGGING__ = False
__REMOTE__ = False
__STATE__ = 'stopped'

def set_debugging(debugging=True, remote=False, suspend=False):
    global __DEBUGGING__, __REMOTE__, __STATE__
    if debugging:
        __STATE__ = 'running'
        if remote:
            import wingdbstub
            wingdbstub.Ensure()
            if suspend:
                suspend_debugging()
                __STATE__ = 'suspended'
    elif not debugging and __DEBUGGING__ and __REMOTE__:
        if wingdbstub.debugger != None:
            wingdbstub.debugger.StopDebug()
        __STATE__ = 'stopped'
    __DEBUGGING__ = debugging
    __REMOTE__ = remote

def is_running():
    global __DEBUGGING__, __REMOTE__, __STATE__
    return __STATE__ == 'running'
def is_stopped():
    global __DEBUGGING__, __REMOTE__, __STATE__
    return __STATE__ == 'stopped'
def is_suspended():
    global __DEBUGGING__, __REMOTE__, __STATE__
    return __STATE__ == 'suspended'
def get_debugging_state():
    global __DEBUGGING__, __REMOTE__, __STATE__
    return __STATE__

def is_debugging():
    global __DEBUGGING__, __REMOTE__, __STATE__
    return __DEBUGGING__

def is_debugging_remote():
    global __DEBUGGING__, __REMOTE__, __STATE__
    return __REMOTE__

def set_break():
    global __DEBUGGING__, __REMOTE__, __STATE__
    if __DEBUGGING__ and __REMOTE__:
        if wingdbstub.debugger != None:
            wingdbstub.debugger.Break()

def start_debugging():
    global __DEBUGGING__, __REMOTE__, __STATE__
    if __DEBUGGING__ and __REMOTE__:
        if wingdbstub.debugger != None:
            wingdbstub.debugger.StartDebug()
            __STATE__ = 'running'

def stop_debugging():
    global __DEBUGGING__, __REMOTE__, __STATE__
    if __DEBUGGING__ and __REMOTE__:
        if wingdbstub.debugger != None:
            wingdbstub.debugger.StopDebug()
            __STATE__ = 'stopped'

def resume_debugging():
    global __DEBUGGING__, __REMOTE__, __STATE__
    if __DEBUGGING__ and __REMOTE__:
        if wingdbstub.debugger != None:
            wingdbstub.debugger.ResumeDebug()
            __STATE__ = 'running'

def suspend_debugging():
    global __DEBUGGING__, __REMOTE__, __STATE__
    if __DEBUGGING__ and __REMOTE__:
        if wingdbstub.debugger != None:
            wingdbstub.debugger.SuspendDebug()
            __STATE__ = 'suspended'
