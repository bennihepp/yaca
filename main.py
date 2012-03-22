#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
main.py -- This module is the entry point of the application.
"""

# This software is distributed under the FreeBSD License.
# See the accompanying file LICENSE for details.
# 
# Copyright 2011 Benjamin Hepp

import sys
import argparse

import src.core.debug

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Yaca. Yet another cell analyser.')
    parser.add_argument('-b', '--headless', dest='mode', action='store_const',
                        const='headless', help='Run in headless mode')
    parser.add_argument('-i', '--interactive', dest='mode', action='store_const',
                        const='interactive', default='interactive',
                        help='Run in interactive mode (default)')
    parser.add_argument('--log-file', help='Log file.')
    parser.add_argument('--log-id', help='ID for the log file.')
    parser.add_argument('-d', '--debug', dest='debug_mode', action='append_const',
                        default=[], const='debug', help='Enable debug mode.')
    parser.add_argument('--debug-remote', dest='debug_mode', action='append_const',
                        const='remote', help='Enable remote debug mode.')
    parser.add_argument('--debug-suspend', dest='debug_mode', action='append_const',
                        const='suspend', help='Suspend debug mode.')
    parser.add_argument('--version', action='version', version='%(prog)s 0.3')
    parser.add_argument('--opengl', dest='opengl', action='store_true',
                        help='Enable OpenGL.')
    parser.add_argument('--no-opengl', dest='opengl', action='store_false',
                        help='Disable OpenGL.')
    parser.add_argument('-f', '--project-file', dest='project_file',
                        default=None, help='Yaca project file.')
    parser.add_argument('-r', '--run-filtering', dest='run_filtering', action='store_true',
                        default=False, help='Immediately run pre-filtering.')

    args = parser.parse_args()

    """
    headless = False
    batch = False
    interactive = False
    log_file = None
    log_id = 'yaca'
    debug = False
    debug_remote = False
    debug_suspend = False

    j = 1
    prev_len = len(sys.argv) + 1
    while len(sys.argv) != prev_len:
        prev_len = len(sys.argv)

        for i in xrange(j, len(sys.argv)):
            j = i
            arg = sys.argv[i]
            #if arg == '':
            #    del sys.argv[i]
            #    break
            if arg == '--headless':
                headless = True
                del sys.argv[i]
                break
            elif arg == '--interactive':
                interactive = True
                del sys.argv[i]
                break
            elif arg == '--batch':
                batch = True
                del sys.argv[i]
                break
            elif arg == '--log-file':
                log_file = sys.argv[i+1]
                del sys.argv[i+1]
                del sys.argv[i]
                break
            elif arg == '--log-id':
                log_id = sys.argv[i+1]
                del sys.argv[i+1]
                del sys.argv[i]
                break
            elif arg == '--debug':
                debug = True
                del sys.argv[i]
                break
            elif arg == '--debug-remote':
                debug = True
                debug_remote = True
                del sys.argv[i]
                break
            elif arg == '--debug-suspend':
                debug = True
                debug_suspend = True
                del sys.argv[i]
                break

    src.core.debug.set_debugging(debug, debug_remote, debug_suspend)

    if batch or headless or interactive:
        for i in xrange(1, len(sys.argv)):
            arg = sys.argv[i]
            if arg == '--no-opengl':
                del sys.argv[i]
                break
    """

    debug = False
    debug_remote = False
    debug_suspend = False
    debug = 'debug' in args.debug_mode
    debug_remote = 'remote' in args.debug_mode
    debug_suspend = 'suspend' in args.debug_mode
    src.core.debug.set_debugging(debug, debug_remote, debug_suspend)

    if args.mode == 'headless':

        log = None
        #if log_file:
        #    log = file(log_file, 'w')

        try:
            from src import main_batch
        except:
            if log_file:
                import traceback
                import datetime
                t = traceback.format_exc()
                t = t.replace('\n', '\n  ')
                t = '  ' + t
 
                dt_now = datetime.datetime.now()
                error_str = ('Error from %s at %s...:\n' % (log_id, dt_now)) + t + '\n'
                f = open(log_file, 'a')
                f.write(error_str)
                f.close()
            else:
                sys.stderr.write('No log file specified\n')
            raise

    #elif headless:

        #from src import main_headless

        #main_headless.run(args)

    #elif interactive:

        #from src import main_interactive

        #main_interactive.run(args)

    #else:

        #from src import main_gui

        #main_gui.run(args)

    if args.mode == 'headless':

        from src import main_headless

        main_headless.run(args)

    else:

        from src import main_gui

        main_gui.run(args)
