#!/usr/bin/env python2.6

import sys

if __name__ == '__main__':

    headless = False
    batch = False
    log_file = None
    log_id = 'yaca'

    j = 1
    prev_len = len( sys.argv ) + 1
    while len( sys.argv ) != prev_len:
        prev_len = len( sys.argv )
        for i in xrange( j, len( sys.argv ) ):
            j = i
            arg = sys.argv[ i ]
            if arg == '':
                del sys.argv[ i ]
                break
            elif arg == '--headless':
                headless = True
                del sys.argv[ i ]
                break
            elif arg == '--batch':
                batch = True
                del sys.argv[ i ]
                break
            elif arg == '--log-file':
                log_file = sys.argv[ i+1 ]
                del sys.argv[ i+1 ]
                del sys.argv[ i ]
                break
            elif arg == '--log-id':
                log_id = sys.argv[ i+1 ]
                del sys.argv[ i+1 ]
                del sys.argv[ i ]
                break

    if batch or headless:
        for i in xrange( 1, len( sys.argv ) ):
            arg = sys.argv[ i ]
            if arg == '--no-opengl':
                del sys.argv[ i ]
                break

    if batch:

        log = None
        #if log_file:
        #    log = file( log_file, 'w' )

        try:
            from src import main_batch
        except:
            if log_file:
                import traceback
                import datetime
                t = traceback.format_exc()
                t = t.replace( '\n', '\n  ' )
                t = '  ' + t
 
                f = open( log_file, 'a' )
                dt_now = datetime.datetime.now()
                f.write( ( 'Error from %s at %s...:\n' % ( log_id, dt_now ) ) + t + '\n' )
                f.close()
            else:
                sys.stderr.write( 'No log file specified\n' )
            raise

    elif headless:

        from src import main_headless

    else:

        from src import main_gui

        main_gui.run()

