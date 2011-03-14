#!/usr/bin/env python


PBS_script_template = \
"""#!/bin/bash
#PBS -M hepp@embl.de
#PBS -o /g/pepperkok/hepp/yaca/logs/cluster_job_out_%s
#PBS -e /g/pepperkok/hepp/yaca/logs/cluster_job_err_%s

/g/pepperkok/hepp/yaca/run_yaca_cluster.sh --batch --config-file %s
"""


import os
import sys


config_file = sys.argv[1]

name = os.path.splitext( os.path.split( config_file )[1] )[0]


import subprocess

PBS_script = PBS_script_template % ( name, name, config_file )

PBS_script_filename = '/g/pepperkok/hepp/yaca/jobs/yaca_cluster_job_%s' % name

f = open( PBS_script_filename, 'w' )
f.write( PBS_script )
f.close()

p = subprocess.Popen( [ 'ssh', 'sub-master', '/usr/pbs/bin/qsub', '-q', 'clng_new', PBS_script_filename ] )
p.wait()

