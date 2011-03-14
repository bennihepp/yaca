#!/bin/bash
#PBS -M hepp@embl.de
#PBS -o /g/pepperkok/yaca/logs/cluster_job_out_$PBS_JOBID
#PBS -e /g/pepperkok/yaca/logs/cluster_job_err_$PBS_JOBID

. /g/pepperkok/hepp/yaca/cluster_profile

python /g/pepperkok/hepp/yaca/run_yaca_cluster.sh $1 $2 $3 $4 $5

