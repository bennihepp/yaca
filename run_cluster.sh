#!/bin/sh

YACA_PREFIX="/g/pepperkok/hepp/yaca"

. ${YACA_PREFIX}/cluster_profile

python ${YACA_PREFIX}/main.py --no-opengl $1 $2 $3 $4 $5

