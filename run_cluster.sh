#!/bin/sh

PHENONICE_PREFIX="/g/pepperkok/hepp/PhenoNice"

. ${PHENONICE_PREFIX}/cluster_profile

python ${PHENONICE_PREFIX}/main.py --no-opengl $1 $2 $3 $4 $5

