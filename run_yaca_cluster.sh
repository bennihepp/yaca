#!/bin/bash

#JAVA_HOME=/struct/software/linux/jdk-6u18
#LD_LIBRARY_PATH=/struct/software/linux/jdk-6u18
#PATH=/struct/software/linux/jdk-6u18/bin:${PATH}
#PKG_CONFIG_PATH=/g/almf/software/pkgconfig

#PATH=/g/almf/software/bin:${PATH}
#LD_LIBRARY_PATH=/g/almf/software/lib:${LD_LIBRARY_PATH}
#PYTHONPATH=/g/almf/software/CP2C/lib/python2.5

PREFIX=/g/pepperkok/hepp/cluster

PATH=$PREFIX/bin
PATH=$PREFIX/qt/bin:${PATH}
LD_LIBRARY_PATH=$PREFIX/qt/lib
LD_LIBRARY_PATH=$PREFIX/lib:${LD_LIBRARY_PATH}
LD_LIBRARY_PATH=$PREFIX/lib64:${LD_LIBRARY_PATH}
PYTHONPATH=$PREFIX/lib/python2.6/site-packages
PYTHONPATH=$PREFIX/lib64/python2.6/site-packages:${PYTHONPATH}

#export JAVA_HOME
export PATH
export LD_LIBRARY_PATH
export PYTHONPATH

python2.6 /g/pepperkok/hepp/yaca_working/main.py --no-opengl "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8"

