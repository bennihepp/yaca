#!/bin/bash

#JAVA_HOME=/struct/software/linux/jdk-6u18
#LD_LIBRARY_PATH=/struct/software/linux/jdk-6u18
#PATH=/struct/software/linux/jdk-6u18/bin:${PATH}
#PKG_CONFIG_PATH=/g/almf/software/pkgconfig

#PATH=/g/almf/software/bin:${PATH}
#LD_LIBRARY_PATH=/g/almf/software/lib:${LD_LIBRARY_PATH}
#PYTHONPATH=/g/almf/software/CP2C/lib/python2.5

PREFIX=/g/pepperkok/hepp/cluster

#PATH=$PREFIX/bin
#PATH=$PREFIX/qt/bin:${PATH}
#LD_LIBRARY_PATH=$PREFIX/qt/lib
#LD_LIBRARY_PATH=$PREFIX/lib:${LD_LIBRARY_PATH}
#LD_LIBRARY_PATH=$PREFIX/lib64:${LD_LIBRARY_PATH}
#PYTHONPATH=$PREFIX/lib/python2.6/site-packages
#PYTHONPATH=$PREFIX/lib64/python2.6/site-packages:${PYTHONPATH}

PYTHONPATH="/g/pepperkok/hepp/code/snippets:${PYTHONPATH}"

LD_LIBRARY_PATH=/g/software/linux/pack/qt-4.7.3/lib:$LD_LIBRARY_PATH
PYTHONPATH=/g/software/linux/pack/pyqt-4.8.4/lib/python2.7/site-packages:/g/software/linux/pack/sip-4.12.3/lib/python2.7/site-packages:$PYTHONPATH

#export JAVA_HOME
export PATH
export LD_LIBRARY_PATH
export PYTHONPATH

source /etc/profile

#python2.6 /g/pepperkok/hepp/yaca/main.py --no-opengl "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9"
python-2.7 /g/pepperkok/hepp/yaca3/main.py --no-opengl "$@"

