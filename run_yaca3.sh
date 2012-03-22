#!/bin/bash

PYTHONPATH="/g/pepperkok/hepp/code/snippets:${PYTHONPATH}"

export PYTHONPATH

python /g/pepperkok/hepp/yaca3/main.py "$@"

