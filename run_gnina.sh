#!/bin/bash
export LD_LIBRARY_PATH=/tmp/libs:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
/tmp/gnina "$@"
