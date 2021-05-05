#!/bin/bash
# Arguments:
#   $1: binary file to profile
#   $2: output directory

EVENTS_TO_ANALYZE=("branch-misses," "cache-misses")

sudo perf -e ${EVENTS_TO_ANALYZE} | tee $2/$1