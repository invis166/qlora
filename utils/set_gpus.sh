#!/bin/bash

# This scripts sets CUDA_VISIBLE_DEVICES and CUDA_DEVICE_ORDER

# Function to check whether this script is sourced or not
is_sourced() {
	if [ -n "$ZSH_VERSION" ]; then
	        case $ZSH_EVAL_CONTEXT in *:file:*) return 0;; esac
	else  # Add additional POSIX-compatible shell names here, if needed.
	        case ${0##*/} in dash|-dash|bash|-bash|ksh|-ksh|sh|-sh) return 0;; esac
	fi
	return 1  # NOT sourced.
}

# Setting variable with this value
is_sourced && sourced=1 || sourced=0

if [[ "$sourced" -eq "0" ]]; then
	echo "This script should be sourced"
	echo "usage: source $0 [CUDA_VISIBLE_DEVICES]"
	exit 1
fi

if [[ $# -ne 1 ]]; then
    echo "Not enough arguments"
    echo "usage: source ${BASH_SOURCE[0]} [CUDA_VISIBLE_DEVICES]"
    return
fi

# Setting GPUs to use
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="$1"
