#!/bin/bash
# Runs minimal examples to make sure they are working properly. Intended to be
# run from the root directory of the repo. This script takes a few minutes to
# run.
#
# Usage:
#   bash tests/examples.sh

set -e  # Exit if any of the commands fail.
set -x  # Print out commands as they are run.

TMPDIR="./examples_tmp"
if [ ! -d "${TMPDIR}" ]; then
  mkdir "${TMPDIR}"
fi

function install_deps() {
  install_cmd=$(grep "pip install" "$1")
  $install_cmd
}

# Single-threaded for consistency.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1


# Add script calls here. Example from pyribs:

# sphere.py
# install_deps examples/sphere.py
# SPHERE_OUTPUT="${TMPDIR}/sphere_output"
# python examples/sphere.py map_elites --itrs 10 --outdir "${SPHERE_OUTPUT}"
# python examples/sphere.py line_map_elites --itrs 10 --outdir "${SPHERE_OUTPUT}"


# Cleanup.
rm -rf $TMPDIR
echo "Success in $SECONDS seconds"
