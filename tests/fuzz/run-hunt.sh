#!/bin/sh
# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

set -eu

if [ "$#" -ne 2 ]; then
    echo "usage: run-hunt.sh DRIVER DEFAULT-INPUT-FILE" >&2
    exit 2
fi

driver=$1
default_input=$2

exec "$driver" "${C50_HUNT_SEED:-1}" "${C50_HUNT_COUNT:-10000}" \
    "${C50_HUNT_INPUT:-$default_input}"
