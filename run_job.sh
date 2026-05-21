#!/usr/bin/env bash
set -euo pipefail

./run_ABCD.sh \
  --year "2017-2018" \
  --unblind 21 \
  --regions-scheme default \
  --flavour MET \
  --trigger notPhoton200_PFMET250 \
  --tree monopoles \
  --outdir /eos/user/t/tmenezes/www/Monopoles/ABCD/22Feb2026 \
  --filelist inputs_HEM.txt
