#!/usr/bin/env bash
set -euo pipefail

./run_ABCD.sh \
  --year "2017-2018" \
  --unblind 9904 \
  --regions-scheme default \
  --flavour MET \
  --trigger notPhoton200_PFMET250 \
  --tree monopoles \
  --outdir /eos/user/t/tmenezes/www/Monopoles/ABCD/18Jun2026_closure \
  --filelist inputs_HEM.txt
