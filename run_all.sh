#!/usr/bin/env bash
# Run every step in video order and print what each one finds.
# Takes about one minute after the download. Step 5 only prints the log of our own
# training run; the trained vectors are already in vectors.npy.
# To train them yourself (about 15 minutes): python3 step5_train.py
set -e
cd "$(dirname "$0")"
echo "## get_data"; python3 get_data.py
for step in step1_vectors step2_pairs step3_model step4_negative; do
  echo "## $step"; python3 $step.py
done
echo "## step5_train --show"; python3 step5_train.py --show
for step in step6_look step7_test; do
  echo "## $step"; python3 $step.py
done
echo "## solutions/window"; python3 solutions/window.py
echo "## tests"; python3 tests.py
