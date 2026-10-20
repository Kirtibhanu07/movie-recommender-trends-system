#!/bin/sh
set -eu
python download_data.py
python train_evaluate.py
