#!/usr/bin/env bash
set -e
python -m src.data_prep --output data/processed/prefix_stories.json --n 50000
python -m src.train
