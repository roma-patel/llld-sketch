#!/bin/bash
cd /home/rpatel59/nlp/llld-sketch/
python main.py --epochs 100 --run_name 'bm-photo-sketch-nall' --datadir /data/nlp/sketchy_splits/sketch_3/train/ --eval True --test_datadir /data/nlp/sketchy_splits/sketch_3/test/ --fin_run 'sketch-reconstr'







