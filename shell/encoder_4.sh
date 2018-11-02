#!/bin/bash
cd /home/rpatel59/nlp/llld-sketch/
python main.py --epochs 100 --run_name 'sketch_sketchy_nall' --datadir /data/nlp/sketchy_splits/sketch_3/train/ --eval True --test_datadir /data/nlp/sketchy_splits/sketch_3/test/ 


