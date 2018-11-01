#!/bin/bash
cd /home/rpatel59/nlp/llld-sketch/
python main.py --epochs 100 --run_name 'sketch_sketchy' --datadir /data/nlp/sketchy_splits/sketch/train/ --eval True --test_datadir /data/nlp/sketchy_splits/sketch/test/ 

