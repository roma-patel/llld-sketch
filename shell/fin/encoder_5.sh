#!/bin/bash
cd /home/rpatel59/nlp/llld-sketch/
python main.py --epochs 100 --run_name 'bm-sketch-attr' --datadir /data/nlp/sketchy_splits/sketch/train/ --eval True --test_datadir /data/nlp/sketchy_splits/sketch/test/ --fin_run 'attr-reconstr'




