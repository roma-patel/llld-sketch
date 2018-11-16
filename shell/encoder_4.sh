#!/bin/bash
cd /home/rpatel59/nlp/llld-sketch/
#python main.py --epochs 100 --run_name 'sketch_tu_int' --datadir /data/nlp/tu_splits/train/ --eval True --test_datadir /data/nlp/tu_splits/test/
python main.py --epochs 100 --run_name 'sketch_tu_int_word' --datadir /data/nlp/tu_splits/train/ --eval True --test_datadir /data/nlp/tu_splits/test/


