#!/bin/bash
cd /home/rpatel59/nlp/llld-sketch/
#python main.py --epochs 100 --run_name 'photo_sketchy' --datadir /data/nlp/sketchy_splits/photo/train/ --eval True --test_datadir /data/nlp/sketchy_splits/photo/test/
#python main.py --epochs 100 --run_name 'photo_sketchy_word' --datadir /data/nlp/sketchy_splits/photo/train/ --eval True --test_datadir /data/nlp/sketchy_splits/photo/test/ 
python main.py --epochs 100 --run_name 'photo_sketchy_word_encoder' --datadir /data/nlp/sketchy_splits/photo/train/ --eval True --test_datadir /data/nlp/sketchy_splits/photo/test/ 







