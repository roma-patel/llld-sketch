#!/bin/bash
cd /home/rpatel59/nlp/sketch-attr/
#python src/model.py sketch_sketchy_all_2 sketch /data/nlp/sketchy_splits/sketch/train/
python src/eval.py sketch_sketchy_all_1 /data/nlp/sketchy_splits/sketch/train/
python src/eval.py sketch_sketchy_all_2 /data/nlp/sketchy_splits/sketch/train/

#python src/eval.py sketch_sketchy_all_1 /data/nlp/sketchy_splits/sketch/train/

