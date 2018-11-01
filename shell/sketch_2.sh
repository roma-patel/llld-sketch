#!/bin/bash
cd /home/rpatel59/nlp/sketch-attr/
#python src/model.py sketch_sketchy_2 sketch /data/nlp/sketchy_splits/sketch_3/train/
python src/eval.py sketch_sketchy_1 /data/nlp/sketchy_splits/sketch_3/train/
python src/eval.py sketch_sketchy_2 /data/nlp/sketchy_splits/sketch_3/train/

#python src/eval.py sketch_sketchy_1 /data/nlp/sketchy_splits/sketch_3/train/

