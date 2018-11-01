#!/bin/bash
cd /home/rpatel59/nlp/sketch-attr/
#python src/model.py photo_sketchy_2 sketch /data/nlp/sketchy_splits/photo/train/
python src/eval.py photo_sketchy_1 /data/nlp/sketchy_splits/photo/train/
python src/eval.py photo_sketchy_2 /data/nlp/sketchy_splits/photo/train/

#python src/eval.py photo_sketchy_1 /data/nlp/sketchy_splits/photo/train/

