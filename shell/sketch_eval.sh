#!/bin/bash
cd /home/rpatel59/nlp/sketch-attr/
python src/eval.py sketch_tu_int_3 sketch /data/nlp/tu_splits/train/
python src/eval.py photo_sketchy_3 sketch /data/nlp/sketchy_splits/photo/train/
python src/eval.py sketch_sketchy_3 sketch /data/nlp/sketchy_splits/sketch_1/train/
python src/eval.py sketch_sketchy_4 sketch /data/nlp/sketchy_splits/sketch_3/train/
python src/eval.py sketch_sketchy_5 sketch /data/nlp/sketchy_splits/sketch_3/train/
