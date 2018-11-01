#!/bin/bash
cd /home/rpatel59/nlp/sketch-attr/
#python src/model.py sketch_tu_int_2 sketch /data/nlp/tu_splits/train/
python src/eval.py sketch_tu_int_1 /data/nlp/tu_splits/train/
python src/eval.py sketch_tu_int_2 /data/nlp/tu_splits/train/

#python src/eval.py sketch_tu_int_1 /data/nlp/tu_splits/train/

