#!/bin/bash

# with graph
echo "--model_name final-01-graph-gt --graph --do_train --enable_progress_bar=False"
python run_lightning_tandem.py --model_name final-01-graph-gt --graph --do_train --enable_progress_bar=False
echo "--model_name final-01-graph-pret-gt --graph --do_train --no_pretrain --enable_progress_bar=False"
python run_lightning_tandem.py --model_name final-01-graph-pret-gt --graph --do_train --no_pretrain --enable_progress_bar=False

# without graph
echo "--model_name final-01-gt --do_train --enable_progress_bar=False"
python run_lightning_tandem.py --model_name final-01-gt --do_train --enable_progress_bar=False
echo "--model_name final-01-pret-gt --do_train --no_pretrain --enable_progress_bar=False"
python run_lightning_tandem.py --model_name final-01-pret-gt --do_train --no_pretrain --enable_progress_bar=False
