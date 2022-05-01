#!/usr/bin/env bash
python run_lightning_pretrain.py --graph --output_dir ../saved/lightning-pre-s
python run_lightning_predict.py --graph --output_dir ../saved/lightning-predict-s --pretrain_dir ../saved/lightning-pre-s/checkpoints