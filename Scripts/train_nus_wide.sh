#!/bin/bash

python train.py --gzsl \
--gammaD 10 --gammaG 10 \
--manualSeed 3483 --preprocessing --cuda \
--nepoch 50 \
--ndh 4096 \
--lr 0.0001 --classifier_lr 0.001 \
--lambda1 10 --critic_iter 5 \
--nclass_all 1006 --nseen_class 925 \
--batch_size 64 --workers 8 --attSize 300 \
--resSize 4096 \
--encoder_layer_sizes1 4096 --encoder_layer_sizes2 4096 \
--decoder_layer_sizes1 4096 --decoder_layer_sizes2 4096 \
--N 10 \
--hybrid_fusion \
--syn_num 1 \
--trimmed_train \
--hiddensize 8192 \
--per_seen 1.0 --per_unseen 0.50 --per_seen_unseen 0.50 \
--classifier_batch_size 64 --fake_batch_size 64 --classifier_epoch 50