#!/bin/bash

SEEDS=(127862 415094 324976 820140 902097)

for seed in ${SEEDS[*]}
do
    python ./main.py --seed $seed
done