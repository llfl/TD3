#!/bin/bash

#SEEDS=(443411 311204 214052 166452 415320 543117 838596 410234 419338 167558 692777 404628 266677 491666 830345 645725 101564 497637 138424 525446)
SEEDS=(692777 404628 266677 491666 830345 645725 101564 497637 138424 525446)

for seed in ${SEEDS[*]}
do
    python ./main.py --seed $seed
done