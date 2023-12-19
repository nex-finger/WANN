#!/bin/bash

prog[0]="python wann_train.py -n 11 -p p/masuda/biped1003.json -m p/propose/propose1.json"
prog[1]="python wann_train.py -n 11 -p p/masuda/biped1003.json -m p/propose/propose2.json"
prog[2]="python wann_train.py -n 11 -p p/masuda/biped1003.json -m p/propose/propose3.json"
prog[3]="python wann_train.py -n 11 -p p/masuda/biped1003.json -m p/propose/propose4.json"

while true
do
    . ./gitup.sh aaa master

    # get time
    current_time=$(date +"%Y%m%d_%H%M%S")

    for i in {0..3}
    do
        execute="${prog[i]} > output/${current_time}_${i}.txt"
        echo $execute
        eval "$execute"
        sleep 1
    done
done
