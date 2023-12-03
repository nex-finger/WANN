#!/bin/bash

prog[0]="python wann_train.py -n 11 -p p/masuda/biped1003.json -m p/propose/kizon.json"
prog[1]="python wann_train.py -n 11 -p p/masuda/biped1003.json -m p/propose/propose1.json"
prog[2]="python wann_train.py -n 11 -p p/masuda/biped1003.json -m p/propose/propose2.json"

while true
do
    git pull origin main

    # get time
    current_time=$(date +"%Y%m%d_%H%M%S")

    for i in {0..2}
    do
        execute="${prog[i]} > output/${current_time}_${i}.txt"
        echo $execute
        eval "$execute"
        sleep 1
    done

    echo $current_time
    echo uploading...
    
    git init
    git add -A
    git commit -m $current_time
    git remote rm master
    git remote add master git@github.com:nex-finger/WANN.git
    git push master

    echo done
done
