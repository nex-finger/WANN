#!/bin/bash
cd /home/masuda/Docments/WANN/WANN/

. ./gitup.sh aaa master

python wann_train.py -n 11 -p p/masuda/biped1003.json -m p/propose/propose1.json

. ./gitup.sh aaa master

python wann_train.py -n 11 -p p/masuda/biped1003.json -m p/propose/propose1.json

. ./gitup.sh aaa master

python wann_train.py -n 11 -p p/masuda/biped1003.json -m p/propose/propose1.json

. ./gitup.sh aaa master

python wann_train.py -n 11 -p p/masuda/biped1003.json -m p/propose/propose1.json

. ./gitup.sh aaa master
