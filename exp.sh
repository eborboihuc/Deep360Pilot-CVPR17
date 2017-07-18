#!/bin/bash

# Usage: ./exp.sh {domain} {gpu}
if [ "$#" -ne 2 ]; then
    echo "Usage: ${0} DOMAIN GPU" >&2
    exit 1
fi

# Variables
_domain=${1} #'parkour'
_gpu=${2} #'0'
_phase=('classify' 'regress')
_dir=${_domain}'_16boxes_lam10.0'

# Function
inquire () {
    select yn in "Yes" "No"; do
        case $yn in
            Yes ) echo 1; exit;;
            No )  echo 0; exit;;
        esac
    done
}

echo_time () {
    echo [$(date)] ${1}
}

# Check choices
echo_time "Do you wish to train this model?" 
_train=$( inquire )
echo_time "Do you wish to save this model after training?"
_save=$( inquire  )
echo_time "Do you wish to test this model?"
_test=$( inquire  )

# Mkdir
for dirname in "${_phase[@]}"
do
    if [ ! -d ${_domain}_${dirname} ]; then
        echo_time "mkdir ${_domain}_${dirname}"
        mkdir ${_domain}_${dirname}
    else
        echo_time "${_domain}_${dirname} exist"
    fi
done

# Train Classify
if [ $_train = 1 ]; then
    echo_time "python main.py --mode train --gpu ${_gpu} -d ${_domain} -l 10 -b 16 -p classify"
    python main.py --mode train --gpu ${_gpu} -d ${_domain} -l 10 -b 16 -p classify
fi

if [ $_save = 1 ]; then
    echo_time "cp checkpoint/${_dir}/${_domain}_lam1_classify_best_model.* checkpoint/${_dir}/checkpoint ${_domain}_${_phase[0]}"
    cp checkpoint/${_dir}/${_domain}_lam1_classify_best_model.* checkpoint/${_dir}/checkpoint ${_domain}_${_phase[0]}
fi

# Test Classify result
if [ $_test = 1 ]; then
    echo_time "python main.py --mode test --model checkpoint/${_dir}/${_domain}_lam1_classify_best_model --gpu ${_gpu} -d ${_domain} -l 10 -b 16 -p classify &> ${_domain}_${_phase[0]}/${_phase[0]}_test.log"
    python main.py --mode test --model checkpoint/${_dir}/${_domain}_lam1_classify_best_model --gpu ${_gpu} -d ${_domain} -l 10 -b 16 -p classify &> ${_domain}_${_phase[0]}/${_phase[0]}_test.log
fi

# Train Regress on Classify result
if [ $_train = 1 ]; then
    echo_time "python main.py --mode train --gpu ${_gpu} -d ${_domain} -l 10 -b 16 -p regress --model checkpoint/${_dir}/${_domain}_lam1_classify_best_model"
    python main.py --mode train --gpu ${_gpu} -d ${_domain} -l 10 -b 16 -p regress --model checkpoint/${_dir}/${_domain}_lam1_classify_best_model
fi

if [ $_save = 1 ]; then
    echo_time "cp checkpoint/${_dir}/${_domain}_lam10.0_regress_best_model.* checkpoint/${_dir}/checkpoint ${_domain}_${_phase[1]}"
    cp checkpoint/${_dir}/${_domain}_lam10.0_regress_best_model.* checkpoint/${_dir}/checkpoint ${_domain}_${_phase[1]}
fi

# Test Regress result
if [ $_test = 1 ]; then
    echo_time "python main.py --mode test --model checkpoint/${_dir}/${_domain}_lam10.0_regress_best_model --gpu ${_gpu} -d ${_domain} -l 10 -b 16 -p regress &> ${_domain}_${_phase[1]}/${_phase[1]}_test.log"
    python main.py --mode test --model checkpoint/${_dir}/${_domain}_lam10.0_regress_best_model --gpu ${_gpu} -d ${_domain} -l 10 -b 16 -p regress &> ${_domain}_${_phase[1]}/${_phase[1]}_test.log
fi
