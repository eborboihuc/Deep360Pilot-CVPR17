#!/bin/bash

# Usage: ./testall.sh {gpu} {show or run}
if [ "$#" -ne 2 ]; then
    echo "Usage: ${0} SHOW(1)_OR_RUN(0)_OR_BOTH(2) GPU(0~N) " >&2
    exit 1
fi

# Variables
_domains=('parkour' 'bmx' 'skate' 'dance' 'basketball')
_show=${1} # RUN '0', SHOW '1', BOTH '2'
_gpu=${2} #'0', '1', '2', '3'
_phase=('classify' 'regress')

# Mkdir
for _domain in "${_domains[@]}"
do
    if [ ${_show} = 0 ] || [ ${_show} = 2 ]; then
        _dir=${_domain}'_16boxes_lam10.0'
        for dirname in "${_phase[@]}"
        do
            if [ ! -d ${_domain}_${dirname} ]; then
                echo "mkdir ${_domain}_${dirname}"
                mkdir ${_domain}_${dirname}
            else
                echo "${_domain}_${dirname} exist"
            fi
        done

        echo python main.py --mode test --model checkpoint/${_dir}/${_domain}_lam1_classify_best_model --gpu ${_gpu} -d ${_domain} -l 10 -b 16 -p classify &> ${_domain}_${_phase[0]}/${_phase[0]}_test.log
        python main.py --mode test --model checkpoint/${_dir}/${_domain}_lam1_classify_best_model --gpu ${_gpu} -d ${_domain} -l 10 -b 16 -p classify &> ${_domain}_${_phase[0]}/${_phase[0]}_test.log
        
        echo python main.py --mode test --model checkpoint/${_dir}/${_domain}_lam10.0_regress_best_model --gpu ${_gpu} -d ${_domain} -l 10 -b 16 -p regress &> ${_domain}_${_phase[1]}/${_phase[1]}_test.log
        python main.py --mode test --model checkpoint/${_dir}/${_domain}_lam10.0_regress_best_model --gpu ${_gpu} -d ${_domain} -l 10 -b 16 -p regress &> ${_domain}_${_phase[1]}/${_phase[1]}_test.log
    fi

    if [ ${_show} = 1 ] || [ ${_show} = 2 ]; then

        echo ${_domain}_${_phase[0]}
        tail ${_domain}_${_phase[0]}/${_phase[0]}_test.log
        echo ${_domain}_${_phase[1]}
        tail ${_domain}_${_phase[1]}/${_phase[1]}_test.log
    fi

done
