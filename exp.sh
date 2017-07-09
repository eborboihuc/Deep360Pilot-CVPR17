inquire () {
    select yn in "Yes" "No"; do
        case $yn in
            Yes ) echo 1; exit;;
            No )  echo 0; exit;;
        esac
    done
}

# Check choices
echo "Do you wish to train this model?" 
_train=$( inquire )
echo "Do you wish to save this model after training?"
_save=$( inquire  )
echo "Do you wish to test this model?"
_test=$( inquire  )

_phase=('classify' 'regress')
_dir='parkour_16boxes_lam10.0'

for dirname in "${_phase[@]}"
do
    if [ ! -d ${dirname} ]; then
        echo "mkdir ${dirname}"
        mkdir ${dirname}
    else
        echo "${dirname} exist"
    fi
done

# Train Classify
if [ $_train = 1 ]; then
    python main.py --mode train --gpu 0 -d parkour -l 10 -b 16 -p classify
fi

if [ $_save = 1 ]; then
    cp checkpoint/${_dir}/parkour_lam1_classify_best_model.* checkpoint/${_dir}/checkpoint ${_phase[0]}
fi

# Test Classify result
if [ $_test = 1 ]; then
    python main.py --mode test --model ${_dir}/parkour_lam1_classify_best_model --gpu 0 -d parkour -l 10 -b 16 -p classify &> ${_phase[0]}/${_phase[0]}_test.log
fi

# Train Regress on Classify result
if [ $_train = 1 ]; then
    python main.py --mode train --gpu 0 -d parkour -l 10 -b 16 -p regress --model ${_dir}/parkour_lam1_classify_best_model
fi

if [ $_save = 1 ]; then
    cp checkpoint/${_dir}/parkour_lam10.0_regress_best_model.* checkpoint/${_dir}/checkpoint ${_phase[1]}
fi

# Test Regress result
if [ $_test = 1 ]; then
    python main.py --mode test --model ${_dir}/parkour_lam10.0_regress_best_model --gpu 0 -d parkour -l 10 -b 16 -p regress &> ${_phase[1]}/${_phase[1]}_test.log
fi
