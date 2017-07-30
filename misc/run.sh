set -e

# ./run.sh {DOMAIN} {N_BOXES}
if [ "$#" -ne 2 ]; then
    echo "Usage: ${0} {DOMAIN} {N_BOXES}" >&2
    exit 1
fi
DOMAIN=${1}
N_BOXES=${2}

# Function
echo_time () {
    echo [$(date)] ${1}
}

# Mkdir
_dirs=('dataset' 'metadata')
for dirname in "${_dirs[@]}"
do
    if [ ! -d ${dirname} ]; then
        echo_time "mkdir ${dirname}"
        mkdir ${dirname}
    else
        echo_time "${dirname} exist"
    fi
done

python parse_csv.py -d $DOMAIN

# """ In case you want to label new domain """
#python parse_raw_label.py -d $DOMAIN -b $N_BOXES
#python one_hot_convert.py -m data -d $DOMAIN -b $N_BOXES

# """ Convert flow and apply to features """
echo_time "python Deep360Pilot-optical-flow/compute_video_flow.py -d $DOMAIN"
python Deep360Pilot-optical-flow/compute_video_flow.py -d $DOMAIN
echo_time "python Deep360Pilot-optical-flow/divide_area_motion_salient_boxes.py -d $DOMAIN -b $N_BOXES"
python Deep360Pilot-optical-flow/divide_area_motion_salient_boxes.py -d $DOMAIN -b $N_BOXES
echo_time "python Deep360Pilot-optical-flow/pruned_box_features.py -d $DOMAIN -b $N_BOXES"
python Deep360Pilot-optical-flow/pruned_box_features.py -d $DOMAIN -b $N_BOXES
echo_time "python Deep360Pilot-optical-flow/get_motion_features.py -d $DOMAIN -b $N_BOXES"
python Deep360Pilot-optical-flow/get_motion_features.py -d $DOMAIN -b $N_BOXES
echo_time "python Deep360Pilot-optical-flow/hist_of_opt_flow/get_hof.py -d $DOMAIN -b $N_BOXES"
python Deep360Pilot-optical-flow/hist_of_opt_flow/get_hof.py -d $DOMAIN -b $N_BOXES


# """ In case you need to generate test batches """
#cd data/${DOMAIN}_${N_BOXES}
#python mkdir.py -d $DOMAIN
#python gen_batch.py -d $DOMAIN -b $N_BOXES
#python test_gen_batch.py -d $DOMAIN -b $N_BOXES
