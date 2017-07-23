set -e

# ./run.sh {DOMAIN} {N_BOXES}
if [ "$#" -ne 2 ]; then
    echo "Usage: ${0} {DOMAIN} {N_BOXES}" >&2
    exit 1
fi
DOMAIN=${1}
N_BOXES=${2}

python parse_csv.py -d $DOMAIN

# """ In case you want to label new domain """
#sudo python parse_raw_label.py -d $DOMAIN -b $N_BOXES
#sudo python one_hot_convert.py -m data -d $DOMAIN -b $N_BOXES


python OF_360/divide_area_motion_salient_boxes.py -d $DOMAIN -b $N_BOXES
python OF_360/pruned_box_features.py -d $DOMAIN -b $N_BOXES
python OF_360/get_motion_features.py -d $DOMAIN -b $N_BOXES
python OF_360/hist_of_opt_flow/get_hof.py -d $DOMAIN -b $N_BOXES


# """ In case you need to generate test batches """
#cd batch_data/$DOMAIN
#cd ../..
#python mkdir.py -d $DOMAIN
#python gen_batch.py -d $DOMAIN -b $N_BOXES
#python test_gen_batch.py -d $DOMAIN -b $N_BOXES
