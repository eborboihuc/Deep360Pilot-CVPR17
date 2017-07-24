# make subsequent commands which fail will cause the shell script to exit immediately
set -e

# delete redundant lines in dataset_domain.csv
DOMAIN=main
N_BOXES=16
#sudo python parse_csv.py -d $DOMAIN
#sudo python parse_raw_label.py -d $DOMAIN -b $N_BOXES
#sudo python one_hot_convert.py -m data -d $DOMAIN -b $N_BOXES
sudo python ~/Workspace/OF_360/divide_area_motion_salient_boxes.py -d $DOMAIN -b $N_BOXES
sudo python ~/Workspace/OF_360/pruned_box_features.py -d $DOMAIN -b $N_BOXES
sudo python ~/Workspace/OF_360/get_motion_features.py -d $DOMAIN -b $N_BOXES
sudo python ~/Workspace/OF_360/hist_of_opt_flow/get_hof.py -d $DOMAIN -b $N_BOXES
#cd batch_data/$DOMAIN 
#find . -name "*.npy" -type f -delete
#cd ../..
#sudo python ./batch_data/mkdir.py -d $DOMAIN
#sudo python pilot_gen_batch.py -d $DOMAIN -b $N_BOXES
#sudo python test_pilot_gen_batch.py -d $DOMAIN -b $N_BOXES
