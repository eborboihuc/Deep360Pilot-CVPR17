# 360 videos data pipeline

## How to Run?

*NOTE*: You can use this [example](https://drive.google.com/uc?export=download&id=0B9wE6h4m--wjaTNPYUk4NkM0UDA) file to go through data pipeline

1. Make sure `dataset.csv` is placed in the repository, which contains the metadata of our dataset. 
   And also place `Deep360Pilot-optical-flow` repository in the current location.

2. Make sure the generated feature `roisavg.npy`, `roislist.npy` and every video frames placed at `data/feature_{DOMAIN}_{N_BOXES}boxes` and `data/frame_{DOMAIN}`

3. Execute `run.sh {DOMAIN} {N_BOXES}` once and you can get the ideal data format under `data/feature_{DOMAIN}_{N_BOXES}boxes`

  `./run.sh test 16`



## Details inside run.sh
  
1. Parse `dataset.csv` to generate `metadata.npy`, run:

  `python parse_csv.py`

2. Split raw label according to each clip which has length 50, run:

  `python parse_raw_label.py`

3. Deep360Pilot-Optical-Flow will generate ideal data input format and also hof.npy files, see `run.sh`:

  ```bash
  python Deep360Pilot-optical-flow/compute_video_flow.py -d $DOMAIN
  python Deep360Pilot-optical-flow/divide_area_motion_salient_boxes.py -d $DOMAIN -b $N_BOXES
  python Deep360Pilot-optical-flow/pruned_box_features.py -d $DOMAIN -b $N_BOXES
  python Deep360Pilot-optical-flow/get_motion_features.py -d $DOMAIN -b $N_BOXES
  python Deep360Pilot-optical-flow/hist_of_opt_flow/get_hof.py -d $DOMAIN -b $N_BOXES
  ```

4. Generate batches, run:

  `python gen_batch.py`

5. Test generated batches, run:

  `python test_gen_batch,py`
