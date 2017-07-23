# 360 videos data pipeline

## How to Run?
1. Make sure `dataset.csv` is placed in the repository, which contains the metadata of our dataset. And also place `OF_360` repository in the current location.

2. Parse `dataset.csv` to generate `metadata.npy`, run:

  `python parse_csv.py`

3. Split raw label according to each clip which has length 50, run:

  `python parse_raw_label.py`

4. Generate batches, run:

  `python gen_batch.py`

5. Test generated batches, run:

  `python test_gen_batch,py`
