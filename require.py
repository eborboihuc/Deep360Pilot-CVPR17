#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
root = os.getcwd()

if os.path.isdir(os.path.join(root, 'data')) == False:
    os.mkdir(os.path.join(root, 'data'))

if os.path.isdir(os.path.join(root, 'checkpoint')) == False:
    os.mkdir(os.path.join(root, 'checkpoint'))

if os.path.isdir(os.path.join(root, 'misc/data')) == False:
    os.symlink(os.path.join(root, 'data'), os.path.join(root, 'misc/data'))

cmd='wget --no-check-certificate "https://drive.google.com/uc?export=download&id={ID}" -O {LOCAL_FILE_NAME}'
show='https://drive.google.com/uc?export=download&id={ID} and place it at {LOCAL_FILE_NAME}'

print "Please Download these files:"

print show.format(**{'ID': '0B9wE6h4m--wjNWdFbnVYbG9kNm8', 'LOCAL_FILE_NAME': 'checkpoint/Deep360Pilot-model.zip'})
#os.system(cmd.format(**{'ID': '0B9wE6h4m--wjNWdFbnVYbG9kNm8', 'LOCAL_FILE_NAME': 'checkpoint/Deep360Pilot-model.zip'}))
print show.format(**{'ID': '0B9wE6h4m--wjaTNPYUk4NkM0UDA', 'LOCAL_FILE_NAME': 'data/Deep360Pilot-test.zip'})
#os.system(cmd.format(**{'ID': '0B9wE6h4m--wjaTNPYUk4NkM0UDA', 'LOCAL_FILE_NAME': 'data/Deep360Pilot-test.zip'}))
print show.format(**{'ID': '0B9wE6h4m--wjWnF3LV9WUXdZMzA', 'LOCAL_FILE_NAME': 'data/Deep360Pilot-feature.zip'})
#os.system(cmd.format(**{'ID': '0B9wE6h4m--wjWnF3LV9WUXdZMzA', 'LOCAL_FILE_NAME': 'data/Deep360Pilot-feature.zip'}))
print show.format(**{'ID': '0B9wE6h4m--wjZzJkZnNLZW1BNE0', 'LOCAL_FILE_NAME': 'data/Deep360Pilot-batch-feature.zip'})
#os.system(cmd.format(**{'ID': '0B9wE6h4m--wjZzJkZnNLZW1BNE0', 'LOCAL_FILE_NAME': 'data/Deep360Pilot-batch-feature.zip'}))
print show.format(**{'ID': '0B9wE6h4m--wjSXVYblBWNktacFk', 'LOCAL_FILE_NAME': 'data/Deep360Pilot-label.zip'})
#os.system(cmd.format(**{'ID': '0B9wE6h4m--wjSXVYblBWNktacFk', 'LOCAL_FILE_NAME': 'data/Deep360Pilot-label.zip'}))
