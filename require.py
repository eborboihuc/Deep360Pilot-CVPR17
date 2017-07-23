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


