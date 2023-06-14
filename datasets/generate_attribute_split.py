import torch
import torchvision
import torchvision.models as models
import json
import pandas as pd
import numpy as np
import torch.nn as nn
import scipy.io as io

from torch.utils.data import Dataset
from torch.utils import data
import os
from PIL import Image
from tqdm import tqdm
import h5py
import re
import random

datasets_unknown = {'CUB':[29, 50, 62, 69, 72, 87, 95, 100, 116, 120, 122, 125, 129, 139, 141, 159, 160, 167, 174, 176, 185, 189, 191, 192, 193],
                   'AWA2':[9, 30, 34, 41, 50],
                   'FLO':[2, 7, 9, 10, 12, 13, 14, 15, 18, 19],
                   'SUN':[4, 24, 58, 86, 96, 104, 113, 125, 131, 139, 153, 159, 197, 246, 247, 260, 299, 354, 380, 382, 421, 424, 426, 441, 472, 509, 518, 530, 581, 636, 680, 682, 696, 711, 713, 716]}

for dataset in datasets_unknown:
    ori_mat = io.loadmat('./datasets/{}/res101.mat'.format(dataset))
    ori_attr = io.loadmat('./datasets/{}/att_splits.mat'.format(dataset))

    ori_unseen_loc = ori_attr['test_unseen_loc']
    labels = ori_mat['labels']
    ori_unseen_classes = set(np.unique([labels[i-1] for i in ori_attr['test_unseen_loc']]))
    unknown_classes = set(datasets_unknown[dataset])
    unseen_classes = ori_unseen_classes - unknown_classes
    test_unseen_loc = []
    test_unknown_loc = []
    for i in range(len(labels)):
        if labels[i][0] in unseen_classes:
            test_unseen_loc.append(i+1)
        else:
            if labels[i][0] in unknown_classes:
                test_unknown_loc.append(i+1)
    test_unseen_loc = np.array(test_unseen_loc).reshape(-1, 1)
    test_unknown_loc = np.array(test_unknown_loc).reshape(-1, 1)
    ori_attr['test_unseen_loc'] = test_unseen_loc
    ori_attr['test_unknown_loc'] = test_unknown_loc
    io.savemat('./datasets/{}/att_splits.mat'.format(dataset), ori_attr)
    io.savemat('./datasets/{}/res101.mat'.format(dataset), ori_mat)