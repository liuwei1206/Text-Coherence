# author = liuwei
# date = 2021-09-16

import os
import json
import random
import time
import torch
import numpy as np
from torch.utils.data import Dataset
random.seed(123456)

class GCDCDataset(Dataset):
    """
    The dataset instance for GCDC corpus.
    Note, this dataset should support for sentence-document hierarchy, paragraph-document
    hierarchy and document-level modeling.
    """
    def __init__(self, file, params):
        """
        Args:
            file:
            params:
        """
