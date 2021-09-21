# author = liuwei
# date = 2021-09-21

import os
import numpy as np
from sklearn.metrics import accuracy_score

def accuracy(gold_ids, pred_ids):
    """
    Args:
        gold_ids:
        pred_ids:
    """
    assert len(gold_ids) == len(pred_ids), (len(gold_ids), len(pred_ids))

    acc = accuracy_score(gold_ids, pred_ids)

    return acc
