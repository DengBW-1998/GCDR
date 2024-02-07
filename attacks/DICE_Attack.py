import torch
import argparse
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.linalg import eigh
import deeprobust.graph.defense as dr
from sklearn.model_selection import train_test_split
from attacks.dice import DICE
import scipy.sparse as sp
from utils import Bu2Adj
import warnings
warnings.filterwarnings("ignore")

def get_train_val_test(idx, train_size, val_size, test_size, stratify):
    idx_train_and_val, idx_test = train_test_split(idx,
                                                   random_state=None,
                                                   train_size=train_size + val_size,
                                                   test_size=test_size,
                                                   stratify=stratify)

    if stratify is not None:
        stratify = stratify[idx_train_and_val]

    idx_train, idx_val = train_test_split(idx_train_and_val,
                                          random_state=None,
                                          train_size=(train_size / (train_size + val_size)),
                                          test_size=(val_size / (train_size + val_size)),
                                          stratify=stratify)

    return idx_train, idx_val, idx_test

def DICE_Attack(Bu, n_flips):
    u=Bu.shape[0]
    v=Bu.shape[1]
    labels = np.zeros((u + v))
    labels[u:] = 1

    val_size = 0.1
    test_size = 0.8
    train_size = 1 - test_size - val_size

    idx = np.arange(u+v)
    idx_train, idx_val, idx_test = get_train_val_test(idx, train_size, val_size, test_size, stratify=labels)
    #idx_unlabeled = np.union1d(idx_val, idx_test)

    features = np.ones((u+v, 32))
    #features = sp.csr_matrix(features)

    device = 'cpu'
    surrogate = dr.GCN(nfeat=32, nclass=2,
                       nhid=16, dropout=0, with_relu=False, with_bias=False, device=device).to(device)

    adj=Bu2Adj(Bu)

    surrogate.fit(torch.Tensor(features), adj, torch.Tensor(labels).to(torch.int64), idx_train, idx_val, patience=30)

    model = DICE(surrogate, nnodes=u+v,
                 attack_structure=True, attack_features=False, device=device).to(device)

    model.attack(adj, labels, n_perturbations=n_flips)

    adj_matrix_flipped = sp.csr_matrix(model.modified_adj)
    return adj_matrix_flipped