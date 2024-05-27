from SELFRec import SELFRec
from util.conf import ModelConf
import time
import torch.utils.data as data
from utils import *
import torch
import pickle
import numpy as np
from attacks.DICE_Attack import DICE_Attack
from attacks.Heuristic_Attack import Heuristic_Attack
import scipy.sparse as sp
import pandas as pd
import random
from model import stable
from ProGNN.train import pro_GNN_Hyper,proGNNTrain

flip_rate=20/100

model = 'LightGCN' #LightGCN SGL SimGCL XSimGCL
attack='HA' #no rnd DICE HA
dataset = 'yelp' #yelp gowalla amazon

if dataset=='yelp':
    u=11840
    v=9894 #332268  95161
    n_flips=int(332268*flip_rate)
if dataset=='gowalla':
    u=10164
    v=11488 #220256  24510
    n_flips = int(220256 * flip_rate)
if dataset=='amazon':
    u=7858
    v=7780 #125592 35633
    n_flips = int(125592 * flip_rate)

n_candidates = 10*n_flips

device = 'cuda:0'

s = time.time()

conf = ModelConf('./conf/' + model + '.conf')


# load data
path = 'data/' + dataset + '/'
f = open(path+'trnMat.pkl','rb')
train = pickle.load(f)

train_pd = pd.DataFrame.sparse.from_spmatrix(train)
train_pd = train_pd.iloc[0:u,0:v]
train = sp.coo_matrix(train_pd.values)


# adversarial attack
seed=0
dim=64
window_size = 5
train_torch=torch.from_numpy(train.toarray())
if attack=='dice' or attack=='DICE':
    adj = DICE_Attack(train_torch.cpu(),n_flips)
    train = sp.coo_matrix(adj.todense()[:u, u:])

if attack=='rnd':
    adj = Bu2Adj(train_torch.cpu())
    candidates = generate_candidates_addition(adj_matrix=adj, n_candidates=n_candidates, u=u, v=v, seed=seed)
    rnd_flips = random.sample(list(candidates.copy()), n_flips)
    flips = np.array(rnd_flips)
    adj = flip_candidates(adj, flips)
    train = sp.coo_matrix(adj.todense()[:u, u:])

if attack=='HA' or attack=='ha':
    adj = Heuristic_Attack(train_torch.cpu(), n_flips)
    train = sp.coo_matrix(adj.todense()[:u, u:])

f = open(path+'tstMat.pkl','rb')
test = pickle.load(f)
test_pd = pd.DataFrame.sparse.from_spmatrix(test)
test_pd = test_pd.iloc[0:u,0:v]
test = sp.coo_matrix(test_pd.values)

rec = SELFRec(conf, train, test)

rec.execute()
e = time.time()
print("Running time: %f s" % (e - s))

print(model)
print(attack)
print(dataset)
