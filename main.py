import numpy as np
import torch
import pickle
from model import LightGCL
from utils import metrics, scipy_sparse_mat_to_torch_sparse_tensor
#import pandas as pd
#from parser import args
from tqdm import tqdm
import time
import torch.utils.data as data
from utils import *
import time
import argparse
import pandas as pd
import random
import scipy.sparse as sp
from attacks.DICE_Attack import DICE_Attack
from attacks.Heuristic_Attack import Heuristic_Attack
from similarity.similarity import GCNSimDefense,ImpAddEdges

def parse_args():
    parser = argparse.ArgumentParser(description='Model Params')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--decay', default=0.99, type=float, help='learning rate')
    parser.add_argument('--batch', default=256, type=int, help='batch size')
    parser.add_argument('--inter_batch', default=4096, type=int, help='batch size')
    parser.add_argument('--note', default=None, type=str, help='note')
    parser.add_argument('--lambda1', default=0.2, type=float, help='weight of cl loss')
    parser.add_argument('--epoch', default=100, type=int, help='number of epochs')
    #默认epoch为100
    parser.add_argument('--d', default=64, type=int, help='embedding size')
    parser.add_argument('--q', default=5, type=int, help='rank')
    parser.add_argument('--gnn_layer', default=2, type=int, help='number of gnn layers')
    parser.add_argument('--data', default='yelp', type=str, help='name of dataset')
    parser.add_argument('--dropout', default=0.0, type=float, help='rate for edge dropout')
    parser.add_argument('--temp', default=0.2, type=float, help='temperature in cl loss')
    parser.add_argument('--lambda2', default=1e-7, type=float, help='l2 reg weight')
    parser.add_argument('--cuda', default='0', type=str, help='the gpu to use')
    return parser.parse_args()
args = parse_args()

simFunc = 'jac'
aug_rate=0.0005 #ar
flip_rate=20/100

args.data='yelp' #yelp gowalla amazon
attack='HA' #no rnd DICE HA
rndInit = False #False in Full Model
aug=True #True in Full Model
step1_rate=1.0 #sr, 0.1 in Full Model, 1.0 in Ablation Model

data_file = open("result_yelp.txt",'w+')


if args.data=='yelp':
    u=11840
    v=9894 #332268  95161
    aug_num=int(332268*aug_rate)
    n_flips=int(332268*flip_rate)
    thj=0.02
    thx=0.005
if args.data=='gowalla':
    u=10164
    v=11488 #220256  24510
    aug_num=int(220256*aug_rate)
    n_flips = int(220256 * flip_rate)
    thj=0.01
    thx=0.001
if args.data=='amazon':
    u=7858
    v=7780 #125592 35633
    aug_num=int(125592*aug_rate)
    n_flips = int(125592 * flip_rate)
    thj=0.02
    thx=0.002
n_candidates = 10*n_flips


start_time=time.time()
device = 'cuda:' + args.cuda

# hyperparameters
d = args.d
l = args.gnn_layer
temp = args.temp
batch_user = args.batch
epoch_no = args.epoch
max_samp = 40
lambda_1 = args.lambda1
lambda_2 = args.lambda2
dropout = args.dropout
lr = args.lr
decay = args.decay
svd_q = args.q

# load data
path = 'data/' + args.data + '/'
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

train_csr = (train!=0).astype(np.float32)

f = open(path+'tstMat.pkl','rb')
test = pickle.load(f)
test_pd = pd.DataFrame.sparse.from_spmatrix(test)
test_pd = test_pd.iloc[0:u,0:v]
test = sp.coo_matrix(test_pd.values)

print(train.data.sum())
print(test.data.sum())


epoch_user = min(train.shape[0], 30000)

# normalizing the adj matrix
rowD = np.array(train.sum(1)).squeeze()
#print(rowD.shape)#(29601,)
colD = np.array(train.sum(0)).squeeze()
#print(colD.shape)#(24734,)

for i in range(len(train.data)):
    train.data[i] = train.data[i] / pow(rowD[train.row[i]]*colD[train.col[i]], 0.5)

# construct data loader
train = train.tocoo()
train_data = TrnData(train)

train_loader = data.DataLoader(train_data, batch_size=args.inter_batch, shuffle=True, num_workers=0)

adj_norm = scipy_sparse_mat_to_torch_sparse_tensor(train)

adj_norm = adj_norm.coalesce().cuda(torch.device(device))

# perform svd reconstruction
adj = scipy_sparse_mat_to_torch_sparse_tensor(train).coalesce().cuda(torch.device(device))
#<class 'torch.Tensor'>

if aug:
    adj=adj.cpu().to(torch.float).to_dense()
    adj=sp.csr_matrix(adj)
    candidates = augmentation(adj_matrix=adj, n_candidates=10 * aug_num, u=u, v=v, seed=seed)
    rnd_flips = random.sample(list(candidates.copy()), aug_num)
    flips = np.array(rnd_flips)
    adj = flip_Bu_candidates(adj, flips)
    adj = sp.coo_matrix(adj.todense())
    adj = scipy_sparse_mat_to_torch_sparse_tensor(adj).coalesce().cuda(torch.device(device))


#print('Performing SVD...')
svd_u,s,svd_v = torch.svd_lowrank(adj, q=svd_q) #输入的adj应为torch
u_mul_s = svd_u @ (torch.diag(s))
v_mul_s = svd_v @ (torch.diag(s))
del s
#print('SVD done.')

# process test set
test_labels = [[] for i in range(test.shape[0])]
for i in range(len(test.data)):
    row = test.row[i]
    col = test.col[i]
    test_labels[row].append(col)
#print(len(test_labels)) #29601

loss_list = []
loss_r_list = []
loss_s_list = []
recall_20_x = []
recall_20_y = []
ndcg_20_y = []
recall_40_y = []
ndcg_40_y = []


#step1: pre-training
#print(adj_norm)
#print(train_csr)
model = LightGCL(adj_norm.shape[0], adj_norm.shape[1], d, u_mul_s, v_mul_s, svd_u.T, svd_v.T, train_csr, adj_norm, l, temp, lambda_1, lambda_2, dropout, batch_user, device)
#model.load_state_dict(torch.load('saved_model.pt'))
model.cuda(torch.device(device))
optimizer = torch.optim.Adam(model.parameters(),weight_decay=0,lr=lr)
#optimizer.load_state_dict(torch.load('saved_optim.pt'))

current_lr = lr
#print(model.E_u_0)

for epoch in range(int(epoch_no * step1_rate)):
    epoch_loss = 0
    epoch_loss_r = 0
    epoch_loss_s = 0
    train_loader.dataset.neg_sampling()

    for i, batch in enumerate(tqdm(train_loader)):
        uids, pos, neg = batch #(user, pos item, neg item)
        uids = uids.long().cuda(torch.device(device))
        pos = pos.long().cuda(torch.device(device))
        neg = neg.long().cuda(torch.device(device))
        iids = torch.concat([pos, neg], dim=0)

        # feed
        optimizer.zero_grad()
        loss, loss_r, loss_s= model(uids, iids, pos, neg)

        loss.backward()
        optimizer.step()
        #print('batch',batch)
        epoch_loss += loss.cpu().item()
        epoch_loss_r += loss_r.cpu().item()
        epoch_loss_s += loss_s.cpu().item()

        torch.cuda.empty_cache()
        #print(i, len(train_loader), end='\r')

    batch_no = len(train_loader)
    epoch_loss = epoch_loss/batch_no
    epoch_loss_r = epoch_loss_r/batch_no
    epoch_loss_s = epoch_loss_s/batch_no
    loss_list.append(epoch_loss)
    loss_r_list.append(epoch_loss_r)
    loss_s_list.append(epoch_loss_s)
    #print('Epoch:',epoch,'Loss:',epoch_loss,'Loss_r:',epoch_loss_r,'Loss_s:',epoch_loss_s)

#print(model.E_u_0)

#step2: Graph Filtering
if step1_rate!=1.0:
    model = model.to(torch.device('cpu'))
    ori_train_csr = sp.csr_matrix(train_torch.cpu())
    rep_u=model.G_u_list[model.l].detach().cpu().numpy()
    rep_v=model.G_i_list[model.l].detach().cpu().numpy()
    del model
    del optimizer
    #print(rep_u[:20],file=data_file)
    train,emb_u,emb_v = GCNSimDefense(rep_u, rep_v, u, v, ori_train_csr, thx=thx, thj=thj, simFunc='jac')
    #print(sum(sum(train)))
    train_Bu = train[:u,u:]
    train = sp.coo_matrix(train_Bu)
    
    rowD = np.array(train.sum(1)).squeeze()
    colD = np.array(train.sum(0)).squeeze()
    for i in range(len(train.data)):
        train.data[i] = train.data[i] / pow(rowD[train.row[i]]*colD[train.col[i]], 0.5)
    
    train_csr = (train != 0).astype(np.float32)

    train_data = TrnData(train)
    train_loader = data.DataLoader(train_data, batch_size=args.inter_batch, shuffle=True, num_workers=0)
    adj_norm = scipy_sparse_mat_to_torch_sparse_tensor(train)
    adj_norm = adj_norm.coalesce().cuda(torch.device(device))

    # perform svd reconstruction
    adj = scipy_sparse_mat_to_torch_sparse_tensor(train).coalesce().cuda(torch.device(device))
    svd_u, s, svd_v = torch.svd_lowrank(adj, q=svd_q)  # 输入的adj应为torch
    u_mul_s = svd_u @ (torch.diag(s))
    v_mul_s = svd_v @ (torch.diag(s))
    del s
    #print(adj_norm)
    #print(train_csr)
    if rndInit:
        model = LightGCL(adj_norm.shape[0], adj_norm.shape[1], d, u_mul_s, v_mul_s, svd_u.T, svd_v.T, train_csr,
                         adj_norm, l, temp, lambda_1, lambda_2, dropout, batch_user, device)
    else:
        model = LightGCL(adj_norm.shape[0], adj_norm.shape[1], d, u_mul_s, v_mul_s, svd_u.T, svd_v.T, train_csr, adj_norm,
                     l, temp, lambda_1, lambda_2, dropout, batch_user, device, torch.Tensor(rep_u), torch.Tensor(rep_v))
    # model.load_state_dict(torch.load('saved_model.pt'))
    model.cuda(torch.device(device))
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0, lr=lr)

#print(model.E_u_0)
#step3: training
for epoch in range(int(epoch_no * (1-step1_rate))):
    epoch_loss = 0
    epoch_loss_r = 0
    epoch_loss_s = 0
    train_loader.dataset.neg_sampling()

    #print(epoch)
    #print(model.E_u_0)

    for i, batch in enumerate(tqdm(train_loader)):
        uids, pos, neg = batch
        uids = uids.long().cuda(torch.device(device))
        pos = pos.long().cuda(torch.device(device))
        neg = neg.long().cuda(torch.device(device))
        iids = torch.concat([pos, neg], dim=0)

        # feed
        optimizer.zero_grad()
        loss, loss_r, loss_s = model(uids, iids, pos, neg)

        loss.backward()
        optimizer.step()
        # print('batch',batch)
        epoch_loss += loss.cpu().item()
        epoch_loss_r += loss_r.cpu().item()
        epoch_loss_s += loss_s.cpu().item()

        torch.cuda.empty_cache()
        # print(i, len(train_loader), end='\r')

    batch_no = len(train_loader)
    epoch_loss = epoch_loss / batch_no
    epoch_loss_r = epoch_loss_r / batch_no
    epoch_loss_s = epoch_loss_s / batch_no
    loss_list.append(epoch_loss)
    loss_r_list.append(epoch_loss_r)
    loss_s_list.append(epoch_loss_s)
#print(model.E_u_0)

# final test
test_uids = np.array([i for i in range(adj_norm.shape[0])])
batch_no = int(np.ceil(len(test_uids)/batch_user))

all_recall_20 = 0
all_ndcg_20 = 0
all_recall_40 = 0
all_ndcg_40 = 0
for batch in range(batch_no):
    start = batch*batch_user
    end = min((batch+1)*batch_user,len(test_uids))

    test_uids_input = torch.LongTensor(test_uids[start:end]).cuda(torch.device(device))
    predictions = model(test_uids_input,None,None,None,test=True)
    predictions = np.array(predictions.cpu())

    #top@20
    recall_20, ndcg_20 = metrics(test_uids[start:end],predictions,20,test_labels)
    #top@40
    recall_40, ndcg_40 = metrics(test_uids[start:end],predictions,40,test_labels)

    all_recall_20+=recall_20
    all_ndcg_20+=ndcg_20
    all_recall_40+=recall_40
    all_ndcg_40+=ndcg_40
    #print('batch',batch,'recall@20',recall_20,'ndcg@20',ndcg_20,'recall@40',recall_40,'ndcg@40',ndcg_40)

print('-------------------------------------------')
print('Final test:','Recall@20:',all_recall_20/batch_no,'Ndcg@20:',all_ndcg_20/batch_no,'Recall@40:',all_recall_40/batch_no,'Ndcg@40:',all_ndcg_40/batch_no)

recall_20_x.append('Final')
recall_20_y.append(all_recall_20/batch_no)
ndcg_20_y.append(all_ndcg_20/batch_no)
recall_40_y.append(all_recall_40/batch_no)
ndcg_40_y.append(all_ndcg_40/batch_no)

end_time=time.time()
print(end_time-start_time)

current_t = time.gmtime()

print(end_time-start_time,file=data_file)

print('Recall@20:'+str(all_recall_20/batch_no)+'  Ndcg@20:'+str(all_ndcg_20/batch_no)+'\n Recall@40:'+str(all_recall_40/batch_no)+' Ndcg@40:'+str(all_ndcg_40/batch_no),file=data_file)
print(args.data,file=data_file)
print("thx",file=data_file)
print(thx,file=data_file)
print("thj",file=data_file)
print(thj,file=data_file)
print("aug_rate",file=data_file)
print(aug_rate,file=data_file)
print("step1_rate",file=data_file)
print(step1_rate,file=data_file)
print("rndInit",file=data_file)
print(rndInit,file=data_file)
print("aug",file=data_file)
print(aug,file=data_file)
print("attack",file=data_file)
print(attack,file=data_file)

data_file.close()

print('Recall@20:'+str(all_recall_20/batch_no)+'  Ndcg@20:'+str(all_ndcg_20/batch_no)+'\n Recall@40:'+str(all_recall_40/batch_no)+' Ndcg@40:'+str(all_ndcg_40/batch_no))
print(args.data)
print("thx")
print(thx)
print("thj")
print(thj)
print("aug_rate")
print(aug_rate)
print("step1_rate")
print(step1_rate)
print("rndInit")
print(rndInit)
print("aug")
print(aug)
print("attack")
print(attack)