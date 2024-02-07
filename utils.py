import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import scipy.sparse as sp

def metrics(uids, predictions, topk, test_labels):
    user_num = 0
    all_recall = 0
    all_ndcg = 0
    for i in range(len(uids)):
        uid = uids[i]
        prediction = list(predictions[i][:topk])
        label = test_labels[uid]
        if len(label)>0:
            hit = 0
            idcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(topk, len(label)))])
            dcg = 0
            for item in label:
                if item in prediction:
                    hit+=1
                    loc = prediction.index(item)
                    dcg = dcg + np.reciprocal(np.log2(loc+2))
            all_recall = all_recall + hit/len(label)
            all_ndcg = all_ndcg + dcg/idcg
            user_num+=1
    return all_recall/user_num, all_ndcg/user_num

def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def sparse_dropout(mat, dropout):
    if dropout == 0.0:
        return mat
    indices = mat.indices()
    values = nn.functional.dropout(mat.values(), p=dropout)
    size = mat.size()
    return torch.sparse.FloatTensor(indices, values, size)

def spmm(sp, emb, device):
    sp = sp.coalesce()
    cols = sp.indices()[1]
    rows = sp.indices()[0]
    col_segs =  emb[cols] * torch.unsqueeze(sp.values(),dim=1)
    result = torch.zeros((sp.shape[0],emb.shape[1])).cuda(torch.device(device))
    result.index_add_(0, rows, col_segs)
    return result

def Bu2Adj(Bu):
    '''
    :param Bu: Bu矩阵,torch类型
    :return: adj 全邻接矩阵 csr
    '''
    Bu = Bu.to(torch.float)
    #Bu = Bu.to_dense()
    u = Bu.shape[0]
    v = Bu.shape[1]
    adj = np.zeros((u + v, u + v))
    adj[:u, u:] = Bu
    Bv = Bu.T
    adj[u:, :u] = Bv
    adj = sp.csr_matrix(adj)
    return adj

def Bu2Adjnn(Bu,th=5):
    '''
    :param Bu: Bu矩阵,torch类型
    :return:adjnn 隐式关系全邻接矩阵 csr
    '''
    Bu = Bu.to(torch.float)
    #Bu = Bu.to_dense()
    u = Bu.shape[0]
    v = Bu.shape[1]
    adjnn = np.zeros((u+v,u+v))
    adjnn[:u,u:] = Bu
    Bv = Bu.T
    adjnn[u:,:u] = Bv
    Bu = sp.csr_matrix(Bu)
    Bv = sp.csr_matrix(Bv)
    adj_uu = Bu.dot(Bv).toarray()
    adj_vv = Bv.dot(Bu).toarray()
    adj_uu -= th
    adj_uu[adj_uu >= 0] = 1
    adj_uu[adj_uu < 0] = 0
    adj_vv -= th
    adj_vv[adj_vv >= 0] = 1
    adj_vv[adj_vv < 0] = 0
    adjnn[:u, :u] = adj_uu
    adjnn[u:, u:] = adj_vv
    adjnn=sp.csr_matrix(adjnn)
    return adjnn

def augmentation(adj_matrix, n_candidates, u, v, seed=0):
    """Generates candidate edge flips for addition (non-edge -> edge).

    :param adj_matrix: sp.csr_matrix, shape [u, v]
        Adjacency matrix of the graph
    :param n_candidates: int
        Number of candidates to generate.
    :param seed: int
        Random seed
    :return: np.ndarray, shape [?, 2]
        Candidate set of edge flips
    """
    u_candidates = np.random.randint(0, u, [n_candidates*2, 1])
    v_candidates = np.random.randint(0, v, [n_candidates*2, 1])
    candidates = np.column_stack((u_candidates,v_candidates))
    candidates = candidates[adj_matrix[candidates[:, 0], candidates[:, 1]].A1 == 0]
    candidates = np.array(list(set(map(tuple, candidates))))
    candidates = candidates[:n_candidates]
    return candidates

class TrnData(data.Dataset):
    def __init__(self, coomat):
        self.rows = coomat.row #每一个交互对应的用户
        self.cols = coomat.col #每一个交互对应的项目
        self.dokmat = coomat.todok() #稀疏矩阵的一种表达方式
        self.negs = np.zeros(len(self.rows)).astype(np.int32)#每一个交互的用户对应的负项目


    def neg_sampling(self):
        for i in range(len(self.rows)):
            u = self.rows[i]
            while True:
                i_neg = np.random.randint(self.dokmat.shape[1])
                if (u, i_neg) not in self.dokmat:
                    break
            self.negs[i] = i_neg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx], self.negs[idx]