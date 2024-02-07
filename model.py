import torch
import torch.nn as nn
from utils import sparse_dropout, spmm
import torch.nn.functional as F

class LightGCL(nn.Module):
    def __init__(self, n_u, n_i, d, u_mul_s, v_mul_s, ut, vt, train_csr, adj_norm, l, temp, lambda_1, lambda_2, dropout, batch_user, device, Eu=None, Ei=None):
        '''
        :param n_u: 用户总数
        :param n_i: 项目总数
        :param d: 嵌入维数
        :param u_mul_s: svd分解后u*s
        :param v_mul_s: svd分解后v*s
        :param ut: u矩阵转置
        :param vt: v矩阵转置
        :param train_csr: 评分矩阵的csr形式，维数为u*v
        :param adj_norm: 评分矩阵归一化后torch形式
        :param l: gnn层数
        :param temp: Ls即互信息损失中的超参数
        :param lambda_1: Ls损失的权重
        :param lambda_2: L2正则化权重
        :param dropout:
        :param batch_user: 一个小批量的大小，等价于batch_size
        :param device:
        :param Eu:Eu的初始值
        :param Ei:Ei的初始值
        '''
        super(LightGCL,self).__init__()
        #用户初始嵌入，维度为(u*d)，可学习参数
        if (Eu is None) or (Ei is None):
            self.E_u_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_u,d)))
            self.E_i_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_i,d)))
        else:
            self.E_u_0 = nn.Parameter(Eu)
            self.E_i_0 = nn.Parameter(Ei)

        self.train_csr = train_csr
        self.adj_norm = adj_norm
        self.l = l
        # 每一个gnn层输出的用户嵌入，维数为(l+1,u,d);第i层的E为0-i层的Z求和
        # 前一层的E卷积后得到后一层的Z，前面所有的Z加起来得到本层的E
        self.E_u_list = [None] * (l+1)
        self.E_i_list = [None] * (l+1)
        self.E_u_list[0] = self.E_u_0
        self.E_i_list[0] = self.E_i_0
        self.Z_u_list = [None] * (l+1) #每一个gnn层输出的用户嵌入，维数为(l+1,u,d)
        self.Z_i_list = [None] * (l+1)
        # 每一个gnn层输出的用户嵌入，维数为(l+1,u,d)，卷积规则为近似SVD
        self.G_u_list = [None] * (l+1)
        self.G_i_list = [None] * (l+1)
        self.G_u_list[0] = self.E_u_0
        self.G_i_list[0] = self.E_i_0
        self.temp = temp
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.dropout = dropout
        self.act = nn.LeakyReLU(0.5)
        self.batch_user = batch_user

        self.E_u = None
        self.E_i = None

        self.u_mul_s = u_mul_s
        self.v_mul_s = v_mul_s
        self.ut = ut
        self.vt = vt

        self.device = device

    def forward(self, uids, iids, pos, neg, test=False):
        '''
        :param uids: 所有参与训练/测试的用户的下标
        :param iids: pos和neg的拼接
        :param pos:  所有参与训练/测试的用户交互的正项目的下标
        :param neg:  所有参与训练/测试的负项目的下标
        :param test: 训练还是测试
        :return: 训练状态：三个损失函数值，均为常量；测试状态：返回本批次用户对所有测试项目的喜好顺序，维数为(用户批次大小，项目数)
        '''
        #print(uids.shape) #(4096)
        #print(iids.shape) #(8192)
        if test==True:  # testing phase
            preds = self.E_u[uids] @ self.E_i.T #该批次用户对所有项目的评分
            #print(preds.shape) #torch.Size([256, 24734])
            mask = self.train_csr[uids.cpu().numpy()].toarray() #排除参与训练的U-I对
            #print(mask.shape) #torch.Size([256, 24734])
            mask = torch.Tensor(mask).cuda(torch.device(self.device))
            preds = preds * (1-mask) - 1e8 * mask #该批次用户对测试项目的评分（训练项目记大负分）
            predictions = preds.argsort(descending=True) #该批次用户对测试项目的喜好顺序
            #print(predictions.shape) #torch.Size([256, 24734])
            return predictions
        else:  # training phase
            for layer in range(1,self.l+1):
                # GNN propagation
                # 公式1，由上一层的E卷积生成本层的Z
                self.Z_u_list[layer] = (torch.spmm(sparse_dropout(self.adj_norm,self.dropout), self.E_i_list[layer-1]))
                self.Z_i_list[layer] = (torch.spmm(sparse_dropout(self.adj_norm,self.dropout).transpose(0,1), self.E_u_list[layer-1]))

                # svd_adj propagation
                # 公式5，由上一层的E近似SVD卷积生成本层的G
                vt_ei = self.vt @ self.E_i_list[layer-1]
                self.G_u_list[layer] = (self.u_mul_s @ vt_ei)
                ut_eu = self.ut @ self.E_u_list[layer-1]
                self.G_i_list[layer] = (self.v_mul_s @ ut_eu)

                # aggregate
                # 公式2，由0到本层的Z求和生成本层的E
                self.E_u_list[layer] = self.Z_u_list[layer]
                self.E_i_list[layer] = self.Z_i_list[layer]

            self.G_u = sum(self.G_u_list)
            self.G_i = sum(self.G_i_list)
            '''
            print("len(self.G_u_list)")
            print(len(self.G_u_list)) #3
            print("len(self.G_u)")
            print(len(self.G_u)) #29601
            G_u_list 维数为(l+1,u,d)
            G_u 维数为(u,d)，即将G_u_list每个用户的0-l层表示求和
            '''
            # aggregate across layers
            self.E_u = sum(self.E_u_list)
            self.E_i = sum(self.E_i_list)

            # Ls即互信息损失，公式6
            G_u_norm = self.G_u
            E_u_norm = self.E_u
            G_i_norm = self.G_i
            E_i_norm = self.E_i
            neg_score = torch.log(torch.exp(G_u_norm[uids] @ E_u_norm.T / self.temp).sum(1) + 1e-8).mean()
            neg_score += torch.log(torch.exp(G_i_norm[iids] @ E_i_norm.T / self.temp).sum(1) + 1e-8).mean()
            pos_score = (torch.clamp((G_u_norm[uids] * E_u_norm[uids]).sum(1) / self.temp,-5.0,5.0)).mean() + (torch.clamp((G_i_norm[iids] * E_i_norm[iids]).sum(1) / self.temp,-5.0,5.0)).mean()
            loss_s = -pos_score + neg_score

            # bpr loss Lr
            u_emb = self.E_u[uids]
            pos_emb = self.E_i[pos]
            neg_emb = self.E_i[neg]
            pos_scores = (u_emb * pos_emb).sum(-1)
            neg_scores = (u_emb * neg_emb).sum(-1)
            loss_r = -(pos_scores - neg_scores).sigmoid().log().mean()

            # reg loss
            loss_reg = 0
            for param in self.parameters():
                loss_reg += param.norm(2).square()
            loss_reg *= self.lambda_2

            # total loss
            loss = loss_r + self.lambda_1 * loss_s + loss_reg
            #print('loss',loss.item(),'loss_r',loss_r.item(),'loss_s',loss_s.item())
            return loss, loss_r, self.lambda_1 * loss_s
