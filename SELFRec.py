import scipy.sparse as sp
import numpy as np


class SELFRec(object):
    def __init__(self, config, train_coo, test_coo):
        '''
        :param config: 配置文件信息，可忽略
        :param train_coo: 训练交互矩阵coo_matrix
        :param test_coo: 测试交互矩阵coo_matrix
        '''
        self.social_data = []
        self.feature_data = []
        self.config = config
        self.training_data = coo2list(train_coo)
        self.test_data = coo2list(test_coo)

        self.kwargs = {}
        if config.contain('social.data'):
            social_data = FileIO.load_social_data(self.config['social.data'])
            self.kwargs['social.data'] = social_data

    def execute(self):
        # import the model module
        #'''
        import_str = 'from baselineModels.'+ self.config['model.type'] +'.' + self.config['model.name'] + ' import ' + self.config['model.name']
        exec(import_str)
        recommender = self.config['model.name'] + '(self.config,self.training_data,self.test_data,**self.kwargs)'
        eval(recommender).execute()
        '''
        from model.graph.XSimGCL import XSimGCL
        XSimGCL(self.config,self.training_data,self.test_data,**self.kwargs)
        '''

def coo2list(coo):
    res = []
    mat = coo.toarray()
    u = mat.shape[0]
    v = mat.shape[1]
    for i in range(u):
        for j in range(v):
            if mat[i, j] != 0:
                res.append([str(i), str(j), mat[i, j]])
    return res
