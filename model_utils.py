import torch
import torch.nn as nn
import torch.nn.functional as F


class nconv(nn.Module):
    '''
    Affine Transformation (y = Ax)
    '''
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        if x.dim()==4:
            x = torch.einsum('ncvl,knvw->kncwl',(x,A))
        if x.dim()==5:
            x = torch.einsum('kncvl,knvw->kncwl',(x,A))

        return x.contiguous()

class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, num_graphs, num_nodes, noise,order=2):
        '''
        Graph Convolution Network
         - c_in : Input ch
         - c_out : Output ch
         - dropout : dropout ratio in FC layer
         - num_graphs : number of Adjacency matrices
         - order : order of gcn convolution
        '''
        super(gcn,self).__init__()
        self.nconv = nconv()
        self.c_in = (order*num_graphs+1)*c_in
        self.conv = nn.Conv2d(self.c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1))
        self.dropout = dropout
        self.order = order
        self.num_graphs = num_graphs
        self.num_nodes = num_nodes
        self.noise = noise
    
    def forward(self, X, A_list, train_mode = True):
        graph_concat_idx = [0] + list(range(1,self.num_graphs*2+1,2)) + list(range(2,self.num_graphs*2+1,2))
        out = [X]
        
        if self.noise & train_mode :
            A_list = A_list * (1 + torch.randn(A_list.size(), device = A_list.device)/(self.num_nodes**0.5))
        X1 = self.nconv(X, A_list)                 
        out.append(X1)
        for _ in range(self.order - 1):
            X2 = self.nconv(X1, A_list)
            out.append(X2)
            X1 = X2
        
        out = torch.vstack(out)
        out = out[graph_concat_idx,:,:,:]
        out = torch.cat(list(out), dim=1)
        out = self.conv(out)
        out = F.dropout(out, self.dropout, training = self.training)
        return out

def attention_score(global_graphs, input_sim_mat, num_graphs, bs):
    global_graphs = global_graphs.reshape(num_graphs,-1)
    input_sim_mat = input_sim_mat.reshape(bs,-1)
    global_graphs_sum = torch.sum(global_graphs**2,axis=-1)
    input_sim_mat_sum = torch.sum(input_sim_mat**2,axis=-1)
    DOWN = global_graphs_sum.unsqueeze(1)*input_sim_mat_sum**0.5
    UP = torch.sum((global_graphs.unsqueeze(1)*input_sim_mat),axis=-1)
    att_score = F.softmax(UP/DOWN,dim=0)
    return att_score



class gaf_layers(nn.Module):
    def __init__(self, mid_ch = 32, num_graphs = 6):
        super(gaf_layers,self).__init__()
        self.embed_layers = nn.Sequential(*[
            nn.Conv2d(1, mid_ch, kernel_size = 3, stride = 3),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(mid_ch, num_graphs*2, kernel_size = 3, stride = 3),
            nn.BatchNorm2d(num_graphs*2)
            ])

    def forward(self, x):
        x = self.embed_layers(x)
        return x    

