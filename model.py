import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from model_utils import *

class madgl2(nn.Module):
    def __init__(self, supports, 
                 input_length = 12, output_length = 12, num_nodes = 207, input_dim = 2, 
                 A_gt = False, CSM = 2,
                 gaf_mid_ch = 32,
                 residual_ch = 32, dilation_ch = 32, skip_ch = 256, end_ch = 512,
                 blocks = 4, layers = 2, kernel_size = 2, dropout = 0.3, num_graphs = 12, noise = False, hard=False):
        super(madgl2,self).__init__()
        
        #------------------------------------------------#
        # Data & Architecture related attributes
        #------------------------------------------------#
        self.input_length = input_length
        self.output_length = output_length
        self.num_nodes = num_nodes
        self.input_dim = input_dim

        #------------------------------------------------#
        # Graph related attributes
        #------------------------------------------------#
        self.A_gt = A_gt
        self.CSM = CSM
        self.noise = noise
        self.hard = hard
        self.num_graphs_learned = num_graphs
        self.num_graphs_total = self.num_graphs_learned + self.A_gt*len(supports) 
        print('Number of Adjacency Matrices : {}'.format(self.num_graphs_total))
        
        if self.A_gt:
            self.A_gt = supports
        #------------------------------------------------#
        # CSM (Connectivity Strengh Matrix)
        #------------------------------------------------#
        assert self.CSM in [0, 1, 2, 3]
        
        if self.CSM == 1 :
            self.CSV_in = nn.Parameter(torch.ones(num_nodes))
            
        if self.CSM == 2 :
            self.CSV_in = nn.Parameter(torch.ones(num_nodes))
            self.CSV_out = nn.Parameter(torch.ones(num_nodes))

        if self.CSM == 3 :
            self.CSV_out = nn.Parameter(torch.ones(num_nodes))
        #------------------------------------------------#
        # Prediction Module ( Dilated Causal Convolution )
        #------------------------------------------------#
        self.TCNa = nn.ModuleList()
        self.TCNb = nn.ModuleList()
        self.resid_1d = nn.ModuleList()
        self.skip_1d = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.bn = nn.ModuleList()

        self.blocks = blocks
        self.layers = layers
        
        #------------------------------------------------#
        # Graph Learning Module 
        #------------------------------------------------#
        self.gaf_conv = gaf_layers(mid_ch = gaf_mid_ch, num_graphs = self.num_graphs_learned)
        
        #------------------------------------------------#
        # Spatial-Temporal Embedding
        #------------------------------------------------#
        receptive_field = 1
        for _ in range(blocks):
            additional_scope = kernel_size - 1
            for i in range(layers):
                self.TCNa.append(nn.Conv2d(residual_ch, dilation_ch, kernel_size=(1,kernel_size), dilation = 2**i))
                self.TCNb.append(nn.Conv2d(residual_ch, dilation_ch, kernel_size=(1, kernel_size), dilation = 2**i))
                self.resid_1d.append(nn.Conv2d(dilation_ch, residual_ch, kernel_size=(1, 1)))
                self.skip_1d.append(nn.Conv2d(dilation_ch, skip_ch, kernel_size=(1, 1)))
                self.gconv.append(gcn(dilation_ch, residual_ch, dropout, num_graphs = self.num_graphs_total, 
                                      num_nodes = self.num_nodes, noise=self.noise, order=2))
                self.bn.append(nn.BatchNorm2d(residual_ch))
                receptive_field += additional_scope
                additional_scope *= 2
        self.receptive_field = receptive_field
        
        ## Start & End Convolution
        self.start_conv = nn.Conv2d(input_dim, residual_ch, kernel_size = (1,1))
        self.end_conv = nn.Sequential(*[
            nn.Conv2d(skip_ch, end_ch, kernel_size = (1,1)),
            nn.ReLU(),
            nn.Conv2d(end_ch, output_length, kernel_size = (1,1))
            ])

        
    def forward(self, x, gaf, train_mode = True):
        bs = x.size(0)
        #======================================================================#
        # [STEP 1] GRAPH LEARNING
        #======================================================================#
        A_list = []
        #----------------------------------------------------------------------#
        # step 1-1) (OPTIONAL) Add groud truth graphs
        if self.A_gt:
            A_list += [x.unsqueeze(0).repeat(bs,1,1) for x in self.A_gt]
            
        #----------------------------------------------------------------------#
        # step 1-2) Embed GAF images
        gaf_embed = self.gaf_conv(gaf.unsqueeze(1))

        #----------------------------------------------------------------------#
        # step 1-3) Generate K global graphs 
        gaf_embed = gaf_embed.reshape(self.num_nodes, 2*self.num_graphs_learned, -1)
        global_graphs = F.relu(torch.bmm(gaf_embed[:,0::2,:].permute(1,0,2),
                                         gaf_embed[:,1::2,:].permute(1,2,0)))
        global_graphs = F.softmax(global_graphs, dim=2)
        if self.CSM == 1:
            global_graphs = global_graphs * self.CSV_in.reshape(1,-1)            
        if self.CSM == 3:
            global_graphs = global_graphs * self.CSV_out.reshape(-1,1)  
        if self.CSM == 2:
            global_graphs = global_graphs * self.CSV_in.reshape(1,-1)            
            global_graphs = global_graphs * self.CSV_out.reshape(-1,1)  
            
        if self.hard:
            global_graphs = torch.clamp(global_graphs, 1e-5, 1 - (1e-5))
        #======================================================================#
        # [STEP 2] GRAPH SELECTION
        #======================================================================#
        # step 2-1) Input padding ( considering receptive field )
        input_len = x.size(3)
        if input_len < self.receptive_field:
            x = nn.functional.pad(x,(self.receptive_field-input_len,0,0,0)) # x shape : (64, 2, 207, 13)
        
        #----------------------------------------------------------------------#
        # step 2-2) Input TS similarity ( = local_TS_matrix )
        target_idx = 0
        x_target = x[:,target_idx,:,:]
        local_TS_matrix = torch.bmm(x_target, x_target.permute(0,2,1))
        local_TS_matrix = F.softmax(F.relu(local_TS_matrix), dim = 2)  
        
        #----------------------------------------------------------------------#
        # step 2-3) Calculate Attention weights ( = att_weight )
        ## --- Similarity between (1) K graphs & (2) A_raw
        att_weight = attention_score(global_graphs, local_TS_matrix, self.num_graphs_learned, bs)

        # step 2-4) Generate global graphs ( = global_weighted_graphs )
        ## --- shape : (batch size, K, num_nodes, num_nodes)
        global_weighted_graphs = torch.einsum('ab,acd->bacd',(att_weight, global_graphs))  
        if self.hard:
            global_weighted_graphs = torch.clamp(global_weighted_graphs, 1e-5, 1 - (1e-5))
            global_weighted_graphs = F.gumbel_softmax(torch.log(global_weighted_graphs/(1-global_weighted_graphs)), tau = 0.5, hard=True)

 
        A_list.extend(list(global_weighted_graphs.permute(1,0,2,3)))
        A_list = torch.stack(A_list)
        #======================================================================#
        # [STEP 3] Spatial & Temporal Embedding & Prediction
        #======================================================================#
        x = self.start_conv(x)

        skip = 0

        for i in range(self.blocks * self.layers):
            residual = x
            x_filter = torch.tanh(self.TCNa[i](residual))
            x_gate = torch.sigmoid(self.TCNb[i](residual))
            x = x_filter * x_gate
            s = x
            s = self.skip_1d[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip += s 
            x = self.gconv[i](x.unsqueeze(0), A_list, train_mode)
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        x = F.relu(skip)
        x = self.end_conv(x)
        return x

    def viz(self, gaf):
        gaf_embed = self.gaf_conv(gaf.unsqueeze(1))
        gaf_embed = gaf_embed.reshape(self.num_nodes, 2*self.num_graphs_learned, -1)
        global_graphs = F.relu(torch.bmm(gaf_embed[:,0::2,:].permute(1,0,2),
                                         gaf_embed[:,1::2,:].permute(1,2,0)))
        global_graphs = F.softmax(global_graphs, dim=2)
        if self.CSM == 1:
            global_graphs = global_graphs * self.CSV_in.reshape(1,-1)            
        if self.CSM == 3:
            global_graphs = global_graphs * self.CSV_out.reshape(-1,1)            
        if self.CSM == 2:
            global_graphs = global_graphs * self.CSV_in.reshape(1,-1)            
            global_graphs = global_graphs * self.CSV_out.reshape(-1,1)            
        
        if self.CSM == 0:
            csv_in = np.zeros(self.num_nodes)
            csv_out = np.zeros(self.num_nodes)
            graph_viz = global_graphs.detach().cpu().numpy()
        elif self.CSM == 1:
            csv_in = self.CSV_in.detach().cpu().numpy()
            csv_out = np.zeros(self.num_nodes)
            graph_viz = global_graphs.detach().cpu().numpy()
        elif self.CSM == 3:
            csv_out = self.CSV_out.detach().cpu().numpy()
            csv_in = np.zeros(self.num_nodes)
            graph_viz = global_graphs.detach().cpu().numpy()
        else:
            csv_in = self.CSV_in.detach().cpu().numpy()
            csv_out = self.CSV_out.detach().cpu().numpy()
            graph_viz = global_graphs.detach().cpu().numpy()
        return csv_in, csv_out, graph_viz


