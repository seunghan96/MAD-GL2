import torch
import torch.nn as nn
import torch.optim as optim
from model import *
import util


class trainer():
    def __init__(self, scaler, supports, 
                 input_length, output_length, num_nodes, input_dim, 
                 A_gt, CSM, gaf_mid_ch, nhid, blocks, layers, kernel_size, 
                 dropout, num_graphs, skip_alpha, end_alpha, noise, hard, lrate, wdecay, device):
        self.model = madgl2(supports = supports, input_length = input_length,
                            num_nodes = num_nodes, input_dim = input_dim, output_length = output_length, 
                            A_gt = A_gt, CSM = CSM,  gaf_mid_ch = gaf_mid_ch,
                            residual_ch = nhid, dilation_ch = nhid, skip_ch = nhid * skip_alpha, 
                            end_ch = nhid * end_alpha,blocks = blocks, layers = layers, 
                            kernel_size = kernel_size, dropout = dropout, num_graphs = num_graphs, 
                            noise = noise, hard = hard)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5

    def train(self, X, y, gaf, train_mode = True):
        self.model.train()
        self.optimizer.zero_grad()
        
        # (1) Feed Forward
        X = nn.functional.pad(X,(1,0,0,0))
        y_pred = self.model(X, gaf, train_mode)
        y_pred = y_pred.transpose(1,3)
        y_pred = self.scaler.inverse_transform(y_pred)
        
        # (2) Back Propagation
        y = torch.unsqueeze(y, dim = 1)
        loss = self.loss(y_pred, y, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        
        # (3) MAPE & RMSE
        mape = util.masked_mape(y_pred, y, 0.0).item()
        rmse = util.masked_rmse(y_pred, y, 0.0).item()
        return loss.item(), mape, rmse

    def eval(self, X, y, gaf, train_mode = False):
        self.model.eval()
        
        # (1) Feed Forward
        X = nn.functional.pad(X,(1,0,0,0))
        y_pred = self.model(X, gaf, train_mode)
        y_pred = y_pred.transpose(1,3)
        y_pred = self.scaler.inverse_transform(y_pred)
        
        # (2) Back Propagation
        y = torch.unsqueeze(y,dim=1)
        loss = self.loss(y_pred, y, 0.0)
        mape = util.masked_mape(y_pred, y, 0.0).item()
        rmse = util.masked_rmse(y_pred, y, 0.0).item()
        return loss.item(),mape,rmse
