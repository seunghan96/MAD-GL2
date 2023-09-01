import torch
import torch.nn as nn
import numpy as np
import argparse
import random
import time
import util
import yaml
from engine import trainer
import os
import sys
import pandas as pd
import shutil
from model import madgl2

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))

parser = argparse.ArgumentParser()
parser.add_argument('--config_filename', default='data/config.yaml', type=str,
                        help='Configuration filename for restoring the model.')
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--data',type=str,default='METR-LA',help='data path')
parser.add_argument('--adjdata',type=str,default='data/sensor_graph/adj_mx.pkl',help='adj data path')
parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
parser.add_argument('--img_type',type=str,default='GAF')
parser.add_argument('--A_gt', type=str2bool, nargs='?', const=True, default=False)
parser.add_argument('--CSM',type=int,default=2)
parser.add_argument('--blocks',type=int,default=4)
parser.add_argument('--layers',type=int,default=2)
parser.add_argument('--input_length',type=int,default=12,help='')
parser.add_argument('--output_length',type=int,default=12,help='')
parser.add_argument('--nhid',type=int,default=32,help='')
parser.add_argument('--skip_alpha',type=int,default=8,help='')
parser.add_argument('--end_alpha',type=int,default=16,help='')
parser.add_argument('--gaf_mid_ch',type=int,default=32,help='')
parser.add_argument('--input_dim',type=int,default=2,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=207,help='number of nodes')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--kernel_size',type=int,default=2,help='')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=120,help='')
parser.add_argument('--print_every',type=int,default=50,help='')
parser.add_argument('--seed',type=int,default=99,help='random seed')
parser.add_argument('--num_graphs', type=int,default=8,help='number of graphs')
parser.add_argument('--noise', type=str2bool, nargs='?', const=True, default=True)
parser.add_argument('--hard', type=str2bool, nargs='?', const=True, default=False)
parser.add_argument('--expid', type=int,default=1,help='experiment id')

args = parser.parse_args()

def main():
    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    #=======================================================#
    # (1) Load Experimental Settings    
    #=======================================================#
    device = torch.device(args.device)
    
    with open(args.config_filename) as f:
        config_file = yaml.load(f, Loader = yaml.Loader)


    data = args.data
    configs = config_file.get(data)
    if args.img_type == 'GAF':
        model_type = "MADGL2_{}_Agt_{}_CSM_{}_gafch_{}_numgraphs_{}_noise_{}_{}"
    elif args.img_type == 'GAF_norm':
        model_type = "GAF_mm_MADGL2_{}_Agt_{}_CSM_{}_gafch_{}_numgraphs_{}_noise_{}_{}"
    elif args.img_type == 'MTF':
        model_type = "MTF_MADGL2_{}_Agt_{}_CSM_{}_gafch_{}_numgraphs_{}_noise_{}_{}"
    elif args.img_type == 'MTF_norm':
        model_type = "MTF_mm_MADGL2_{}_Agt_{}_CSM_{}_gafch_{}_numgraphs_{}_noise_{}_{}"
        
    model_type = model_type.format(data,  'O' if args.A_gt else 'X', args.CSM,
                                    configs['gaf_mid_ch'], configs['num_graphs'], 
                                    'O' if args.noise else 'X',
                                  'hard' if args.hard else 'soft')
    
    print(f'========== [ MODEL TYPE = {model_type} ] ==========')
    PATH_PARAMS = os.path.join('./params', data, model_type, f'seed_{SEED}')
    if not os.path.exists(PATH_PARAMS):
        print('Making MODEL PARAMETER directory...')
        os.makedirs(PATH_PARAMS)
    else:
        params_list = os.listdir(PATH_PARAMS)
        is_best_params = [x for x in params_list if 'best' in x]
        if len(is_best_params) == 1 :
            print(f'Best parameter already exists')
            sys.exit(0)
        else:
            print('Training is not finished')
    
    PATH_LOSS = os.path.join('./loss', data, model_type, f'seed_{SEED}')
    if not os.path.exists(PATH_LOSS):
        print('Making LOSS directory...')
        os.makedirs(PATH_LOSS)

    
    #=======================================================#
    # (2) Load Datasets
    #=======================================================#
    supports = []
    if args.A_gt:
        _, _, adj_mx = util.load_adj(configs['adjdata'], args.adjtype)
        supports = [torch.tensor(i).to(device) for i in adj_mx]
        del adj_mx
    dataloader = util.load_dataset(f'data/{data}', args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']

    #=======================================================#
    # (3) Load GAF images
    #=======================================================#
    gaf = np.loadtxt(f'./data/{data}/{args.img_type.split("_")[0]}_{data}.txt')
    if 'norm' in args.img_type:
        gaf = (gaf-gaf.min())/(gaf.max()-gaf.min())
    H = int(gaf.shape[1]**0.5)
    gaf = gaf.reshape(gaf.shape[0], H, H)
    print(f'size of {args.img_type} image :', gaf.shape)
    ################################################
    #device = 'cpu'
    ################################################
    gaf = torch.Tensor(gaf).to(device)

    #=======================================================#
    # (4) Import Trainer
    #=======================================================#
    print('current device :',device)
    engine = trainer(scaler, supports, configs['input_length'], configs['output_length'], 
                     configs['num_nodes'], configs['input_dim'], 
                     args.A_gt, args.CSM, 
                     configs['gaf_mid_ch'], configs['nhid'], configs['blocks'], configs['layers'], 
                     args.kernel_size, args.dropout, configs['num_graphs'], 
                     args.skip_alpha, args.end_alpha, args.noise, args.hard,
                     configs['learning_rate'], args.weight_decay, device)

    #=======================================================#
    # (5) Start Training & Validation
    #=======================================================#
    print("start training...", flush = True)
    tr_time_lst, tr_loss_lst, tr_mape_lst, tr_rmse_lst = [], [], [], []
    val_time_lst, val_loss_lst, val_mape_lst, tr_rmse_lst = [], [], [], []
    
    best_epoch = 0
    best_val_loss = np.infty
    
    for epoch in range(1, configs['epochs']+1):
        train_loss, train_mape, train_rmse = [], [], []
        valid_loss, valid_mape, valid_rmse = [], [], []
        
        #-----------------------------------------------------------------------------#
        # [ 5-1. Train ]
        dataloader['train_loader'].shuffle()
        time_start = time.time()
        for iter, (X, y) in enumerate(dataloader['train_loader'].get_iterator()):
            X = torch.Tensor(X).to(device).transpose(1, 3)
            y = torch.Tensor(y).to(device).transpose(1, 3)[:,0,:,:]
            metrics = engine.train(X, y, gaf)
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            if iter % args.print_every == 0 :
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]), flush = True)
        time_end = time.time()
        tr_time_lst.append(time_end - time_start)

        #-----------------------------------------------------------------------------#
        # [ 5-2. Validation ]
        time_start = time.time()
        for iter, (X, y) in enumerate(dataloader['val_loader'].get_iterator()):
            X = torch.Tensor(X).to(device).transpose(1, 3)
            y = torch.Tensor(y).to(device).transpose(1, 3)[:,0,:,:]
            metrics = engine.eval(X, y, gaf)
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        time_end = time.time()
        val_time_lst.append(time_end - time_start)
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(epoch, (time_end - time_start)))
        #-----------------------------------------------------------------------------#
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)
        tr_loss_lst.append(mtrain_loss)
        tr_mape_lst.append(mtrain_mape)
        tr_rmse_lst.append(mtrain_rmse)
        
        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        val_loss_lst.append(mvalid_loss)
        val_mape_lst.append(mvalid_mape)
        tr_rmse_lst.append(mvalid_rmse)
        
        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(epoch, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, 
                         mvalid_mape, mvalid_rmse, tr_time_lst[-1]), flush=True)
        weight_name = os.path.join(PATH_PARAMS , f"epoch_{str(epoch)}_loss_{str(round(mvalid_loss,4))}.pth")
        torch.save(engine.model.state_dict(), weight_name)
        torch.cuda.empty_cache()
        
        if mvalid_loss < best_val_loss:
            best_epoch = epoch
            best_val_loss = mvalid_loss
    
    # save best weight
    best_weight_name = os.path.join(PATH_PARAMS , f"epoch_{str(best_epoch)}_loss_{str(round(best_val_loss,4))}.pth")
    best_weight_name2 = os.path.join(PATH_PARAMS , f"best_epoch_{str(best_epoch)}_loss_{str(round(best_val_loss,4))}.pth")
    shutil.copy(best_weight_name, best_weight_name2)
    
    # calculate Training/Evaluation time
    TIME_train_per_epoch = np.mean(tr_time_lst)
    TIME_val_per_epoch = np.mean(val_time_lst)
    time_list = [TIME_train_per_epoch, TIME_val_per_epoch]
    print("Average Training Time: {:.4f} secs/epoch".format(TIME_train_per_epoch))
    print("Average Evaluation Time: {:.4f} secs".format(TIME_val_per_epoch))
    
    np.savetxt(os.path.join(PATH_LOSS, 'train_loss.txt'), tr_loss_lst, delimiter=',')
    np.savetxt(os.path.join(PATH_LOSS, 'train_mape.txt'), tr_mape_lst, delimiter=',')
    np.savetxt(os.path.join(PATH_LOSS, 'train_rmse.txt'), tr_rmse_lst, delimiter=',')
    np.savetxt(os.path.join(PATH_LOSS, 'valid_loss.txt'), val_loss_lst, delimiter=',')
    np.savetxt(os.path.join(PATH_LOSS, 'valid_mape.txt'), val_mape_lst, delimiter=',')
    np.savetxt(os.path.join(PATH_LOSS, 'valid_rmse.txt'), tr_rmse_lst, delimiter=',')
    np.savetxt(os.path.join(PATH_LOSS, 'time.txt'), time_list, delimiter=',')
    
    #=======================================================#
    # (6) Start Testing
    #=======================================================#
    # [6-1] Set paths
    SAVE_PATH = os.path.join('./results', data, model_type)
    PRED_PATH = os.path.join('./predictions', data, model_type)
    os.makedirs(SAVE_PATH, exist_ok = True)
    os.makedirs(PRED_PATH, exist_ok = True)
    PATH_METRIC = os.path.join(SAVE_PATH, f'metric_{data}_seed_{SEED}.csv')
    PATH_Y_REAL = os.path.join(PRED_PATH, f'y_test_real_{data}_seed_{SEED}.csv')
    PATH_Y_PRED = os.path.join(PRED_PATH, f'y_test_pred_{data}_seed_{SEED}.csv')
    PATH_CSV_IN = os.path.join(SAVE_PATH, f'CSV_in_{data}_seed_{SEED}.npy')
    PATH_CSV_OUT = os.path.join(SAVE_PATH, f'CSV_out_{data}_seed_{SEED}.npy')
    PATH_GRAPH = os.path.join(SAVE_PATH, f'global_graph_{data}_seed_{SEED}.npy')
    
    # [6-2] Load best models
    model = madgl2(supports = supports, input_length = configs['input_length'], output_length = configs['output_length'], 
                   num_nodes = configs['num_nodes'], input_dim = configs['input_dim'], A_gt = args.A_gt, CSM = args.CSM,
                   gaf_mid_ch = configs['gaf_mid_ch'], residual_ch=configs['nhid'], dilation_ch=configs['nhid'], 
                   skip_ch=configs['nhid'] * 8, end_ch=configs['nhid'] * 16, blocks = configs['blocks'], 
                   layers = configs['layers'], kernel_size = args.kernel_size, dropout = args.dropout, 
                   num_graphs = configs['num_graphs'], noise = args.noise, hard = args.hard)
    #model.load_state_dict(torch.load(best_weight_name))
    model.load_state_dict(torch.load(best_weight_name2))
    model.to(device)
    model.eval()
    
    # [6-3] Prediction
    ## -- single step forecast
    ## -- multi step forecast
    
    y_true = torch.Tensor(dataloader['y_test']).to(device)
    y_true = y_true.transpose(1,3)[:,0,:,:]
    
    y_test_pred = []
    y_test_real = []
    if configs['output_length'] == 1:
        print('Single Step forecasting')
        loss_fn_l1 = nn.L1Loss(reduction='sum').to(device)
        loss_fn_l2 = nn.MSELoss(reduction='sum').to(device)
        
        n_samples = 0
        total_loss_l1 = 0
        total_loss_l2 = 0
        
        for iter, (X, y) in enumerate(dataloader['test_loader'].get_iterator()):
            X = torch.Tensor(X).to(device).transpose(1,3)
            y = torch.Tensor(y).to(device).squeeze(-1).squeeze(1)
            
            with torch.no_grad():
                y_pred = model(X, gaf).transpose(1,3)
                if iter == 0:
                    CSV_in, CSV_out, global_graph = model.viz(gaf)
            
            y_pred = y_pred.squeeze(-1).squeeze(1)
            y_pred = scaler.inverse_transform(y_pred)
            total_loss_l1 += loss_fn_l1(y_pred, y)
            total_loss_l2 += loss_fn_l2(y_pred, y)
            n_samples += (y_pred.size(0) * configs['num_nodes'])
            y_test_pred.append(y_pred.squeeze().detach().cpu().numpy())
            y_test_real.append(y.detach().cpu().numpy())
        

        y_test_pred = np.concatenate(np.array(y_test_pred),axis=0)
        #y_test_pred = torch.cat(y_test_pred, dim=0) # (6912,207)
        y_test_pred = y_test_pred[:y_true.size(0),...] # (6850,207)
        #y_test_pred = y_test_pred.detach().cpu().numpy()
        
        PRED = y_test_pred
        TRUE = y_true.data.cpu().numpy().squeeze(2)
        sigma_p = PRED.std(axis=0)
        mean_p = PRED.mean(axis=0)
        sigma_g = TRUE.std(axis=0)#.squeeze(1)
        mean_g = TRUE.mean(axis=0)#.squeeze(1)


        corr = ((PRED - mean_p) * (TRUE - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
        corr = (corr[(sigma_g != 0)]).mean()
        
        # (2) RAE & RSE
        y_true = y_true.squeeze(-1)
        dat_RSE = normal_std(y_true)
        dat_RAE = torch.mean(torch.abs(y_true - torch.mean(y_true)))
        rae = ((total_loss_l1 / n_samples) / dat_RAE).cpu().numpy()
        rse = ((total_loss_l2 / n_samples)**0.5 / dat_RSE).cpu().numpy()
        print(f'RAE : {rae.round(4)}, RSE : {rse.round(4)}, CORR : {corr.round(4)}')
        
        metric_df = pd.DataFrame(columns = ['rse','rae','corr'])
        metric_df.loc[0,:] = [rse, rae, corr]        
        
    else:
        print('Multi Step forecasting')
        y_preds = []
        for iter, (X, _) in enumerate(dataloader['test_loader'].get_iterator()):
            X = torch.Tensor(X).to(device).transpose(1,3)
            with torch.no_grad():
                y_pred = model(X, gaf).transpose(1,3)
                if iter == 0:
                    CSV_in, CSV_out, global_graph = model.viz(gaf)
            y_preds.append(y_pred.squeeze())
        
        y_preds = torch.cat(y_preds, dim = 0) # (6912,207,12)
        y_preds = y_preds[:y_true.size(0), ...] # (6850,207,12)
        if y_preds.dim() == 2:
            y_preds = y_preds.unsqueeze(-1)

        metric_df = pd.DataFrame(columns=['mae','mape','rmse'],
                                 index = range(configs['output_length']))
        for i in range(configs['output_length']):
            y_preds_scaled = scaler.inverse_transform(y_preds[:,:,i])  # (6850,207)
            metrics = util.metric(y_preds_scaled, y_true[:,:,i]) 
            log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
            print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
            metric_df.iloc[i,:] = [metrics[0], metrics[1], metrics[2]]
            y_test_pred.append(y_preds_scaled.detach().cpu().numpy())
            y_test_real.append(y_true[:,:,i].detach().cpu().numpy())

    #=======================================================#
    # (7) Save Test results
    #=======================================================#
    metric_df.to_csv(PATH_METRIC, index = False)
    with open(PATH_Y_REAL, 'wb') as f:
        np.save(f, np.array(y_test_real))
    with open(PATH_Y_PRED, 'wb') as f:
        np.save(f, np.array(y_test_pred))
    
    with open(PATH_CSV_IN, 'wb') as f:
        np.save(f, CSV_in)
    with open(PATH_CSV_OUT, 'wb') as f:
        np.save(f, CSV_out)
    with open(PATH_GRAPH, 'wb') as f:
        np.save(f, global_graph)

if __name__ == "__main__":
    time_start = time.time()
    main()
    time_end = time.time()
    print("Total time spent: {:.4f}".format(time_end-time_start))
