import argparse

import numpy as np
import os

from pyts.image import GramianAngularField, MarkovTransitionField, RecurrencePlot

def generate_images(args):
    data_name = args.data
    size = args.size
    img_type = args.type
    
    data_path = "data/{}/train.npz".format(data_name)
    output_dir = "data/{}".format(data_name)
    
    train_data = np.load(data_path)
    train_data = train_data['x']
    train_data_flat1 = train_data[::train_data.shape[1],:,:,:]
    train_data_flat1 = train_data_flat1.reshape(-1,train_data.shape[2],train_data.shape[3])
    train_data_flat2 = train_data[-1,-train_data.shape[0]%train_data.shape[1]:,:,:]
    train_data_final = np.concatenate([train_data_flat1, train_data_flat2])
    x = train_data_final[:,:,0].T    
    print('X :',x.shape)
    
    if img_type == 'GAF':
        save_path = os.path.join(output_dir, "GAF_{}.txt".format(data_name))
        img_trans = GramianAngularField(image_size = min(x.shape[1], size))
        x = img_trans.transform(np.array(x))
        x = x.reshape(x.shape[0],-1)
        
    elif img_type == 'MTF':
        save_path = os.path.join(output_dir, "MTF_{}.txt".format(data_name))
        img_trans = MarkovTransitionField(image_size = min(x.shape[1], size))
        temp = []
        for i in range(0,x.shape[0],9):
            print(i)
            x_temp = x[i:i+9]
            x_temp = img_trans.transform(np.array(x_temp))
            x_temp = x_temp.reshape(x_temp.shape[0],-1)
            temp.append(x_temp)
        x = np.vstack(temp)    
    '''
    elif img_type == 'RP':
        save_path = os.path.join(output_dir, "RP_{}.txt".format(data_name))
        img_trans = RecurrencePlot()
        x = img_trans.transform(np.array(x))
        x = x.reshape(x.shape[0],-1)
    '''
    print('Image Size:',x.shape)
    
    np.savetxt(save_path, x)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="METR-LA", help="Data name")
    parser.add_argument("--size", type=int, default=224, help="GAF image size (n x n)")
    parser.add_argument("--type", type=str, default='GAF', help="GAF or MTF or RP")
    args = parser.parse_args()
    generate_images(args)
