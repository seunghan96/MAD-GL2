from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import time
import argparse
import numpy as np
import gzip
import os
import pandas as pd


def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=False, univariate=False, scaler=None
):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """

    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
    
    feature_list = [data]
    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(time_in_day)
        del time_ind, time_in_day
    if add_day_in_week:
        dow = df.index.dayofweek
        dow_tiled = np.tile(dow, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(dow_tiled)
        del dow, dow_tiled

    data = np.concatenate(feature_list, axis=-1)
    del feature_list, df
        
    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    
    for t in range(min_t, max_t):  # t is the index of the last observation.
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])
    
    del data, x_offsets, y_offsets
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    
    if univariate:
        x = x.squeeze(-1)
        y = y.squeeze(-1)
    
    return x, y


def generate_train_val_test(args):
    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y
    if 'h5' in args.df_filename:
        df = pd.read_hdf(args.df_filename)
    elif 'txt.gz' in args.df_filename:
        with gzip.open(args.df_filename, 'rb') as f:
            df = pd.read_csv(f, header=None)

    
    # 0 is the latest observed sample.
    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    # Predict the next one hour
    y_offsets = np.sort(np.arange(args.y_start, (seq_length_y + args.y_start), 1))
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    if ('metr-la' in args.df_filename) or ('pems-bay' in args.df_filename):
        tid = True
    else:
        tid = False
    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=tid,
        add_day_in_week=args.dow,
        univariate = args.uni
    )
    
    # Write the data into npz file.
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train
    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    x_test, y_test = x[-num_test:], y[-num_test:]

    if args.min_max:
        print('Min Max Scaling')
        temp = np.transpose(x_train,(2,0,1))
        temp = temp.reshape(temp.shape[0],-1)
        max_val = np.max(temp,1)
        min_val = np.min(temp,1)
        
        x_train = (x_train - min_val) / (max_val - min_val)
        y_train = (y_train - min_val) / (max_val - min_val)
        x_val = (x_val - min_val) / (max_val - min_val)
        y_val = (y_val - min_val) / (max_val - min_val)
        x_test = (x_test - min_val) / (max_val - min_val)
        y_test = (y_test - min_val) / (max_val - min_val)
    
    if args.uni:
        x_train = np.expand_dims(x_train,-1)
        y_train = np.expand_dims(y_train,-1)
        x_val = np.expand_dims(x_val,-1)
        y_val = np.expand_dims(y_val,-1)
        x_test = np.expand_dims(x_test,-1)
        y_test = np.expand_dims(y_test,-1)

    print("x shape: ", x.shape, ", y shape: ", y.shape)
    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        
        
        np.savez_compressed(
            os.path.join(args.output_dir, f"{cat}.npz"),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/METR-LA", help="Output directory.")
    parser.add_argument("--df_filename", type=str, default="data/metr-la.h5", help="Raw traffic readings.",)
    parser.add_argument("--seq_length_x", type=int, default=12, help="Sequence Length.",)
    parser.add_argument("--seq_length_y", type=int, default=12, help="Sequence Length.",)
    parser.add_argument("--y_start", type=int, default=1, help="Y pred start", )
    parser.add_argument("--uni", action='store_true',)
    parser.add_argument("--dow", action='store_true',)
    parser.add_argument("--min_max", action='store_true',)

    args = parser.parse_args()
    if os.path.exists(args.output_dir):
        reply = str(input(f'{args.output_dir} exists. Do you want to overwrite it? (y/n)')).lower().strip()
        if reply[0] != 'y': exit
    else:
        os.makedirs(args.output_dir)
    generate_train_val_test(args)
