# 1. Datasets

## 1-1. Multi-step Forecasting datasets

- NASDAQ10
- NASDAQ50

Above datasets are in the `data` folder.



- METR-LA
- PEMS-BAY

Download above datasets from [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g) provided by [Li et al.](https://github.com/liyaguang/DCRNN.git) . 

Move them into the `data` folder.



## 1-2. Single-step Forecasting datasets

- Electricity
- Exchange-rate

Download above datasets from https://github.com/laiguokun/multivariate-time-series-data. 

Uncompress them and move them to the `data` folder.



# 2. Installations

```bash
pip install pyts
```



# 3. Data Preprocessing

Split data into train, validation, test datasets

```bash
python generate_training_data.py --output_dir=data/METR-LA --df_filename=data/metr-la.h5 --seq_length_x 12 --seq_length_y 12
python generate_training_data.py --output_dir=data/PEMS-BAY --df_filename=data/pems-bay.h5 --seq_length_x 12 --seq_length_y 12

python generate_training_data.py --output_dir=data/NASDAQ10 --df_filename=data/nasdaq10.h5 --seq_length_x 15 --seq_length_y 5
python generate_training_data.py --output_dir=data/NASDAQ50 --df_filename=data/nasdaq50.h5 --seq_length_x 15 --seq_length_y 5

python generate_training_data.py --output_dir=data/EXCHANGE_H3 --df_filename=data/exchange_rate.txt.gz --seq_length_x 168 --seq_length_y 1 --y_start 3
python generate_training_data.py --output_dir=data/EXCHANGE_H6 --df_filename=data/exchange_rate.txt.gz --seq_length_x 168 --seq_length_y 1 --y_start 6
python generate_training_data.py --output_dir=data/EXCHANGE_H12 --df_filename=data/exchange_rate.txt.gz --seq_length_x 168 --seq_length_y 1 --y_start 12
python generate_training_data.py --output_dir=data/EXCHANGE_H24 --df_filename=data/exchange_rate.txt.gz --seq_length_x 168 --seq_length_y 1 --y_start 24


python generate_training_data.py --output_dir=data/ELECTRICITY_H3 --df_filename=data/electricity.txt.gz --seq_length_x 48 --seq_length_y 1 --y_start 3
python generate_training_data.py --output_dir=data/ELECTRICITY_H6 --df_filename=data/electricity.txt.gz --seq_length_x 48 --seq_length_y 1 --y_start 6
python generate_training_data.py --output_dir=data/ELECTRICITY_H12 --df_filename=data/electricity.txt.gz --seq_length_x 48 --seq_length_y 1 --y_start 12
python generate_training_data.py --output_dir=data/ELECTRICITY_H24 --df_filename=data/electricity.txt.gz --seq_length_x 48 --seq_length_y 1 --y_start 24
```



# 3. Time Series to GAF images

```bash
python generate_gaf_images.py --data='METR-LA'
python generate_gaf_images.py --data='PEMS-BAY'

python generate_gaf_images.py --data='NASDAQ10'
python generate_gaf_images.py --data='NASDAQ50'

python generate_gaf_images.py --data='EXCHANGE_H3'
python generate_gaf_images.py --data='EXCHANGE_H6'
python generate_gaf_images.py --data='EXCHANGE_H12'
python generate_gaf_images.py --data='EXCHANGE_H24'

python generate_gaf_images.py --data='ELECTRICITY_H3'
python generate_gaf_images.py --data='ELECTRICITY_H6'
python generate_gaf_images.py --data='ELECTRICITY_H12'
python generate_gaf_images.py --data='ELECTRICITY_H24'
```



# 4. Training

## 4-1. Multi-step Forecasting datasets

### Traffic domain

```bash
python train.py --data 'data/METR-LA'
python train.py --data 'data/PEMS-BAY'
```



### Stock price domain

```bash
python train.py --data 'data/NASDAQ10'
python train.py --data 'data/NASDAQ50'
```



## 4-2. Single-step Forecasting datasets

### Exchange-Rate

```bash
python train.py --data 'data/EXCHANGE_H3'
python train.py --data 'data/EXCHANGE_H6'
python train.py --data 'data/EXCHANGE_H12'
python train.py --data 'data/EXCHANGE_H24'
```



### Electricity

```bash
python train.py --data 'data/ELECTRICITY_H3'
python train.py --data 'data/ELECTRICITY_H6'
python train.py --data 'data/ELECTRICITY_H12'
python train.py --data 'data/ELECTRICITY_H24'
```



# 5. Testing

## 5-1. Multi-step Forecasting datasets

### Traffic domain

```bash
python test_multistep.py --data 'data/METR-LA'
python test_multistep.py --data 'data/PEMS-BAY'
```



### Stock price domain

```bash
python test_multistep.py --data 'data/NASDAQ10'
python test_multistep.py --data 'data/NASDAQ50'
```



## 5-2. Single-step Forecasting datasets

### Exchange-Rate

```bash
python test_singlestep.py --data 'data/EXCHANGE_H3'
python test_singlestep.py --data 'data/EXCHANGE_H6'
python test_singlestep.py --data 'data/EXCHANGE_H12'
python test_singlestep.py --data 'data/EXCHANGE_H24'
```



### Electricity

```bash
python test_singlestep.py --data 'data/ELECTRICITY_H3'
python test_singlestep.py --data 'data/ELECTRICITY_H6'
python test_singlestep.py --data 'data/ELECTRICITY_H12'
python test_singlestep.py --data 'data/ELECTRICITY_H24'
```



