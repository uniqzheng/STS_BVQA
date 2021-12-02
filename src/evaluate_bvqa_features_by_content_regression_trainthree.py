# -*- coding: utf-8 -*-
"""
This script shows how to apply 80-20 holdout train and validate regression model to predict
MOS from the features
"""
import pandas
import scipy.io
import numpy as np
import argparse
import time
import math
import os, sys
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, PredefinedSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from scipy.optimize import curve_fit
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import scipy.stats
from concurrent import futures
import functools
import warnings
warnings.filterwarnings("ignore")
# ----------------------- Set System logger ------------- #
class Logger:
  def __init__(self, log_file):
    self.terminal = sys.stdout
    self.log = open(log_file, "a")

  def write(self, message):
    self.terminal.write(message)
    self.log.write(message)  

  def flush(self):
    #this flush method is needed for python 3 compatibility.
    #this handles the flush command by doing nothing.
    #you might want to specify some extra behavior here.
    pass


def arg_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_name', type=str, default='RAPIQUE',
                      help='Evaluated BVQA model name.')
  parser.add_argument('--dataset_name', type=str, default='KONVID_1K',
                      help='Evaluation dataset.') 
  parser.add_argument('--feature_file', type=str,
                      default='feat_files/KONVID_1K_RAPIQUE_feats.mat',
                      help='Pre-computed feature matrix.')
  parser.add_argument('--mos_file', type=str,
                      default='mos_files/KONVID_1K_metadata.csv',
                      help='Dataset MOS scores.')
  #parser.add_argument('--num_cont', type=int,
  #                    default=10,
  #                    help='Number of contents.')
  #parser.add_argument('--num_dists', type=int,
  #                    default=15,
  #                    help='Number of distortions per content.')
  parser.add_argument('--out_file', type=str,
                      default='result/KONVID_1K_RAPIQUE_SVR_corr.mat',
                      help='Output correlation results')
  parser.add_argument('--log_file', type=str,
                      default='logs/KONVID_1K_RAPIQUE_SVR.log',
                      help='Log files.')
  parser.add_argument('--color_only', action='store_true',
                      help='Evaluate color values only. (Only for YouTube UGC)')
  parser.add_argument('--log_short', action='store_true',
                      help='Whether log short')
  parser.add_argument('--use_parallel', action='store_true',
                      help='Use parallel for iterations.')
  parser.add_argument('--num_iterations', type=int, default=50,
                      help='Number of iterations of train-test splits')
  parser.add_argument('--max_thread_count', type=int, default=4,
                      help='Number of threads.')
  args = parser.parse_args()
  return args

def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
  # 4-parameter logistic function
  logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
  yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
  return yhat

def compute_metrics(y_pred, y):
  '''
  compute metrics btw predictions & labels
  '''
  # compute SRCC & KRCC
  SRCC = scipy.stats.spearmanr(y, y_pred)[0]
  try:
    KRCC = scipy.stats.kendalltau(y, y_pred)[0]
  except:
    KRCC = scipy.stats.kendalltau(y, y_pred, method='asymptotic')[0]

  # logistic regression btw y_pred & y
  beta_init = [np.max(y), np.min(y), np.mean(y_pred), 0.5]
  popt, _ = curve_fit(logistic_func, y_pred, y, p0=beta_init, maxfev=int(1e8))
  y_pred_logistic = logistic_func(y_pred, *popt)
  
  # compute  PLCC RMSE
  PLCC = scipy.stats.pearsonr(y, y_pred_logistic)[0]
  RMSE = np.sqrt(mean_squared_error(y, y_pred_logistic))
  return [SRCC, KRCC, PLCC, RMSE]

def formatted_print(snapshot, params, duration):
  print('======================================================')
  print('params: ', params)
  print('SRCC_train: ', snapshot[0])
  print('KRCC_train: ', snapshot[1])
  print('PLCC_train: ', snapshot[2])
  print('RMSE_train: ', snapshot[3])
  print('======================================================')
  print('SRCC_test: ', snapshot[4])
  print('KRCC_test: ', snapshot[5])
  print('PLCC_test: ', snapshot[6])
  print('RMSE_test: ', snapshot[7])
  print('======================================================')
  print(' -- ' + str(duration) + ' seconds elapsed...\n\n')

def final_avg(snapshot):
  def formatted(args, pos):
    mean = np.nanmean(list(map(lambda x: x[pos], snapshot)))
    stdev = np.nanstd(list(map(lambda x: x[pos], snapshot)))
    print('{}: {} (std: {})'.format(args, mean, stdev))

  print('======================================================')
  print('Average training results among all repeated 80-20 holdouts:')
  formatted("SRCC Train", 0)
  formatted("KRCC Train", 1)
  formatted("PLCC Train", 2)
  formatted("RMSE Train", 3)
  print('======================================================')
  print('Average testing results among all repeated 80-20 holdouts:')
  formatted("SRCC Test", 4)
  formatted("KRCC Test", 5)
  formatted("PLCC Test", 6)
  formatted("RMSE Test", 7)
  print('\n\n')

def idx_expand(idx, num_dists):
  idx_out = []
  for ii in idx:
    idx_out.extend(range(ii*num_dists,(ii+1)*num_dists))
  return idx_out

#def evaluate_bvqa_one_split(i, X, y, num_cont, num_dists, log_short):
def evaluate_bvqa_one_split(i, X, y, log_short):
  if log_short:
    print('{} th repeated holdout test'.format(i))
    t_start = time.time()
  # train test split
  idx_train, idx_test = train_test_split(range(1201), test_size=0.2, #Youtube-UGC-1201, LIVE_VQC-585, LIVE_GAME-600, KoNVid-1200
      random_state=math.ceil(8.8*i))
  X_train = X[idx_train,:]
  #print(X_train.shape)
  X_test = X[idx_test,:]
  #print(X_test.shape)
  y_train = y[idx_train]
  #print(y_train.shape)
  y_test = y[idx_test]
  #print(y_test.shape)
  #print(X_train.shape)
  #print(X_test.shape)

  k_folds = 4
  #split_index_cv = [idx // (len(idx_train)//k_folds*num_dists)
  #                  for idx in range(len(idx_train)*num_dists)]  
  split_index_cv = [idx // (len(idx_train)//k_folds)
                    for idx in range(len(idx_train))] 
  pdsplit = PredefinedSplit(test_fold=split_index_cv)
  
  #LIVE_VQC  RN_RP_FV 0:2048] 2048:2728] 2728:3000]
  #Youtube-UGC  HG_RP_FV 0:216] 216:896] 896:1168]
  #KoNVid  FV_RP_FV 0:272] 272:952] 952:1224]
  #LIVE_GAME  RP_FV_FV  0:680] 680: 952] 952:1224]
  X_train_1 = X_train[:,0:216]  
  X_train_2 = X_train[:,216:896]
  X_train_3 = X_train[:,896:1168]
  X_test_1 = X_test[:,0:216]
  X_test_2 = X_test[:,216:896]
  X_test_3 = X_test[:,896:1168]

  # grid search CV on the training set
  if X_train_1.shape[1] <= 1100:
    print(f'{X_train_1.shape[1]}-dim features, using SVR')
     #grid search CV on the training set
    param_grid = {'C': np.logspace(1, 10, 10, base=2),
                  'gamma': np.logspace(-8, 1, 10, base=2)}
    grid_1 = RandomizedSearchCV(SVR(kernel = 'rbf'), param_grid, cv=pdsplit, n_jobs=-1)
  else:
    print(f'{X_train_1.shape[1]}-dim features, using LinearSVR')
    # grid search on liblinear 
    param_grid = {'C': [0.001, 0.01, 0.1, 1., 2.5, 5., 10.],
                  'epsilon': [0.001, 0.01, 0.1, 1., 2.5, 5., 10.]}
    grid_1 = RandomizedSearchCV(LinearSVR(), param_grid, cv=pdsplit)
    
    # grid search CV on the training set
  if X_train_2.shape[1] <= 1100:
    print(f'{X_train_2.shape[1]}-dim features, using SVR')
     #grid search CV on the training set
    param_grid = {'C': np.logspace(1, 10, 10, base=2),
                  'gamma': np.logspace(-8, 1, 10, base=2)}
    grid_2 = RandomizedSearchCV(SVR(kernel = 'rbf'), param_grid, cv=pdsplit, n_jobs=-1)
  else:
    print(f'{X_train_2.shape[1]}-dim features, using LinearSVR')
    # grid search on liblinear 
    param_grid = {'C': [0.001, 0.01, 0.1, 1., 2.5, 5., 10.],
                  'epsilon': [0.001, 0.01, 0.1, 1., 2.5, 5., 10.]}
    grid_2 = RandomizedSearchCV(LinearSVR(), param_grid, cv=pdsplit)
    
    # grid search CV on the training set
  if X_train_3.shape[1] <= 1100:
    print(f'{X_train_3.shape[1]}-dim features, using SVR')
     #grid search CV on the training set
    param_grid = {'C': np.logspace(1, 10, 10, base=2),
                  'gamma': np.logspace(-8, 1, 10, base=2)}
    grid_3 = RandomizedSearchCV(SVR(kernel = 'rbf'), param_grid, cv=pdsplit, n_jobs=-1)
  else:
    print(f'{X_train_3.shape[1]}-dim features, using LinearSVR')
    # grid search on liblinear 
    param_grid = {'C': [0.001, 0.01, 0.1, 1., 2.5, 5., 10.],
                  'epsilon': [0.001, 0.01, 0.1, 1., 2.5, 5., 10.]}
    grid_3 = RandomizedSearchCV(LinearSVR(), param_grid, cv=pdsplit)

  

  ## Scaler first
  scaler_1 = preprocessing.MinMaxScaler().fit(X_train_1)  
  X_train_1 = scaler_1.transform(X_train_1)
  X_test_1 = scaler_1.transform(X_test_1)
  
  scaler_2 = preprocessing.MinMaxScaler().fit(X_train_2)  
  X_train_2 = scaler_2.transform(X_train_2)
  X_test_2 = scaler_2.transform(X_test_2)
  
  scaler_3 = preprocessing.MinMaxScaler().fit(X_train_3)  
  X_train_3 = scaler_3.transform(X_train_3)
  X_test_3 = scaler_3.transform(X_test_3)

  # grid search
  grid_1.fit(X_train_1, y_train)
  best_params_1 = grid_1.best_params_
  
  grid_2.fit(X_train_2, y_train)
  best_params_2 = grid_2.best_params_
  
  grid_3.fit(X_train_3, y_train)
  best_params_3 = grid_3.best_params_
  
  # init model
  if X_train_1.shape[1] <= 1100:
    regressor_1 = SVR(kernel = 'rbf', C=best_params_1['C'], gamma=best_params_1['gamma'])
    #regressor = RandomForestRegressor(**best_params)
  else:
    regressor_1 = LinearSVR(C=best_params_1['C'], epsilon=best_params_1['epsilon'])
    
   # init model
  if X_train_2.shape[1] <= 1100:
    regressor_2 = SVR(kernel = 'rbf', C=best_params_2['C'], gamma=best_params_2['gamma'])
    #regressor = RandomForestRegressor(**best_params)
  else:
    regressor_2 = LinearSVR(C=best_params_2['C'], epsilon=best_params_2['epsilon'])
    
   # init model
  if X_train_3.shape[1] <= 1100:
    regressor_3 = SVR(kernel = 'rbf', C=best_params_3['C'], gamma=best_params_3['gamma'])
    #regressor = RandomForestRegressor(**best_params)
  else:
    regressor_3 = LinearSVR(C=best_params_3['C'], epsilon=best_params_3['epsilon'])
    
  # re-train the model using the best alpha
  regressor_1.fit(X_train_1, y_train)
  regressor_2.fit(X_train_2, y_train)
  regressor_3.fit(X_train_3, y_train)
  # predictions
  y_train_pred_1 = regressor_1.predict(X_train_1)
  y_train_pred_2 = regressor_2.predict(X_train_2)
  y_train_pred_3 = regressor_3.predict(X_train_3)
  y_test_1 = regressor_1.predict(X_test_1)
  y_test_2 = regressor_2.predict(X_test_2)
  y_test_3 = regressor_3.predict(X_test_3)
  # compute metrics
  y_train_pred = [(x+y+z)/3 for x, y, z in zip(y_train_pred_1,y_train_pred_2,y_train_pred_3)]
  y_test_pred = [(x+y+z)/3 for x, y, z in zip(y_test_1,y_test_2,y_test_3)]
  metrics_train = compute_metrics(y_train_pred, y_train)
  metrics_test = compute_metrics(y_test_pred, y_test)
  # print values
  if not log_short:
    t_end = time.time()
    formatted_print(metrics_train + metrics_test, best_params, (t_end - t_start))
  return metrics_train, metrics_test
  
def main(args):
  df = pandas.read_csv(args.mos_file, skiprows=[], header=None)
  array = df.values
  if args.dataset_name == 'LIVE_VQC':
      y = array[1:,1]
  elif args.dataset_name == 'KoNVid':
      y = array[1:,4]
  elif args.dataset_name == 'Youtube-UGC':
      y = array[1:,4]
  elif args.dataset_name == 'LIVE_GAME':
      y = array[1:,1]
  y = np.array(list(y), dtype=np.float)
  X_mat = scipy.io.loadmat(args.feature_file)
  X = np.asarray(X_mat['feats_mat'], dtype=np.float)
  #print(X)


  '''57 grayscale videos in YOUTUBE_UGC dataset, we do not consider them for fair comparison'''
  if args.color_only and args.dataset_name == 'YOUTUBE_UGC':
      gray_indices = [3,6,10,22,23,46,51,52,68,74,77,99,103,122,136,141,158,173,368,426,467,477,506,563,594,\
      639,654,657,666,670,671,681,690,697,702,703,710,726,736,764,768,777,786,796,977,990,1012,\
      1015,1023,1091,1118,1205,1282,1312,1336,1344,1380]
      gray_indices = [idx - 1 for idx in gray_indices]
      X = np.delete(X, gray_indices, axis=0)
      y = np.delete(y, gray_indices, axis=0)
  ## preprocessing
  X[np.isinf(X)] = np.nan
  imp = SimpleImputer(missing_values=np.nan, strategy='mean').fit(X)
  X = imp.transform(X)
  
  #scipy.io.savemat('new.mat',{'STfeat': X})

  all_iterations = []
  t_overall_start = time.time()
  # 100 times random train-test splits
  if args.use_parallel is True:
    evaluate_bvqa_one_split_partial = functools.partial(
       evaluate_bvqa_one_split, X=X, y=y, log_short=args.log_short)
    with futures.ThreadPoolExecutor(max_workers=args.max_thread_count) as executor:
      iters_future = [
          executor.submit(evaluate_bvqa_one_split_partial, i)
          for i in range(1, args.num_iterations)]
      for future in futures.as_completed(iters_future):
        metrics_train, metrics_test = future.result()
        all_iterations.append(metrics_train + metrics_test)
  else:
    for i in range(1, args.num_iterations):
      metrics_train, metrics_test = evaluate_bvqa_one_split(
          i, X, y, args.log_short)
      all_iterations.append(metrics_train + metrics_test)

  # formatted print overall iterations
  final_avg(all_iterations)
  print('Overall {} secs lapsed..'.format(time.time() - t_overall_start))
  # save figures
  dir_path = os.path.dirname(args.out_file)
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)
  scipy.io.savemat(args.out_file, 
      mdict={'all_iterations': np.asarray(all_iterations,dtype=np.float)})

if __name__ == '__main__':
  args = arg_parser()
  log_file = args.log_file
  log_dir = os.path.dirname(log_file)
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)
  sys.stdout = Logger(log_file)
  print(args)
  main(args)

'''

python evaluate_bvqa_features_regression.py \
  --model_name BRISQUE \
  --dataset_name LIVE_VQC \
  --feature_file mos_feat_files/KONIQ_10K_BRISQUE_feats.mat \
  --mos_file mos_feat_files/KONIQ_10K_metadata.csv \
  --out_file result/KONIQ_10K_BRISQUE_SVR_corr.mat \
  --use_parallel


'''
