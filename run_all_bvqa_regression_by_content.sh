#!/bin/bash

MODELS=(
  'BRISQUE_STS_resize'
  'GM-LOG_STS_resize'
  'NIQE_STS_resize'
  'HIGRADE_STS_resize'
  'RAPIQUE_spatial_STS_resize'
  'FAVER_spatial_STS_resize'
  'PAQ2PIQ_STS_resize'
  'resnet50_STS_resize'
  'RN_RP_FV' #LIVE-VQC
  'HG_RP_FV' #Youtube-UGC  
  'FV_RP_FV' #KoNVid  
  'RP_FV_FV' #LIVE_GAME  
)

DATASETS=(
  #'KoNVid'
  #'LIVE_VQC'
  'Youtube-UGC'
  #'LIVE_GAME'
)

for m in "${MODELS[@]}"
do
for DS in "${DATASETS[@]}"
do

  feature_file=features/${DS}_${m}_feats.mat
  mos_file=features/${DS}_metadata.csv
  out_file=result/${DS}_${m}_SVR_corr_bc.mat
  log_file=logs/${DS}_regression_bc.log

#   echo "$m" 
#   echo "${feature_file}"
#   echo "${out_file}"
#   echo "${log_file}"

  cmd="python src/evaluate_bvqa_features_by_content_regression.py"
  cmd+=" --model_name $m"
  cmd+=" --dataset_name ${DS}"
  cmd+=" --feature_file ${feature_file}"
  cmd+=" --mos_file ${mos_file}"
  cmd+=" --out_file ${out_file}"
  cmd+=" --log_file ${log_file}"
  cmd+=" --use_parallel"
  cmd+=" --log_short"
  cmd+=" --num_iterations 100"

  echo "${cmd}"

  eval ${cmd}
done
done
