#!/bin/bash

MODELS=(
  #'BRISQUE'
  #'GM-LOG'
  #'HIGRADE1'
  #'FRIQUEE'
  #'VBLIINDS'
  #'TLVQM'
  #'VIDEVAL'
   #'NIQE'
   #'VSFA'
  #'RAPIQUE'
  #'RAPIQUE_Haar'
  #'RAPIQUE_Db2'
  #'RAPIQUE_Bior22'
  #'RAPIQUE_Haar_ev16'
  #'RAPIQUE_Db2_ev16'
  #'RAPIQUE_Bior22_ev16'
  #'RAPIQUE_Haar_ev8'
  #'RAPIQUE_Db2_ev8'
  #'RAPIQUE_Bior22_ev8'
  #'RAPIQUE_Haar_ev4'
  #'RAPIQUE_Db2_ev4'
  #'RAPIQUE_Bior22_ev4'
  #'CORNIA10K'
  #'HOSA'
   #'Qi_Spatial'
   #'Qi_Spatial_UV'
   #'QUV_NODOG'
  #'ST_FRIQUEE_Haar_ev4'
  #'ST_FRIQUEE_Db2_ev4'
  #'ST_FRIQUEE_Bior22_ev4'
  #'ST_HIGRADE1_Haar_ev16'
  #'ST_HIGRADE1_Db2_ev16'
  #'ST_HIGRADE1_Bior22_ev16'
  #'ST_VBLIINDS_Haar_ev16'
  #'ST_VBLIINDS_Db2_ev16'
  #'ST_VBLIINDS_Bior22_ev16'
  #'ST_VIDEVAL_Haar_ev16'
  #'ST_VIDEVAL_Db2_ev16'
  #'ST_VIDEVAL_Bior22_ev16'
  #'ST_TLVQM_Haar_ev16'
  #'ST_TLVQM_Db2_ev16'
  #'ST_TLVQM_Bior22_ev16'
  #'ST_GM-LOG_Db2'
  #'ST_GM-LOG_Db2_ev16'
  #'ST_GM-LOG_Db2_ev8'
  #'ST_GM-LOG_Db2_ev4'
  #'ST_GM-LOG_Bior22'
  #'ST_GM-LOG_Bior22_ev16'
  #'ST_GM-LOG_Bior22_ev8'
  #'ST_GM-LOG_Bior22_ev4'
  #'ST_QUV_NODOG_Db2'
  #'ST_QUV_NODOG_Db2_ev16'
  #'ST_QUV_NODOG_Db2_ev8'
  #'ST_QUV_NODOG_Haar_ev4'
  #'ST_QUV_NODOG_Db2_ev4'
  'ST_QUV_NODOG_272_GL_Haar'
  'ST_QUV_NODOG_272_GL_Db2'
  'ST_QUV_NODOG_272_GL_Bior22'
  #'ST_QUV_NODOG_GM-LOG_Db2'
  #'ST_QUV_NODOG_GM-LOG_Db2_ev16'
  #'ST_QUV_NODOG_GM-LOG_Db2_ev8'
  #'ST_QUV_NODOG_GM-LOG_Db2_ev4'
  #'ST_QUV_NODOG_GM-LOG_Bior22'
  #'ST_QUV_NODOG_GM-LOG_Bior22_ev8'
  #'ST_QUV_NODOG_GM-LOG_Bior22_ev16'
  #'ST_QUV_NODOG_GM-LOG_Bior22_ev4'
  #'ST_QUV_NODOG_GM-LOG_Db2'
  #'ST_QUV_NODOG_GM-LOG_Db2_ev8'
  #'ST_QUV_NODOG_GM-LOG_Db2_ev4'
  #'Q_GMLOG'
  #'Q_YGMLOG'
  #'QUV_NODOG'
  #'RAPIQUE_Db2_band5'
  #'RAPIQUE_Haar_band6'
  #'RAPIQUE_Haar_band7'
  #'RAPIQUE_Haar_band4'
  #'QUV_NODOG'
  
  
)

DATASETS=(
   'LIVE_VQC'
  #"LIVE_HFR"
)

for m in "${MODELS[@]}"
do
for DS in "${DATASETS[@]}"
do

  feature_file=features/${DS}_${m}_feats.mat
  mos_file=features/${DS}_metadata.csv
  out_file=result/${DS}_${m}_SVR_corr_bc.mat
  log_file=logs/${DS}_regression_bc_25_two.log

#   echo "$m" 
#   echo "${feature_file}"
#   echo "${out_file}"
#   echo "${log_file}"

  cmd="python src/VQC_evaluate_bvqa_features_by_content_regression_traintwo.py"
  cmd+=" --model_name $m"
  cmd+=" --dataset_name ${DS}"
  cmd+=" --feature_file ${feature_file}"
  cmd+=" --mos_file ${mos_file}"
  cmd+=" --out_file ${out_file}"
  cmd+=" --log_file ${log_file}"
  #cmd+=" --num_cont 16"
  #cmd+=" --num_dists 30"
  cmd+=" --use_parallel"
  cmd+=" --log_short"
  cmd+=" --num_iterations 100"

  echo "${cmd}"

  eval ${cmd}
done
done
