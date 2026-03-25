for data_id in DS01
do

for models in DPDG ASTGCNN GRU_CM HAGCN HierCorrPool STFA RGCNU STAGNN LOGO DVGTformer STGNN FC_STGNN 
do

for rate in 0.001
do


python -u "/public1/Shaan/RUL_framework/main.py" \
  --dataset_name 'N_CMAPSS'\
  --model_name $models\
  --DA False\
  --Classify False\
  --Data_id_N_CMAPSS $data_id\
  --batch_size 128\
  --info 'DPDG baslinetest'\
  --train_epochs 200\
  --learning_rate $rate\
  --is_minmax True\

done

done

done
