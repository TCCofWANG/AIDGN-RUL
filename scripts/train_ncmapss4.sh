for data_id in DS07
do

for models in AIGCN 
do


for rate in 0.001 0.0005 0.0001
do

for dff in 256
do

python -u "/public1/Shaan/RUL_framework/main.py" \
  --dataset_name 'N_CMAPSS'\
  --model_name $models\
  --DA False\
  --Classify False\
  --Data_id_N_CMAPSS $data_id\
  --d_ff $dff\
  --batch_size 128\
  --info 'Cross gen test'\
  --train_epochs 200\
  --learning_rate $rate\
  --is_minmax True\

done

done

done

done