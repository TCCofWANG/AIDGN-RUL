for data_id in FD001
do

for Data_id_tests in FD001
do

for rate in 0.001
do

for input_lengths in 50
do

for models in DPDG ASTGCNN GRU_CM HAGCN HierCorrPool STFA RGCNU STAGNN LOGO DVGTformer STGNN FC_STGNN
do

python -u "/public1/Shaan/RUL_framework/FD001_main.py" \
  --dataset_name 'CMAPSS'\
  --model_name $models\
  --input_length $input_lengths\
  --Data_id_CMAPSS $data_id\
  --Data_id_CMAPSS_test $Data_id_tests\
  --info 'New models test'\
  --learning_rate $rate\

done

done

done

done

done

