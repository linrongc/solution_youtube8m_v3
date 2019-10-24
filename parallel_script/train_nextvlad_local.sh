#!/usr/bin/env bash

datapath=/media/linrongc/dream/data/yt8m/frame/2
eval_path=/media/linrongc/dream/data/yt8m/frame/3/validate
test_path=/media/linrongc/dream/data/yt8m/frame/3/test

model_name=NeXtVLADModel
parameters="--groups=8 --nextvlad_cluster_size=128 --nextvlad_hidden_size=2048 \
            --expansion=2 --gating_reduction=8 --drop_rate=0.5 \
            --final_lambda=400 --final_temperature=2 --num_gpu=1"

train_dir=parallel4_20T400_nextvlad_x2_8g_5l2_5drop_128k_2048_80_logistic_final
result_folder=results

echo "model name: " $model_name
echo "model parameters: " $parameters

echo "training directory: " $train_dir
echo "data path: " $datapath
echo "evaluation path: " $eval_path

python parallel_train.py ${parameters} --model=${model_name}  --num_readers=8 --learning_rate_decay_examples 2500000 \
                --video_level_classifier_model=LogisticModel --label_loss=CrossEntropyLoss --start_new_model=False \
                --train_data_pattern=${datapath}/[tv][ra]*/*.tfrecord --train_dir=${train_dir} --frame_features=True \
                --feature_names="rgb,audio" --feature_sizes="1024,128" --batch_size=80 --base_learning_rate=0.0002 \
                --learning_rate_decay=0.8 --l2_penalty=1e-5 --max_step=1000000 --num_epochs=15 --export_model_steps=100000


python parallel_eval.py ${parameters} --batch_size=160 --video_level_classifier_model=LogisticModel --l2_penalty=1e-5 \
               --label_loss=CrossEntropyLoss --eval_data_pattern=${eval_path}/validate*.tfrecord --train_dir ${train_dir} \
               --run_once=True

mkdir -p $result_folder
python inference.py --output_file ${result_folder}/${train_dir}_v3_test.csv \
                    --input_data_pattern=${test_path}/test*.tfrecord --train_dir ${train_dir} \
                    --batch_size=160 --num_readers=8 --segment_labels=False
