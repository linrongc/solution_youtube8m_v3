#!/usr/bin/env bash

datapath=/media/linrongc/dream/data/yt8m/frame/3/validate
eval_path=/media/linrongc/dream/data/yt8m/frame/3/validate_strat_split/test
test_path=/media/linrongc/dream/data/yt8m/frame/3/test

model_name=NeXtVLADModel
parameters="--groups=8 --nextvlad_cluster_size=256 --nextvlad_hidden_size=2048 \
            --expansion=4 --gating_reduction=8 --drop_rate=0.75"

train_dir=nextvlad_8g_x4_5l2_5drop_256k_2048_1x80_logistic_bn_final_610k_5f_10ep_75drop_512_pretrain_validate_train_4l2
pretrain_model=nextvlad_8g_x2_5l2_5drop_128k_2048_80_logistic_final/model.ckpt-610010

log_dir=./
result_folder=results

echo "model name: " $model_name
echo "model parameters: " $parameters

echo "training directory: " $train_dir
echo "data path: " $datapath
echo "evaluation path: " $eval_path

python seg_train.py ${parameters} --model=${model_name} --num_readers=8 --learning_rate_decay_examples 1000000 --num_epochs=10 \
                --video_level_classifier_model=LogisticModel --label_loss=CrossEntropyLoss --start_new_model=False \
                --train_data_pattern=${datapath}/*.tfrecord --train_dir=${train_dir} --frame_features=True \
                --feature_names="rgb,audio" --feature_sizes="1024,128" --batch_size=512 --base_learning_rate=0.0002 \
                --learning_rate_decay=0.8 --l2_penalty=1e-4 --max_step=700000 --num_gpu=4 \
                --segment_labels=True --export_model_steps=1000 \
                --pretrain_model_path=${pretrain_model}

python eval.py ${parameters} --batch_size=1024 --video_level_classifier_model=LogisticModel --l2_penalty=1e-5\
               --label_loss=CrossEntropyLoss --eval_data_pattern=${eval_path}/*.tfrecord --train_dir ${train_dir} \
               --run_once=True --segment_labels=True

mkdir -p $result_folder
python inference.py --output_file ${result_folder}/${train_dir}_test_k1000.csv \
                    --input_data_pattern=${test_path}/*.tfrecord --train_dir ${train_dir} \
                    --batch_size=60 --num_readers=8 --segment_labels=True --top_k=1000