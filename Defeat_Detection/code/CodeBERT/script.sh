#export CUDA_LAUNCH_BLOCKING=1
#python run.py \
#    --output_dir=./saved_models \
#    --model_type=roberta \
#    --tokenizer_name=microsoft/codebert-base-mlm \
#    --model_name_or_path=microsoft/codebert-base-mlm \
#    --do_train \
#    --train_data_file=../dataset/train.jsonl \
#    --eval_data_file=../dataset/valid.jsonl \
#    --test_data_file=../dataset/test.jsonl \
#    --epoch 5 \
#    --block_size 8 \
#    --train_batch_size 32 \
#    --eval_batch_size 64 \
#    --learning_rate 2e-5 \
#    --max_grad_norm 1.0 \
#    --evaluate_during_training \
#    --seed 123456  2>&1 | tee train.log


python attack.py \
     --output_dir=./saved_models \
     --model_type=roberta \
     --tokenizer_name=microsoft/codebert-base-mlm \
     --model_name_or_path=microsoft/codebert-base-mlm \
     --base_model=microsoft/codebert-base-mlm \
     --use_ga \
     --train_data_file=../dataset/train_subs.jsonl \
     --eval_data_file=../dataset/test_subs_0_400.jsonl \
     --eval_data_file_2=../dataset/test_subs_gan_0_400.jsonl \
     --test_data_file=../dataset/test_subs.jsonl \
     --block_size 1024 \
     --eval_batch_size 64 \
     --num_of_changes 2 \
     --seed 123456 \
     --transfer_memory_path mab_preferences_unix.json,mab_preferences_codet5.json


     #     --use_mab_memory \
#     --mab_memory_path codetbert.json
