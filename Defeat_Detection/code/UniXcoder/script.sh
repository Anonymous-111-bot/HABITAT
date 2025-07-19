#export CUDA_LAUNCH_BLOCKING=1
#python run.py \
#    --output_dir=./saved_models \
#    --model_type=unixcoder \
#    --tokenizer_name=microsoft/unixcoder-base \
#    --model_name_or_path=microsoft/unixcoder-base \
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


#attack
python attack.py \
     --output_dir=./saved_models \
     --model_type=unixcoder \
     --config_name=Salesforce/unixcoder-base \
     --model_name_or_path=Salesforce/unixcoder-base \
     --tokenizer_name=codet5-base \
     --train_data_file=../dataset/train_subs.jsonl \
     --eval_data_file=../dataset/test_subs_0_400.jsonl \
     --eval_data_file_2=../dataset/test_subs_gan_0_400.jsonl \
     --test_data_file=../dataset/test_subs.jsonl \
     --block_size 512 \
     --eval_batch_size 64 \
     --num_of_changes 2 \
     --seed 123456 \
     --transfer_memory_path codet5.json,codetbert.json

#     --transfer_memory_path mab_preferences_codet5.json,mab_preferences_codetbe.json