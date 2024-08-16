accelerate launch --gpu_ids 5 --mixed_precision=bf16 pretrain.py --dataset_name nnheui/star-d2_p5_n50 --remove_unused_columns False\
            --config_name configs/gpt2_12.json \
            --tokenizer_name gpt --num_nodes 50\
            --per_device_train_batch_size 256 --per_device_eval_batch_size 256 \
            --learning_rate 5e-5 --num_train_epochs 20 \
            --output_dir outputs \
            --seed 42 \
            --save_total_limit 1 --eval_steps 500 --evaluation_strategy steps --do_eval \
            --report_to none  --bf16 --overwrite_output_dir

accelerate launch --gpu_ids 5 --mixed_precision=bf16 dpo.py configs/dpo_recipes/gpt2_12/config_full.yaml