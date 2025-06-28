
tasks="mbpp"
nshots="3"
lengths="512"
temperatures="0.2"

model=Dream-org/Dream-v0-Base-7B
# Create arrays from space-separated strings
read -ra TASKS_ARRAY <<< "$tasks"
read -ra NSHOTS_ARRAY <<< "$nshots"
read -ra LENGTH_ARRAY <<< "$lengths"
read -ra TEMP_ARRAY <<< "$temperatures"

export HF_ALLOW_CODE_EVAL=1
# accelerate launch --main_process_port 29510 eval_dream_original.py --model dream \
#     --model_args pretrained=${model},max_new_tokens=512,diffusion_steps=512,temperature=0.2,top_p=0.95,add_bos_token=true,escape_until=true,use_cache=true,cache_type="decoded",cache_steps=4,shift_type="un",alg='conv' \
#     --tasks humaneval \
#     --num_fewshot 0 \
#     --batch_size 1 \
#     --output_path evals_results/humaneval-ns0 \
#     --log_samples \
#     --confirm_run_unsafe_code 


# accelerate launch --main_process_port 29510 eval_dream_original.py --model dream \
#     --model_args pretrained=${model},max_new_tokens=512,diffusion_steps=512,temperature=0.2,top_p=0.95,add_bos_token=true,escape_until=true, \
#     --tasks humaneval \
#     --num_fewshot 0 \
#     --batch_size 1 \
#     --output_path evals_results/humaneval-ns0 \
#     --log_samples \
#     --confirm_run_unsafe_code 


## NOTICE: use postprocess for humaneval
# python postprocess_code.py {the samples_xxx.jsonl file under output_path}

# Iterate through the arrays
for i in "${!TASKS_ARRAY[@]}"; do
    output_path=evals_results/${TASKS_ARRAY[$i]}-ns${NSHOTS_ARRAY[$i]}
    echo "Task: ${TASKS_ARRAY[$i]}, Shots: ${NSHOTS_ARRAY[$i]}; Output: $output_path"
    CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch eval_dream.py --model dream \
        --model_args pretrained=${model},max_new_tokens=${LENGTH_ARRAY[$i]},diffusion_steps=${LENGTH_ARRAY[$i]},add_bos_token=true,temperature=${TEMP_ARRAY[$i]},top_p=0.95,use_cache=true,cache_type="decoded",cache_steps=4,shift_type="un",alg='conv' \
        --tasks ${TASKS_ARRAY[$i]} \
        --num_fewshot ${NSHOTS_ARRAY[$i]} \
        --batch_size 1 \
        --output_path $output_path \
        --log_samples \
        --confirm_run_unsafe_code
done



# Iterate through the arrays
# for i in "${!TASKS_ARRAY[@]}"; do
#     output_path=evals_results/${TASKS_ARRAY[$i]}-ns${NSHOTS_ARRAY[$i]}
#     echo "Task: ${TASKS_ARRAY[$i]}, Shots: ${NSHOTS_ARRAY[$i]}; Output: $output_path"
#     accelerate launch eval_dream_original.py --model dream \
#         --model_args pretrained=${model},max_new_tokens=${LENGTH_ARRAY[$i]},diffusion_steps=${LENGTH_ARRAY[$i]},add_bos_token=true,temperature=${TEMP_ARRAY[$i]},top_p=0.95 \
#         --tasks ${TASKS_ARRAY[$i]} \
#         --num_fewshot ${NSHOTS_ARRAY[$i]} \
#         --batch_size 1 \
#         --limit 100 \
#         --output_path $output_path \
#         --log_samples \
#         --confirm_run_unsafe_code
# done