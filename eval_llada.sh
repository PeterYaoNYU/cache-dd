# pip install transformers==4.49.0 lm_eval==0.4.8 accelerate==0.34.2
# pip install antlr4-python3-runtime==4.11 math_verify sympy hf_xet


# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true


# conditional likelihood estimation benchmarks
# accelerate launch eval_llada.py --tasks gpqa_main_n_shot --num_fewshot 5 --model llada_dist --batch_size 8 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.5,is_check_greedy=False,mc_num=128

# accelerate launch eval_llada.py --tasks truthfulqa_mc2 --num_fewshot 0 --model llada_dist --batch_size 8 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=2.0,is_check_greedy=False,mc_num=128

# accelerate launch eval_llada.py --tasks arc_challenge --num_fewshot 0 --model llada_dist --batch_size 8 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.5,is_check_greedy=False,mc_num=128

# accelerate launch eval_llada.py --tasks hellaswag --num_fewshot 0 --model llada_dist --batch_size 8 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.5,is_check_greedy=False,mc_num=128

# accelerate launch eval_llada.py --tasks winogrande --num_fewshot 5 --model llada_dist --batch_size 8 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.0,is_check_greedy=False,mc_num=128

# accelerate launch eval_llada.py --tasks piqa --num_fewshot 0 --model llada_dist --batch_size 8 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.5,is_check_greedy=False,mc_num=128

# CUDA_VISIBLE_DEVICES=3,5 accelerate launch --main_process_port 29512 eval_llada.py --tasks mmlu_high_school_chemistry --num_fewshot 5 --model llada_dist --batch_size 1 --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',cfg=0.0,is_check_greedy=False,mc_num=1

# accelerate launch eval_llada.py --tasks cmmlu --num_fewshot 5 --model llada_dist --batch_size 1 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.0,is_check_greedy=False,mc_num=1

# accelerate launch eval_llada.py --tasks ceval-valid --num_fewshot 5 --model llada_dist --batch_size 1 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.0,is_check_greedy=False,mc_num=1


# # conditional generation benchmarks
# accelerate launch eval_llada.py --tasks bbh --model llada_dist --model_args model_path='GSAI-ML/LLaDA-8B-Base',gen_length=1024,steps=1024,block_length=1024

# accelerate launch eval_llada.py --tasks gsm8k --model llada_dist --model_args model_path='GSAI-ML/LLaDA-8B-Base',gen_length=1024,steps=1024,block_length=1024

# accelerate launch eval_llada.py --tasks minerva_math --model llada_dist --model_args model_path='GSAI-ML/LLaDA-8B-Base',gen_length=1024,steps=1024,block_length=1024

# accelerate launch eval_llada.py --tasks humaneval --model llada_dist --confirm_run_unsafe_code --model_args model_path='GSAI-ML/LLaDA-8B-Base',gen_length=1024,steps=1024,block_length=1024
# export CUDA_VISIBLE_DEVICES=6,7

# output_path=evals_results/mbpp-ns3-conv/

# CUDA_VISIBLE_DEVICES=1,6,7 accelerate launch --main_process_port 29511 eval_llada.py \
#     --tasks humaneval --model llada_dist \
#     --confirm_run_unsafe_code\
#     --batch_size 1 \
#     --output_path $output_path \
#     --log_samples \
#     --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=512,steps=128,block_length=512,remasking="convolution",cache_reload_step=4,enable_cache=True,


# export CUDA_VISIBLE_DEVICES=6,7

output_path=evals_results/gsm8k-llada-zero-shot-baseline

# CUDA_VISIBLE_DEVICES=2,3,4,7 accelerate launch --main_process_port 29511 eval_llada_original.py \
#     --tasks gsm8k --model llada_dist \
#     --batch_size 1 \
#     --output_path $output_path \
#     --log_samples \
#     --num_fewshot 0 \
#     --limit 100 \
#     --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=256,steps=256,block_length=32,remasking="low_confidence",

CUDA_VISIBLE_DEVICES=1 accelerate launch --main_process_port 29511 eval_llada.py \
    --tasks gsm8k --model llada_dist \
    --batch_size 1 \
    --output_path $output_path \
    --log_samples \
    --num_fewshot 0 \
    --limit 200 \
    --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=256,steps=133,block_length=256,remasking="convolution",cache_reload_step=4,enable_cache=True,


# for cache_reload_step in 8 16; do
#   for far_gap in 32; do
#     output_path="evals_results/mbpp-ns3-ar-future/refresh-${cache_reload_step}-fargap${far_gap}"

#     echo "Running eval with cache_reload_step=${cache_reload_step}, far_gap=${far_gap}"
#     accelerate launch --main_process_port 29511 eval_llada.py \
#       --tasks mbpp \
#       --model llada_dist \
#       --confirm_run_unsafe_code \
#       --batch_size 2 \
#       --output_path "$output_path" \
#       --log_samples \
#       --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=512,steps=512,block_length=512,remasking=ar,far_gap=${far_gap},cache_reload_step=${cache_reload_step},enable_cache=True
#   done
# done