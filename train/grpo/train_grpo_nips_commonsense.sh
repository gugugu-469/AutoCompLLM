
CUDA_VISIBLE_DEVICES=1,0 accelerate launch --main_process_port 24500 --num_processes 2 --config_file configs/accelerate_configs/deepspeed_zero2.yaml scripts/run_r1_grpo_ruijin_commonsense.py --config receipes/grpo-chatglm3-deepseek-r1-countdown_ruijin_commonsense_qwen3_nochaijie.yaml

CUDA_VISIBLE_DEVICES=1,0 accelerate launch --main_process_port 24500 --num_processes 2 --config_file configs/accelerate_configs/deepspeed_zero2.yaml scripts/run_r1_grpo_ruijin_commonsense.py --config receipes/grpo-chatglm3-deepseek-r1-countdown_ruijin_commonsense_qwen3_subtask.yaml

CUDA_VISIBLE_DEVICES=1,0 accelerate launch --main_process_port 24500 --num_processes 2 --config_file configs/accelerate_configs/deepspeed_zero2.yaml scripts/run_r1_grpo_ruijin_commonsense.py --config receipes/grpo-chatglm3-deepseek-r1-countdown_ruijin_commonsense_llama3_nochaijie.yaml

CUDA_VISIBLE_DEVICES=1,0 accelerate launch --main_process_port 24500 --num_processes 2 --config_file configs/accelerate_configs/deepspeed_zero2.yaml scripts/run_r1_grpo_ruijin_commonsense.py --config receipes/grpo-chatglm3-deepseek-r1-countdown_ruijin_commonsense_llama3_subtask.yaml

CUDA_VISIBLE_DEVICES=1,0 accelerate launch --main_process_port 24500 --num_processes 2 --config_file configs/accelerate_configs/deepspeed_zero2.yaml scripts/run_r1_grpo_ruijin_commonsense.py --config receipes/grpo-chatglm3-deepseek-r1-countdown_ruijin_commonsense_intern3_nochaijie.yaml

CUDA_VISIBLE_DEVICES=1,0 accelerate launch --main_process_port 24500 --num_processes 2 --config_file configs/accelerate_configs/deepspeed_zero2.yaml scripts/run_r1_grpo_ruijin_commonsense.py --config receipes/grpo-chatglm3-deepseek-r1-countdown_ruijin_commonsense_intern3_subtask.yaml

CUDA_VISIBLE_DEVICES=1,0 accelerate launch --main_process_port 24500 --num_processes 2 --config_file configs/accelerate_configs/deepspeed_zero2.yaml scripts/run_r1_grpo_ruijin_commonsense.py --config receipes/grpo-chatglm3-deepseek-r1-countdown_ruijin_commonsense_glm_nochaijie.yaml

CUDA_VISIBLE_DEVICES=1,0 accelerate launch --main_process_port 24500 --num_processes 2 --config_file configs/accelerate_configs/deepspeed_zero2.yaml scripts/run_r1_grpo_ruijin_commonsense.py --config receipes/grpo-chatglm3-deepseek-r1-countdown_ruijin_commonsense_glm_subtask.yaml





