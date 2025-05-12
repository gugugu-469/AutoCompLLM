import logging
import os
from dataclasses import dataclass
import os
import json
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from datetime import datetime
import logging
import random
import re 
import jsonlines
import torch
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer, get_peft_config, ModelConfig, TrlParser
from peft import LoraConfig, get_peft_model

print('devices:{}'.format(torch.cuda.device_count()))
########################
# Custom dataclasses
########################
@dataclass
class ScriptArguments:
    dataset_id_or_path: str = "Jiayi-Pan/Countdown-Tasks-3to4"
    dataset_splits: str = "train"
    tokenizer_name_or_path: str = None


########################
# Setup logging
########################
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)

########################
# Helper functions
########################
import re
with open('/HL_user01/2025_05_02_cmeie_nips/53_schemas.json', 'r') as f:
    schemas = json.load(f)
direction_schemas = [(item['subject_type'], item['predicate'], item['object_type']) for item in schemas if item['direction'] == '0']
def format_reward(completions, target, ext_info, **kwargs):
    """
    检查输出是否符合指定的Markdown代码块格式要求
    
    参数:
        output (str): 模型输出的字符串
        
    返回:
        float: 格式正确的行数占总行数的比例[\(（]不包括代码块标记[\)）]
    """
    rewards = []
    print('completions:{}'.format(completions))
    print('target:{}'.format(target))
    print('ext_info:{}'.format(ext_info))
    for completion, gt, ext in zip(completions, target, ext_info): 
        output = completion.strip()
        
        task = ext['task']
        if task == 'options_5':
            options = ['A','B','C','D','E']
        elif task == 'options_4':
            options = ['A','B','C','D']
        elif task == 'options_3':
            options = ['A','B','C']
        elif task == 'options_2':
            options = ['A','B']
        elif task == 'yes_no':
            options = ['Yes','No']
        else:
            raise ValueError('task error:{}'.format(task))
        
        if output.strip() in options:
            rewards.append(1.0) 
        else:
            rewards.append(0.0) 

    print('format reward:{}'.format(rewards))
    return rewards

def answer_reward(completions, target, ext_info,**kwargs):
    rewards = []
    for completion, gt, ext in zip(completions, target, ext_info): 
        rewards.append(float(completion.strip() == gt.strip()))
    print('answer reward:{}'.format(rewards))
    return rewards

def get_checkpoint(training_args: GRPOConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint


def grpo_function(
    model_args: ModelConfig, script_args: ScriptArguments, training_args: GRPOConfig
):
    #########################
    # Log parameters
    #########################
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    ################
    # Load tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        (
            script_args.tokenizer_name_or_path
            if script_args.tokenizer_name_or_path
            else model_args.model_name_or_path
        ),
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print('dataset:{}'.format(script_args.dataset_id_or_path))
    ###############
    # Load datasets
    ###############
    # Load dataset from Hugging Face Hub
    with jsonlines.open(script_args.dataset_id_or_path, 'r') as f:
        datas = [data for data in f]
    dataset = Dataset.from_list(datas)

    #####################
    # Prepare and format dataset
    #####################

    # gemerate r1 prompt with a prefix for the model to already start with the thinking process
    def generate_r1_prompt(ins, output):
        r1_prefix = [{
            "role": "user",
            "content": ins
          }]
        return {"prompt": tokenizer.apply_chat_template(r1_prefix, tokenize=False, add_generation_prompt=True, enable_thinking=False, use_system_prompt=False),'target': output}

    # convert our dataset to the r1 prompt
    dataset = dataset.map(lambda x: generate_r1_prompt(x["instruction"], x['output']))
    print('dataset sample:{}'.format(dataset[0]))
    # print('decode dataset sample:{}'.format(tokenizer.decode(dataset[0])))

    # split the dataset into train and test
    train_test_split = dataset.train_test_split(test_size=0.1)
    train_dataset = dataset
    # train_dataset = train_test_split["train"]
    # test_dataset = train_test_split["test"]

    #########################
    # Instantiate DPO trainer
    #########################

    trainer = GRPOTrainer(
      model=model_args.model_name_or_path,
      reward_funcs=[format_reward, answer_reward],
      args=training_args,
      train_dataset=train_dataset,
    #   eval_dataset=test_dataset,
      peft_config=get_peft_config(model_args),
    )


    ###############
    # Training loop
    ###############
    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    # Train the model
    logger.info(
        f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {training_args.num_train_epochs} epochs***'
    )
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    # Log and save metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################

    logger.info("*** Save model ***")
    trainer.model.config.use_cache = True
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")
    training_args.distributed_state.wait_for_everyone()  # wait for all processes to load

    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"Tokenizer saved to {training_args.output_dir}")

    # Save everything else on main process
    if trainer.accelerator.is_main_process:
        trainer.create_model_card({"tags": ["rl","grpo", "tutorial", "philschmid"]})
    # push to hub if needed
    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub()

    logger.info("*** Training complete! ***")


def main():
    parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()
    model_args.trust_remote_code = True
    # Run the main training loop
    grpo_function(model_args, script_args, training_args)


if __name__ == "__main__":
    main()