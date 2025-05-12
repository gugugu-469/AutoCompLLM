import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import jsonlines
from vllm import LLM, SamplingParams
import json
import re
import random
import gc
from collections import defaultdict
from vllm.distributed.parallel_state import destroy_model_parallel
import argparse
from itertools import combinations

inp_prompt = '''You are currently a senior expert in commonsense Q&A.
Your task is to choose the correct answer option based on the given question and five options. The input format of option is: "option number. option content".
The output format of the task is: correct option number.
Given question: "{question}"
Given options: {option}
'''

yes_no_prompt = '''You are currently a senior expert in commonsense true-or-false questions.
Your task is to determine whether a given question and candidate answer are correct.
The output format of the task is: Yes or No.
Given question: "{question}"
Candidate answer: "{answer}"
'''

fewer_option_prompt = '''You are currently a senior expert in commonsense Q&A.
Your task is to choose the correct answer option based on the given question and {op_nums} options. The input format of option is: "option number. option content".
The output format of the task is: correct option number.
Given question: "{question}"
Given options: {option}
'''

with jsonlines.open('xx/ori/CommonsenseQA/dev_rand_split.jsonl', 'r') as f:
    dev_datas = [data for data in f]


def get_apply_prompts(tokenizer, prompt_list):
    processed_prompt_list = []
    for prompt in prompt_list:
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking = False,
            use_system_prompt = False
        )
        processed_prompt_list.append(text)
    return processed_prompt_list

def vllm_chat(tokenizer, model, prompt_list, sampling_params):
    outputs = model.generate(prompt_list, sampling_params)

    response_list = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        response_list.append(generated_text)
    return response_list


# 基本的输出路径
base_out_dir = './pred_result'
tp_size = torch.cuda.device_count()

def main():
    parser = argparse.ArgumentParser(description='这是一个文件处理程序')
    
    # 添加命令行参数
    parser.add_argument('--model_path', type=str, help='模型路径')
    parser.add_argument('--temperature', type=float,help='温度')
    
    # 解析命令行参数
    args = parser.parse_args()
    model_path = args.model_path
    temperature = args.temperature

    if 'checkpoint' in model_path:
        model_name = os.path.basename(model_path)
        model_name_2 = os.path.basename(os.path.dirname(model_path))
        model_name_3 = os.path.basename(os.path.dirname(os.path.dirname(model_path)))
        now_out_dir = os.path.join(base_out_dir, '{}_{}_{}_temp_{}'.format(model_name_3, model_name_2, model_name, temperature))
    else:
        model_name = os.path.basename(model_path)
        model_name_2 = os.path.basename(os.path.dirname(model_path))
        now_out_dir = os.path.join(base_out_dir, '{}_{}_temp_{}'.format(model_name_2, model_name, temperature))
    print('model_path:{}'.format(model_path))
    print('temperature:{}'.format(temperature))
    print('model_name:{}'.format(model_name))
    print('now_out_dir:{}'.format(now_out_dir))
    if not os.path.exists(now_out_dir):
        os.makedirs(now_out_dir)

    sampling_params = SamplingParams(temperature=temperature, top_p=0.95, max_tokens=1000)
    model = LLM(
        model_path,
        tensor_parallel_size = tp_size,
        gpu_memory_utilization = 0.85,
        max_model_len = 2000,
        trust_remote_code = True
        )
    tokenizer = model.get_tokenizer()


    # data_type_list = ['dev', 'test']
    # use_datas_list = [dev_datas, test_datas]
    data_type_list = ['dev']
    use_datas_list = [dev_datas]
    overall_prompt_list = [
        ('c_to_hrt', inp_prompt),
        ('all_fewer_option', fewer_option_prompt),
        ('yes_no', yes_no_prompt),
        ('fewer_option', fewer_option_prompt),
    ]
    ori_task = ['yes_no']
    for data_type, use_datas in zip(data_type_list, use_datas_list):
        for prompt_name, prompt_format in overall_prompt_list:
            out_name = '{}_{}_Commonsense.jsonl'.format(prompt_name, data_type)
            prompt_list = []
            if prompt_name == 'c_to_hrt':
                for data in use_datas:
                    ans = data['answerKey']
                    data = data['question']
                    question = data['stem']
                    option = ''
                    for i,choice in enumerate(data['choices']):
                        label = choice['label']
                        text = choice['text']
                        assert chr(ord('A')+i) == label
                        option += '{}. {};\t'.format(label,text)
                    option = option.strip()
                    prompt = prompt_format.format(
                        question = question, option = option
                    )
                    prompt_list.append(prompt)
            elif prompt_name == 'yes_no':
                option_list = []
                text_list = []
                text_dict = defaultdict(list)
                overall_index = 0
                for index,data in enumerate(use_datas):
                    ans = data['answerKey']
                    data = data['question']
                    question = data['stem']
                    for choice in data['choices']:
                        label = choice['label']
                        text = choice['text']
                        prompt = prompt_format.format(question = question, answer = text)
                        prompt_list.append(prompt)
                        option_list.append(text)
                        text_list.append(question)
                        text_dict[question].append(overall_index)
                        overall_index += 1
            elif prompt_name == 'fewer_option':
                all_options = []
                text_list = []
                read_file = os.path.join(now_out_dir, '{}_{}_Commonsense.jsonl'.format('yes_no', data_type))
                with jsonlines.open(read_file, 'r') as f:
                    read_datas = [data for data in f]
                for data in read_datas:
                    if data['is_finish']:
                        continue
                    question = data['text']
                    options = data['options']
                    option = ''
                    for i,choice in enumerate(options):
                        label = chr(ord('A')+i)
                        option += '{}. {};\t'.format(label,choice)
                    option = option.strip()
                    prompt = prompt_format.format(question = question, option = option, op_nums = data['op_num'])
                    prompt_list.append(prompt)
                    all_options.append(options)
                    text_list.append(question)
            elif prompt_name == 'all_fewer_option':
                op_nums = ['one', 'two', 'three', 'four', 'five']
                all_options = []
                text_list = []
                for data in use_datas:
                    ans = data['answerKey']
                    data = data['question']
                    question = data['stem']
                    
                    # Get all choices first
                    choices = []
                    for i, choice in enumerate(data['choices']):
                        label = choice['label']
                        text = choice['text']
                        assert chr(ord('A')+i) == label
                        choices.append(text)
                    
                    # Generate combinations for all lengths from 2 to number of choices
                    for r in range(2, len(choices)+1):
                        for combo_indices in combinations(range(len(choices)), r):
                            option = ''
                            options = []
                            # Check if the correct answer is in this combination
                            correct_label_present = False
                            label_idx = 0
                            for idx in combo_indices:
                                options.append(choices[idx])
                                text = choices[idx]
                                label = chr(ord('A')+label_idx)
                                label_idx += 1
                                option += '{}. {};\t'.format(label, text)

                            option = option.strip()
                            prompt = prompt_format.format(
                                question = question, 
                                option = option,
                                op_nums = op_nums[len(options)-1]
                            )
                            prompt_list.append(prompt)
                            all_options.append(options)
                            text_list.append(question)

            else:
                raise ValueError('prompt name error:{}'.format(prompt_name))

            prompt_list = get_apply_prompts(tokenizer, prompt_list)
            res_list = vllm_chat(tokenizer, model, prompt_list, sampling_params)
            if prompt_name == 'c_to_hrt':
                out_datas = []
                for prompt, pred in zip(prompt_list, res_list):
                    out_datas.append({
                        'prompt': prompt,
                        'output': pred
                    })
            elif prompt_name == 'all_fewer_option':
                out_datas = []
                for prompt, pred, options, text in zip(prompt_list, res_list, all_options, text_list):
                    out_datas.append({
                        'prompt': prompt,
                        'text': text,
                        'options': options,
                        'output': pred
                    })
            elif prompt_name == 'fewer_option':
                out_datas = []
                for text, prompt, pred,options in zip(text_list, prompt_list, res_list, all_options):
                    out_datas.append({
                        'text': text,
                        'options': options,
                        'prompt': prompt,
                        'output': pred,
                    })
                
            elif prompt_name == 'yes_no':
                out_datas = []
                for text, prompt, pred, option in zip(text_list, prompt_list, res_list, option_list):
                    out_datas.append({
                        'text': text,
                        'option': option,
                        'prompt': prompt,
                        'output': pred
                    })
                new_out_name = '{}_{}_ori_Commonsense.jsonl'.format(prompt_name, data_type)
                with jsonlines.open(os.path.join(now_out_dir, new_out_name), 'w') as f:
                    for data in out_datas:
                        f.write(data)
                out_datas = []
                for question in text_dict.keys():
                    index_list = text_dict[question]
                    text = question
                    options = []
                    for index in index_list:
                        if res_list[index].strip().lower() == 'yes':
                            options.append(option_list[index])
                    if len(options) == 0:
                        for index in index_list:
                            options.append(option_list[index])
                    op_nums = ['one', 'two', 'three', 'four', 'five']
                    out_datas.append({
                        'text': text,
                        'options': options,
                        'op_num': op_nums[len(options)-1],
                        'is_finish': len(options) == 1
                    })
            else:
                raise ValueError('prompt name error:{}'.format(prompt_name))
            with jsonlines.open(os.path.join(now_out_dir, out_name), 'w') as f:
                for data in out_datas:
                    f.write(data)


if __name__ == '__main__':
    main()