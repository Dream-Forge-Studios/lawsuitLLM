# -*- coding: utf-8 -*-

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,DataCollatorForLanguageModeling,TrainingArguments, Trainer
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import os, torch, wandb, platform, warnings
from datasets import load_dataset, concatenate_datasets
import re
import torch
import logging
from utils import ko_wikidata_QA, hugging_precedents, korean_textbooks, ai_hub_precedents, law_qa_datas, law_translate_datas
from collections import Counter
import pandas as pd
from datasets import Dataset

# base_model = "maywell/Synatra-7B-v0.3-dpo"
base_model = "/data/llm/Synatra-7B-v0.3-dpo"
# base_model = "D:\Synatra-7B-v0.3-dpo"
dataset_name, new_model = "joonhok-exo-ai/korean_law_open_data_precedents", "/data/llm/lawsuit-7B-pretain"
dataset_name2 = 'maywell/korean_textbooks'
# dataset_name2 = 'maywell/ko_wikidata_QA'

# Loading a Gath_baize dataset
custom_cache_dir = "/data/huggingface/cache/"
# custom_cache_dir = "D:/huggingface/cache/"

test_case_file = "/data/llm/not_with_wage_case_numbers_300.txt"
# test_case_file = r"D:\test_case_numbers.txt"

cutoff_len = 4096


# 파일에서 판례정보일련번호 목록 로드
# with open(test_case_file, 'r') as f:
#     test_case_numbers = [line.strip() for line in f.readlines()]

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

bnb_config = BitsAndBytesConfig(
    load_in_4bit= True,
    bnb_4bit_quant_type= "nf4",
    bnb_4bit_compute_dtype= torch.bfloat16,
    bnb_4bit_use_double_quant= False,
)


model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto"
)
model.config.use_cache = False # silence the warnings. Please re-enable for inference!
model.config.pretraining_tp = 1

# 그래디언트 체크포인팅 활성화
# model.gradient_checkpointing_enable()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
# tokenizer.padding_side = 'right'
# tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
tokenizer.add_bos_token, tokenizer.add_eos_token



processed_dataset = hugging_precedents()
processed_dataset = processed_dataset.remove_columns([column_name for column_name in processed_dataset.column_names if column_name != 'input_text'])
ai_hub_precedents_dataset = ai_hub_precedents()
law_qa_dataset = law_qa_datas()
law_translate_dataset = law_translate_datas()

# qa_dataset = ko_wikidata_QA(300)
textbooks_dataset = korean_textbooks(945, 'tiny-textbooks')

# 48168개
combined_dataset = concatenate_datasets([processed_dataset, ai_hub_precedents_dataset, law_qa_dataset, law_translate_dataset, textbooks_dataset]).shuffle()

# 원본 데이터셋의 다른 열을 제거하고 'input_text'만 남깁니다.
# final_dataset = processed_dataset.remove_columns([column_name for column_name in processed_dataset.column_names if column_name != 'input_text'])
# final_dataset = combined_dataset.remove_columns([column_name for column_name in combined_dataset.column_names if column_name != 'input_text'])
# 데이터셋 토큰화 함수
def tokenize_function(examples):
    return tokenizer(examples['input_text'], truncation=True, padding=True, max_length=cutoff_len)

# 데이터셋 토큰화 적용
tokenized_dataset = combined_dataset.map(tokenize_function, batched=True)

with open('/data/llm/wandbKey_js.txt', 'r') as file:
    wandb_key = file.read().strip()

wandb.login(key = wandb_key)
run = wandb.init(project='Fine tuning mistral 7B civil wage', job_type="training", anonymous="allow")

model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]
    )
model = get_peft_model(model, peft_config)

# Training Arguments
# Hyperparameters should beadjusted based on the hardware you using
training_arguments_llama2 = TrainingArguments(
    output_dir= "./results",
    num_train_epochs= 1,
    per_device_train_batch_size= 1,
    gradient_accumulation_steps= 4,
    optim = "paged_adamw_32bit",
    save_steps= 30,
    logging_steps= 30,
    learning_rate= 2e-4,
    weight_decay= 0.1,
    fp16= False,
    bf16= False,
    max_grad_norm= 1,
    max_steps= -1,
    warmup_ratio= 0.1,
    group_by_length= True,
    lr_scheduler_type= "cosine",
    report_to="wandb"
)

# training_arguments_deepspeed = TrainingArguments(
#     output_dir="./results",
#     num_train_epochs=1,
#     # per_device_train_batch_size=2,  # DeepSpeed 설정은 전체 배치 크기를 결정하는데 사용되므로, 여기서는 GPU당 배치 크기만 지정합니다.
#     # gradient_accumulation_steps=8,  # 이 값은 DeepSpeed 설정 파일에 명시된 값과 호환되어야 합니다.
#     # optim="adamw_torch",  # DeepSpeed를 사용할 때는 이 옵션 대신 DeepSpeed의 설정을 사용합니다.
#     save_steps=1500,
#     logging_steps=250,
#     learning_rate=2e-4,
#     weight_decay=0.001,
#     # fp16=False,  # DeepSpeed 설정 파일에서 관리합니다.
#     # bf16=False,  # DeepSpeed 설정 파일에서 관리합니다.
#     max_grad_norm=0.3,
#     # warmup_ratio=0.3,  # DeepSpeed 설정 파일에서 관리합니다.
#     group_by_length=True,
#     # lr_scheduler_type="cosine",  # DeepSpeed 설정 파일에서 관리합니다.
#     report_to="wandb",
#     deepspeed="deepspeed_config.json"  # DeepSpeed 설정 파일 참조
# )
training_arguments_c = TrainingArguments(
    output_dir= "./results",
    num_train_epochs= 1,
    per_device_train_batch_size= 1,
    gradient_accumulation_steps= 4,
    optim = "paged_adamw_32bit",
    save_steps= 1500,
    logging_dir="./logs",
    # logging_steps= 250,
    logging_steps= 1,
    learning_rate= 2e-4,
    weight_decay= 0.001,
    fp16= False,
    bf16= False,
    max_grad_norm= 0.3,
    max_steps= -1,
    warmup_ratio= 0.3,
    group_by_length= True,
    lr_scheduler_type= "cosine",
    report_to="wandb"
)

training_arguments_a_cosine = TrainingArguments(
    output_dir= "./results",
    num_train_epochs= 1,
    per_device_train_batch_size= 1,
    gradient_accumulation_steps= 4,
    optim = "paged_adamw_8bit",
    save_steps= 30,
    logging_steps= 30,
    learning_rate= 2e-4,
    weight_decay= 0.001,
    fp16= False,
    bf16= False,
    max_grad_norm= 0.3,
    max_steps= -1,
    warmup_ratio= 0.3,
    group_by_length= True,
    lr_scheduler_type= "cosine",
    report_to="wandb"
)

training_arguments_a_32bit = TrainingArguments(
    output_dir= "./results",
    num_train_epochs= 1,
    per_device_train_batch_size= 1,
    gradient_accumulation_steps= 4,
    optim = "paged_adamw_32bit",
    save_steps= 30,
    logging_steps= 30,
    learning_rate= 2e-4,
    weight_decay= 0.001,
    fp16= False,
    bf16= False,
    max_grad_norm= 0.3,
    max_steps= -1,
    warmup_ratio= 0.3,
    group_by_length= True,
    lr_scheduler_type= "constant",
    report_to="wandb"
)

training_arguments_a = TrainingArguments(
    output_dir= "./results",
    num_train_epochs= 1,
    per_device_train_batch_size= 1,
    gradient_accumulation_steps= 4,
    optim = "paged_adamw_8bit",
    save_steps= 30,
    logging_steps= 30,
    learning_rate= 2e-4,
    weight_decay= 0.001,
    fp16= False,
    bf16= False,
    max_grad_norm= 0.3,
    max_steps= -1,
    warmup_ratio= 0.3,
    group_by_length= True,
    lr_scheduler_type= "constant",
    report_to="wandb"
)

training_arguments_f = TrainingArguments(
    output_dir= "./results",
    num_train_epochs= 1,
    per_device_train_batch_size= 1,
    gradient_accumulation_steps= 4,
    optim = "adamw_torch",
    save_steps= 30,
    logging_steps= 30,
    learning_rate= 4e-4,
    weight_decay= 0.01,
    fp16= False,
    bf16= False,
    max_grad_norm= 1.0,
    max_steps= -1,
    warmup_ratio= 0.06,
    group_by_length= False,
    lr_scheduler_type= "cosine",
    load_best_model_at_end=False,
    ddp_find_unused_parameters=False,
    save_total_limit=5,
    report_to="wandb"
)

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        text = f"<s>[INST]{example['instruction'][i]}[/INST] {example['output'][i]}</s>"
        output_texts.append(text)
    return output_texts



# 데이터 콜레이터 인스턴스 생성
# collator = DataCollatorForCompletionOnlyLM(
#     tokenizer=tokenizer,
#     mlm=False,
#     instruction_template="<s>[INST]",
#     response_template="[/INST]",
#     pad_to_multiple_of=8  # 필요한 경우 패딩을 특정 배수로 맞춤
# )

# Setting sft parameters
# trainer = SFTTrainer(
#     model=model,
#     train_dataset=dataset,
#     formatting_func=formatting_prompts_func,
#     # data_collator=collator,
#     peft_config=peft_config,
#     max_seq_length= None,
#     tokenizer=tokenizer,
#     args=training_arguments_a,
#     packing= False,
# )

trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset,
        eval_dataset=None,
        args=training_arguments_c,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer, mlm=False,  pad_to_multiple_of=8, return_tensors="pt"
        ),
    )

print(trainer.args)

# 로거 설정
logger = logging.getLogger(__name__)

# 사용 가능한 GPU 개수 확인
num_gpus = torch.cuda.device_count()

# 사용 가능한 GPU 개수 로깅
print(f"Using {num_gpus} GPUs for training.")

trainer.train()
model.config.use_cache = False
model.print_trainable_parameters() # 훈련하는 파라미터의 % 체크
# Save the fine-tuned model
trainer.model.save_pretrained(new_model)
wandb.finish()