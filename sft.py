from accelerate import Accelerator
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
import os
import json
import pandas as pd
from datasets import Dataset

new_model = "D:\docChatbot\sangbul-sft-e1"
new_model = r"/data/docLLM/sangbul-sft-e1"

# doc 데이터
# with open('/data/doc_sft_data.json', 'r', encoding='utf-8') as f:
with open('./doc_sft_data.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

df = pd.DataFrame(results)
combined_dataset = Dataset.from_pandas(df)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# base_model = "maywell/Synatra-7B-v0.3-dpo"
base_model = "/data/llm/Synatra-7B-v0.3-dpo"
# base_model = "D:\Synatra-7B-v0.3-dpo"

accelerator = Accelerator()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map = {"": accelerator.local_process_index})

cutoff_len = 4096

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
#
# def tokenize_function(examples):
#     return tokenizer(examples['input_text'], truncation=True, padding=True, max_length=cutoff_len)

# 변환된 데이터를 저장할 리스트
results = []

with open('doc_sft_data.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

system_prompt = "주어진 질문과 답변은 강원대학교 학생 상벌에 관한 규정을 기반으로 합니다."
B_INST, E_INST = "[INST]", "[/INST]"
for item in data:
    for qa in item['qa']:
        temp = f"{system_prompt}{B_INST}{qa['question'].strip()}\n{E_INST}{qa['answer'].strip()}"
        results.append({"chat_sample": temp})

df = pd.DataFrame(results)
dataset = Dataset.from_pandas(df)

from peft import prepare_model_for_kbit_training

# Prepare model for k-bit training
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

from peft import LoraConfig, get_peft_model
config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
# print_trainable_parameters(model)

# # Apply the accelerator. You can comment this out to remove the accelerator.
# model = accelerator.prepare_model(model)

# Accelerator prepares model and other components
# model, tokenizer, tokenized_dataset = accelerator.prepare(model, tokenizer, tokenized_dataset)
# model, tokenizer = accelerator.prepare(model, tokenizer)

from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
# import wandb
#
# with open('/data/llm/wandbKey_js.txt', 'r') as file:
#     wandb_key = file.read().strip()
#
# wandb.login(key=wandb_key)
# run = wandb.init(project='Fine tuning mistral 7B civil wage', job_type="training", anonymous="allow")

# model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

training_arguments_c = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    save_steps=10,
    logging_steps=10,
    learning_rate=2e-4,
    weight_decay=0.1,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.3,
    group_by_length=True,
    lr_scheduler_type="constant",
    # report_to="wandb"
)

training_arguments_one_doc = TrainingArguments(
    output_dir="/data/doc_save_steps",
    num_train_epochs=200,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    deepspeed="deepspeed_one_config.json",
    optim="adamw_torch",
    save_steps=50,
    logging_steps=1,
    learning_rate=5e-05,
    weight_decay=0.1,
    fp16=False,
    bf16=False,
    max_grad_norm=1.0,
    max_steps=-1,
    warmup_ratio=0.1,
    remove_unused_columns=False,
    report_to="none"
)


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
#     data_collator=collator,
#     peft_config=peft_config,
#     max_seq_length= None,
#     tokenizer=tokenizer,
#     args=training_arguments_a,
#     packing= False,
# )

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=config,
    max_seq_length=None,
    dataset_text_field="chat_sample",
    tokenizer=tokenizer,
    args=training_arguments_c,
    packing=False,
)
print(trainer.args)

# 학습 전 메모리 정리
torch.cuda.empty_cache()
print("Cleared CUDA cache before training.")

trainer.train()

# 학습 후 메모리 정리
torch.cuda.empty_cache()
print("Cleared CUDA cache after training.")

# Save the fine-tuned model
if hasattr(trainer.model, 'module'):
    original_model = trainer.model.module
else:
    original_model = trainer.model

original_model.save_pretrained(new_model)
# wandb.finish()