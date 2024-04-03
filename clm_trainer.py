from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling

base_model = "/data/llm/Synatra-7B-v0.3-dpo"
# base_model = "D:\Synatra-7B-v0.3-dpo"

new_model = "/data/llm/lawsuit-7B-pretain-r8"

cutoff_len = 4096

tqdm.pandas()

# Step 1: Load the dataset
tokenizer = AutoTokenizer.from_pretrained(base_model)

from utils import hugging_precedents, korean_textbooks, ai_hub_precedents, law_qa_datas, law_translate_datas
from datasets import concatenate_datasets

#dataset
processed_dataset = hugging_precedents()
processed_dataset = processed_dataset.remove_columns(
    [column_name for column_name in processed_dataset.column_names if column_name != 'input_text'])
ai_hub_precedents_dataset = ai_hub_precedents()
law_qa_dataset = law_qa_datas()
law_translate_dataset = law_translate_datas()

textbooks_dataset = korean_textbooks(945, 'tiny-textbooks')
# textbooks_dataset = korean_textbooks(200, 'tiny-textbooks')
textbooks_dataset = textbooks_dataset.remove_columns(
    [column_name for column_name in textbooks_dataset.column_names if column_name != 'input_text'])

# 48168개
combined_dataset = concatenate_datasets(
    [processed_dataset, ai_hub_precedents_dataset, law_qa_dataset, law_translate_dataset, textbooks_dataset]).shuffle()
combined_dataset = combined_dataset.select(range(10000))
def tokenize_function(examples):
    return tokenizer(examples['input_text'], truncation=True, padding="max_length", max_length=cutoff_len)

tokenized_dataset = combined_dataset.map(tokenize_function, num_proc=4)

# Step 2: Load the model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
# Copy the model to each device
device_map = {"": Accelerator().local_process_index}
torch_dtype = torch.bfloat16

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map=device_map,
    trust_remote_code=True,
    torch_dtype=torch_dtype,
)


# Step 3: Define the training arguments
training_arguments_c = TrainingArguments(
    output_dir="/data/save_steps",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    deepspeed="deepspeed_config.json",
    save_steps=50,
    logging_dir="./logs",
    logging_steps= 10,
    learning_rate=1e-05,
    weight_decay=0.1,
    fp16=True,
    bf16=False,
    max_grad_norm=1,
    max_steps=-1,
    warmup_ratio=0.1,
    report_to="wandb"
)


# Step 4: Define the LoraConfig
peft_config = LoraConfig(
    r=8,
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


# Step 5: Define the Trainer
data_collator = DataCollatorForLanguageModeling(
    tokenizer, mlm=False, pad_to_multiple_of=8, return_tensors="pt"
)

trainer = Trainer(
    model=model,
    args=training_arguments_c,
    train_dataset=tokenized_dataset,
    eval_dataset=None,
    tokenizer=tokenizer,
    data_collator=data_collator,
)
trainer.train()

# Step 6: Save the model
trainer.save_model(new_model)