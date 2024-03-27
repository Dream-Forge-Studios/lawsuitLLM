from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig

new_model = "/data/llm/lawsuit-7B-pretain-r8"

# accelerator
fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
)
accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

from utils import hugging_precedents, korean_textbooks, ai_hub_precedents, law_qa_datas, law_translate_datas
from datasets import concatenate_datasets

#dataset
processed_dataset = hugging_precedents()
processed_dataset = processed_dataset.remove_columns(
    [column_name for column_name in processed_dataset.column_names if column_name != 'input_text'])
ai_hub_precedents_dataset = ai_hub_precedents()
law_qa_dataset = law_qa_datas()
law_translate_dataset = law_translate_datas()

# textbooks_dataset = korean_textbooks(945, 'tiny-textbooks')
textbooks_dataset = korean_textbooks(200, 'tiny-textbooks')
textbooks_dataset = textbooks_dataset.remove_columns(
    [column_name for column_name in textbooks_dataset.column_names if column_name != 'input_text'])

# 48168개
combined_dataset = concatenate_datasets(
    [processed_dataset, ai_hub_precedents_dataset, law_qa_dataset, law_translate_dataset, textbooks_dataset]).shuffle()
combined_dataset = combined_dataset.select(range(10000))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# base_model = "maywell/Synatra-7B-v0.3-dpo"
base_model = "/data/llm/Synatra-7B-v0.3-dpo"
# base_model = "D:\Synatra-7B-v0.3-dpo"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb_config)

tokenizer = AutoTokenizer.from_pretrained(
    base_model,
    model_max_length=512,
    padding_side="left",
    add_eos_token=True)
tokenizer.pad_token = tokenizer.eos_token

cutoff_len = 4096

def tokenize_function(examples):
    return tokenizer(examples['input_text'], truncation=True, padding="max_length", max_length=cutoff_len)

# 데이터셋 토큰화 적용
tokenized_dataset = combined_dataset.map(tokenize_function, batched=True)

from peft import prepare_model_for_kbit_training

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
model = get_peft_model(model, config)
print_trainable_parameters(model)
# Apply the accelerator. You can comment this out to remove the accelerator.
model = accelerator.prepare_model(model)

if torch.cuda.device_count() > 1: # If more than 1 GPU
    model.is_parallelizable = True
    model.model_parallel = True

from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
import wandb

with open('/data/llm/wandbKey_js.txt', 'r') as file:
    wandb_key = file.read().strip()

wandb.login(key=wandb_key)
run = wandb.init(project='Fine tuning mistral 7B civil wage', job_type="training", anonymous="allow")

training_arguments_c = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    save_steps=1500,
    logging_dir="./logs",
    logging_steps= 250,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,
    bf16=True,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.3,
    group_by_length=True,
    lr_scheduler_type="cosine",
    report_to="wandb"
)

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

print(trainer.args)

# 사용 가능한 GPU 개수 확인
num_gpus = torch.cuda.device_count()

# 사용 가능한 GPU 개수 로깅
print(f"Using {num_gpus} GPUs for training.")

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

# Save the fine-tuned model
trainer.model.save_pretrained(new_model)
wandb.finish()