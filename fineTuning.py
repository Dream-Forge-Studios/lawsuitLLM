from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,HfArgumentParser,TrainingArguments,pipeline, logging, TextStreamer
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import os, torch, wandb, platform, warnings
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from huggingface_hub import notebook_login


base_model = "maywell/Synatra-7B-v0.3-dpo"
dataset_name, new_model = "jiwoochris/easylaw_kr", "tyflow/lawsuit-7B-easylaw_kr"

# Loading a Gath_baize dataset
dataset = load_dataset(dataset_name, split="train")

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# Load base model(Mistral 7B)
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
model.gradient_checkpointing_enable()

# 그래디언트 체크포인팅 활성화
model.gradient_checkpointing_enable()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
tokenizer.add_bos_token, tokenizer.add_eos_token

with open('/data/wandbKey_js.txt', 'r') as file:
    wandb_key = file.read().strip()

wandb.login(key = wandb_key)
run = wandb.init(project='Fine tuning mistral 7B easylaw_kr', job_type="training", anonymous="allow")

model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]
    )
model = get_peft_model(model, peft_config)

# Training Arguments
# Hyperparameters should beadjusted based on the hardware you using
training_arguments = TrainingArguments(
    output_dir= "./results",
    num_train_epochs= 1,
    per_device_train_batch_size= 8,
    gradient_accumulation_steps= 2,
    optim = "paged_adamw_8bit",
    save_steps= 5000,
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

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        text = f"user{example['instruction'][i]}<lim_end|> {example['output'][i]}</s><lim_end|>"
        output_texts.append(text)
    return output_texts



response_template = "</s><lim_end|>"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

# Setting sft parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    peft_config=peft_config,
    max_seq_length= None,
    tokenizer=tokenizer,
    args=training_arguments,
    packing= False,
)

trainer.train()
# Save the fine-tuned model
trainer.model.save_pretrained(new_model)
wandb.finish()
model.config.use_cache = True
model.eval()