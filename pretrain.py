from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,DataCollatorForLanguageModeling,TrainingArguments, Trainer
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import os, torch, wandb, platform, warnings
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from huggingface_hub import notebook_login


base_model = "/data/llm/Synatra-7B-v0.3-dpo"
dataset_name, new_model = "joonhok-exo-ai/korean_law_open_data_precedents", "/data/llm/lawsuit-7B-civil-wage-a"

# Loading a Gath_baize dataset
dataset = load_dataset(dataset_name, split="train")

# '민사' 사건 중 '임금'만 포함된 데이터 필터링
civil_cases_with_wage = dataset.filter(lambda x: x['사건종류명'] == '민사' and '임금' in x['사건내용'])

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

# 그래디언트 체크포인팅 활성화
# model.gradient_checkpointing_enable()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
tokenizer.add_bos_token, tokenizer.add_eos_token

with open('/data/llm/wandbKey_js.txt', 'r') as file:
    wandb_key = file.read().strip()

wandb.login(key = wandb_key)
run = wandb.init(project='Fine tuning mistral 7B civil wage', job_type="training", anonymous="allow")

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
training_arguments_a = TrainingArguments(
    output_dir= "./results",
    num_train_epochs= 1,
    per_device_train_batch_size= 4,
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
    per_device_train_batch_size= 4,
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
        train_dataset=civil_cases_with_wage,
        eval_dataset=None,
        args=training_arguments_a,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer, mlm=False,  pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

trainer.train()
model.config.use_cache = False
model.print_trainable_parameters() # 훈련하는 파라미터의 % 체크
# Save the fine-tuned model
trainer.model.save_pretrained(new_model)
wandb.finish()