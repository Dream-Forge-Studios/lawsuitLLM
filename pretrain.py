from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,DataCollatorForLanguageModeling,TrainingArguments, Trainer
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import os, torch, wandb, platform, warnings
from datasets import load_dataset
import re
import torch
import logging

base_model = "maywell/Synatra-7B-v0.3-dpo"
# base_model = "/data/llm/Synatra-7B-v0.3-dpo"
# base_model = "D:\Synatra-7B-v0.3-dpo"
dataset_name, new_model = "joonhok-exo-ai/korean_law_open_data_precedents", "/data/llm/lawsuit-7B-civil-wage-a"

# Loading a Gath_baize dataset
custom_cache_dir = "/data/huggingface/cache/"
# custom_cache_dir = "D:/huggingface/cache/"

test_case_file = "/data/llm/test_case_numbers.txt"
# test_case_file = r"D:\lawsuitLLM\test_case_numbers.txt"

cutoff_len = 4096


def tokenize_and_prepare_for_clm(prompt, add_eos_token=True):
    # 토큰화 및 필요한 경우 EOS 토큰 추가
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors='pt',
    )

    # CLM을 위한 레이블 생성
    # 입력 시퀀스를 한 위치 오른쪽으로 이동하여 다음 토큰을 예측하도록 설정
    labels = result["input_ids"].roll(shifts=-1, dims=1)

    # 마지막 토큰에 대한 레이블을 -100으로 설정하여 모델이 이를 무시하도록 함
    labels[:, -1] = -100

    result["labels"] = labels

    return result

def generate_and_tokenize_prompt(data_point):
    # 입력("input") 없이 "instruction"만 사용하여 프롬프트 생성
    prompt = data_point["input_text"]

    # 생성된 프롬프트를 토큰화
    tokenized_prompt = tokenize_and_prepare_for_clm(prompt, add_eos_token=True)

    return tokenized_prompt

def format_date(numeric_date):
    # 숫자 형식의 날짜를 문자열로 변환
    str_date = str(numeric_date)

    # YYYY-MM-DD 형식으로 변환
    formatted_date = f"{str_date[:4]}-{str_date[4:6]}-{str_date[6:]}"

    return formatted_date

def preprocess_data(examples):
    # '참조조문'이 None이면 빈 문자열로 처리
    laws = examples['참조조문'] if examples['참조조문'] is not None else ""
    precedents = examples['참조판례'] if examples['참조판례'] is not None else ""
    decision = examples['판시사항'] if examples['판시사항'] is not None else ""
    summary = examples['판결요지'] if examples['판결요지'] is not None else ""
    reason = examples['전문'] if examples['전문'] is not None else ""

    if precedents:
        precedents += ', ' + examples['법원명'] + " " + format_date(examples['선고일자']) + " " + examples['선고'] + " " + examples['사건번호'] + " " + '판결'
    else:
        precedents += examples['법원명'] + " " + format_date(examples['선고일자']) + " " + examples['선고'] + " " + examples[
            '사건번호'] + " " + '판결'

    split_text = re.split("【이\s*유】", examples['전문'], maxsplit=1)
    # 분할된 결과 확인 및 처리
    if len(split_text) > 1:
        reason_text = split_text[1]
    else:
        reason_text = split_text[0]

    # final_text = re.split("대법원\s*(.+?)\(재판장\)|판사\s*(.+?)\(재판장\)|대법원판사\s*(.+?)\(재판장\)|대법관\s*(.+?)\(재판장\)", reason_text, maxsplit=1)
    final_text = re.split("대법원\s+|판사\s+|대법원판사\s+|대법관\s+", reason_text, maxsplit=1)
    final_text = final_text[0]


    combined_text = '판시사항: ' + decision + "\n" + '판결요지: ' + summary + "\n" + '참조조문: ' + laws + '\n' + '참조판례: ' + precedents + '\n' + '이유: ' + final_text

    return {'input_text': combined_text}

# 파일에서 판례정보일련번호 목록 로드
with open(test_case_file, 'r') as f:
    test_case_numbers = [line.strip() for line in f.readlines()]

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
# tokenizer.padding_side = 'right'
# tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
tokenizer.add_bos_token, tokenizer.add_eos_token


dataset = load_dataset(dataset_name, cache_dir=custom_cache_dir, split="train")

# '민사' 사건 중 '임금'만 포함된 데이터 필터링하면서 테스트 케이스 제외
civil_cases_with_wage_excluded = dataset.filter(
    lambda x: x['사건종류명'] == '민사' and
              x['사건명'] is not None and
              '임금' in x['사건명'] and
              x['판례정보일련번호'] not in test_case_numbers
)

# 원본 데이터셋에 전처리 함수 적용
processed_dataset = civil_cases_with_wage_excluded.map(preprocess_data)

# 원본 데이터셋의 다른 열을 제거하고 'input_text'만 남깁니다.
final_dataset = processed_dataset.remove_columns([column_name for column_name in processed_dataset.column_names if column_name != 'input_text'])

train_data = final_dataset.map(generate_and_tokenize_prompt)

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
        train_dataset=train_data,
        eval_dataset=None,
        args=training_arguments_a,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer, mlm=False,  pad_to_multiple_of=8, return_tensors="pt"
        ),
    )

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