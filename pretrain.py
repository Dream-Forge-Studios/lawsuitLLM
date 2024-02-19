from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,DataCollatorForLanguageModeling,TrainingArguments, Trainer
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import os, torch, wandb, platform, warnings
from datasets import load_dataset
import re
import torch
import logging
from collections import Counter

# base_model = "maywell/Synatra-7B-v0.3-dpo"
base_model = "/data/llm/Synatra-7B-v0.3-dpo"
# base_model = "D:\Synatra-7B-v0.3-dpo"
dataset_name, new_model = "joonhok-exo-ai/korean_law_open_data_precedents", "/data/llm/lawsuit-7B-wage-f"

# Loading a Gath_baize dataset
custom_cache_dir = "/data/huggingface/cache/"
# custom_cache_dir = "D:/huggingface/cache/"

test_case_file = "/data/llm/test_case_numbers.txt"
# test_case_file = r"D:\test_case_numbers.txt"

cutoff_len = 4096

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

    combined_parts = []

    if decision:
        combined_parts.append(f'판시사항: {decision}')
    if summary:
        combined_parts.append(f'판결요지: {summary}')
    if laws:
        combined_parts.append(f'참조조문: {laws}')

    if precedents:
        precedents += ', ' + examples['법원명'] + " " + format_date(examples['선고일자']) + " " + examples['선고'] + " " + examples['사건번호'] + " " + '판결'
    else:
        precedents += examples['법원명'] + " " + format_date(examples['선고일자']) + " " + examples['선고'] + " " + examples[
            '사건번호'] + " " + '판결'
    combined_parts.append(f'참조판례: {precedents}')

    if reason:
        split_text = re.split("【이\s*유】", examples['전문'], maxsplit=1)
        # 분할된 결과 확인 및 처리
        if len(split_text) > 1:
            reason_text = split_text[1]
        else:
            reason_text = split_text[0]

        final_text = re.split("대법원\s+|판사\s+|대법원판사\s+|대법관\s+", reason_text, maxsplit=1)
        final_text = final_text[0] if final_text else ""
        if final_text:
            combined_parts.append(f'이유: {final_text}')


    combined_text = "\n".join(combined_parts)

    return {'input_text': combined_text}

# 파일에서 판례정보일련번호 목록 로드
with open(test_case_file, 'r') as f:
    test_case_numbers = [line.strip() for line in f.readlines()]

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

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


# dataset = load_dataset(dataset_name, cache_dir=custom_cache_dir, split="train").shuffle()
dataset = load_dataset(dataset_name, cache_dir=custom_cache_dir, split="train")

# '민사' 사건 중 '임금'만 포함된 데이터 필터링하면서 테스트 케이스 제외
civil_cases_with_wage_excluded = dataset.filter(
    lambda x: x['사건종류명'] == '민사' and
              x['사건명'] is not None and
              '임금' in x['사건명']
              # and
              # str(x['판례정보일련번호']) in test_case_numbers
              # str(x['판례정보일련번호']) not in test_case_numbers
)

# civil_cases_with_wage_excluded.to_csv('civil_cases_with_wage.csv')

# '참조조문'에서 법률 이름 추출
law_references = []
for reference in civil_cases_with_wage_excluded['참조조문']:
    if reference:
        for data in reference.split('/'):
            for data2 in data.split(','):
                cleaned_text = re.sub(r"\[\d+\]", "", data2).strip()
                cleaned_text = re.sub(r"([가-힣]+\.)+", "", cleaned_text).strip()
                if '법' not in cleaned_text:
                    cleaned_text = re.findall(r'([가-힣\s]+(?:법|법률))', law_references[-1])[0] + ' ' + cleaned_text
                elif '같은법시행령' in cleaned_text:
                    cleaned_text = re.findall(r'([가-힣\s]+(?:법|법률))', law_references[-1])[0] + ' ' + cleaned_text.replace('같은법시행령', '')
                elif '같은법' in cleaned_text:
                    cleaned_text = re.findall(r'([가-힣\s]+(?:법|법률))', law_references[-1])[0] + ' ' + cleaned_text.replace('같은법', '')
                law_references.append(cleaned_text)
                # 정규 표현식을 사용하여 '___법 ____조' 및 '___법률 ____조' 형식 추출
                # matches = re.findall(r'([가-힣\s]+(?:법|법률))\s제(\d+조)', cleaned_text)
                # for match in matches:
                #     # '법률 이름 제조번호' 형태로 정리
                #     full_reference = f"{match[0]} 제{match[1]}"
                #     law_references.append(full_reference)

law_count = Counter(law_references)
print(law_count)
# 원본 데이터셋에 전처리 함수 적용
processed_dataset = civil_cases_with_wage_excluded.map(preprocess_data)

# 원본 데이터셋의 다른 열을 제거하고 'input_text'만 남깁니다.
final_dataset = processed_dataset.remove_columns([column_name for column_name in processed_dataset.column_names if column_name != 'input_text'])

# 데이터셋 토큰화 함수
def tokenize_function(examples):
    return tokenizer(examples['input_text'], truncation=True, padding=True, max_length=cutoff_len)

# 데이터셋 토큰화 적용
tokenized_dataset = final_dataset.map(tokenize_function, batched=True)

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
    per_device_train_batch_size= 2,
    gradient_accumulation_steps= 2,
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
    per_device_train_batch_size= 2,
    gradient_accumulation_steps= 2,
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
        args=training_arguments_f,
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