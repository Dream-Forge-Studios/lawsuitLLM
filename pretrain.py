from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,DataCollatorForLanguageModeling,TrainingArguments, Trainer
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import os, torch, wandb, platform, warnings
from datasets import load_dataset
import re
import torch
import logging
from collections import Counter
import pandas as pd
from datasets import Dataset

# base_model = "maywell/Synatra-7B-v0.3-dpo"
base_model = "/data/llm/Synatra-7B-v0.3-dpo"
# base_model = "D:\Synatra-7B-v0.3-dpo"
dataset_name, new_model = "joonhok-exo-ai/korean_law_open_data_precedents", "/data/llm/lawsuit-7B-wage-lawNotNone-oneLaw-a"

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
              '임금' in x['사건명'] and
              x['참조조문'] is not None
              # str(x['판례정보일련번호']) in test_case_numbers
              # str(x['판례정보일련번호']) not in test_case_numbers
)

one_laws = ['근로기준법 제127조의2', '민법 제126조', '근로기준법시행령 제33조 제2항', '민법 제5조', '민사소송법 제51조', '민사소송법 제328조', '근로기준법  제19조', '근로기준법  제31조  근로기준법 제42조', '근로기준법  제31조  근로기준법 제46조', '근로기준법  제48조', '민사소송법 제395조', '민사소송법 제383조', '같은 법 제41조', '같은 법 제42조', '같은 법 제43조', '같은 법 제55조', '같은 법 제49조', '근로기준법  제31조 제3항', '근로기준법 제36조의2', '남녀고용평등법 제6조의2', '노동조합법 제36조', '같은 법 ', '같은 법 제19조', '같은 법 제18조', '민사소송법 제384조', '민사소송법  제183조', '민사소송법 제193조 제2항', '근로기준법  제31조 제1항', '같은 법 제95조', '근로기준법 제98조', '노동조합법 제33조', '민법 제538조 제2항', '최저임금법 제7조', '민법 제460조', '민법 제396조', '민사소송법 제422조 제1항 제10호', '노동조합법 제39조', '노동조합법 제43조', '소득세법 제142조 제1항', '근로기준법 제9조 제2항', '민법 제454조', '상법 제42조', '노동조합법 제33조 제1항', '노동조합법 제34조 제1항', '민사소송법 제714조 제2항', '민법 제748조 제2항', '민법 제395조', '민법 제656조 제2항', '의료보험법 제3조', '국가공무원법 제46조 제2항', '입법 폐지) 제8조', '입법 구 국회공무원수당규정(1995. 7. 22. 국회규정 제423호로 개정되기 전의 것) 제9조', '입법 공무원수당규정 제19조 제7항', '국가공무원법 제48조 제1항', '국가공무원법 공무원보수규정 제4조 제1호', '공무원연금법 제24조 제2항', '의료보험법 제25조', '민법 제166조 제1항', '헌법재판소법 제47조 제2항', '공무원연금법 제24조', '청원경찰법 제6조', '청원경찰법  제8조 제1항', '사립학교법 제58조 제1항', '사립학교교원연금법 제33조', '사립학교교원연금법 제42조', '민법 제107조', '민법 제110조', '민법 제712조', '민법 제713조', '파산법 제50조', '파산법 제154조', '파산법 제50조제154조', '상법 제382조 제2항', '새마을금고법 제24조', '지방재정법 제34조', '지방재정법 제35조', '노동조합및노동관계조정법 제32조', '상법 제235조', '민법 제166조', '노동조합및노동관계조정법 제29조', '민법 제742조', '한국전기통신공사법폐지법률 부칙 제6조', '근로기준법 제95조 제1항', '근로기준법 제57조', '근로기준법 제31조', '상법 제227조', '상법 제517조', '상법 제344조', '상법 제345조', '상법 제370조', '증권거래법 제52조', '상법 제341조', '상법 제625조 제2호', '노동조합및노동관계조정법 제24조 제2항', '노동조합및노동관계조정법 제81조 제4호', '노동조합및노동관계조정법 제90조', '민법 제147조', '근로기준법 제34조 제2항', '직업안정법 제33조 제1항', '근로기준법 제8조', '공무원연금법 제3조 제1항', '공무원연금법 제66조', '지방공무원법 제2조', '국가공무원법 제70조 제1항 제3호', '근로기준법 제34조 제1항', '민사소송법 제248조', '민사소송법 제253조', '민사소송법 제436조', '민사소송법 제451조', '헌법재판소법 제75조 제7항', '여객자동차 운수사업법 제22조 제1항', '［1］ 근로기준법 제14조', '［2］ 근로기준법 제5조', ' 근로기준법 제14조', ' 근로기준법 제34조', '외국인근로자의 고용 등에 관한 법률 제22조', '［3］ 파견근로자보호 등에 관한 법률 제2조', ' 파견근로자보호 등에 관한 법 제5조 제1항', '［1］ 민법 제750조', '［2］ 민법 제2조', ' 민법 제750조', '상법 제171조 제1항', '최저임금법 제3조 제1항', '민사소송법 제1조', '국제사법 제2조 제1항', '국제사법 제28조 제5항', '사립학교교직원연금법 제33조', '사립학교교직원연금법 제42조', '최저임금법 제5조의2', '근로기준법 제20조', '근로기준법 시행령 제6조 제2항 제3호', '최저임금법 제6조 제3항', '최저임금법 제1조', '민법 제398조 제2항', '민법 제680조', '민법 제31조', '민사소송법 제432조', '민사소송법 제142조', '민사소송법 제185조 제1항', '민사소송법 제2항', '민사소송법 제186조', '근로기준법(2007. 4. 11. 법률 제8372호로 전부 개정되기 전의 것) 제54조(현행 제55조 참조)', '근로기준법 제59조(현행 제60조 참조)', '국가공무원법 제56조', '국가공무원법 제63조', '사립학교법 제53조의2 제7항', '교원지위향상을 위한 특별법 제7조 제1항', '교원지위향상을 위한 특별법 제10조 제2항', '근로기준법 제18조 제1호', '임금채권보장법 시행령 제4조', '기업구조조정촉진법 제2조 제5호', '기업구조조정촉진법 제7조 제1항', '석탄산업법 제39조의3 제1항', '민사소송 등 인지법 제1조', '민사소송법 제216조', '교원지위 향상을 위한 특별법 제1조', '상법 제376조', '지방공무원법 제2조 제1항', '지방공무원법 제2항 제2호', '지방공무원법 제44조 제4항', '지방공무원법 제45조 제1항 제2호', '지방공무원법 지방공무원 보수규정 제30조', '지방공무원법 지방공무원 수당 등에 관한 규정 제15조', '지방공무원법 제16조', '지방공무원법 제17조', '행정소송법 제3조 제2호', '근로기준법 시행령 제30조', '상법 제530조의10', '헌법 제15조', '근로기준법 제7조', '민법 제106조', '헌법 제33조', '노동조합 및 노동관계조정법 제44조', '남녀고용평등과 일·가정 양립 지원에 관한 법률 제19조', '남녀고용평등과 일·가정 양립 지원에 관한 법률 제8조 제1항', '채무자 회생 및 파산에 관한 법률 제1조', '채무자 회생 및 파산에 관한 법 제382조 제1항', '채무자 회생 및 파산에 관한 법 제384조', '채무자 회생 및 파산에 관한 법 제423조', '채무자 회생 및 파산에 관한 법 제424조', '채무자 회생 및 파산에 관한 법 제446조 제1항 제2호', '채무자 회생 및 파산에 관한 법 제473조 제4호', '채무자 회생 및 파산에 관한 법 제10호', '채무자 회생 및 파산에 관한 법 제475조', '채무자 회생 및 파산에 관한 법 제476조', '채무자 회생 및 파산에 관한 법 제505조', '근로자퇴직급여 보장법 제12조', '근로자퇴직급여 보장법 제8조 제2항', '민법 제497조', '민사집행법 제246조 제1항 제4호', '민법 제476조', '민법 제477조', '민법 제499조', '민사소송법 제216조 제2항', '민법 제406조 제2항', '민법 제766조 제1항', '소득세법 제127조 제1항', '국민연금법 제88조의2 제1항', '국민건강보험법 제79조 제1항', '고용보험 및 산업재해보상보험의 보험료징수 등에 관한 법률 제16조의2 제1항', '파견근로자보호 등에 관한 법률 제6조의2 제3항 제1호', '채무자 회생 및 파산에 관한 법률 제193조 제2항 제6호', '채무자 회생 및 파산에 관한 법률 제179조', '변호사법 제34조 제1항', '변호사법 제2항', '변호사법 제109조 제2호', '변호사법 제116조', '민법 제168조 제3호', '최저임금법 제6조 제5항', '근로기준법 제6항 제1호', '근로기준법 제60조 제6항 제1호', '파견근로자보호 등에 관한 법률 제2조 제1항', '파견근로자보호 등에 관한 법 제34조 제1항', '근로기준법 제7호', '근로기준법 제73조', '근로기준법 제78조', '근로기준법 제79조', '근로기준법 제109조', '근로기준법 제110조', '산업재해보상보험법 제52조', '산업재해보상보험법 제80조 제1항', '민법 제110조 제1항', '노동조합 및 노동관계조합법 제41조 제1항', '노동조합 및 노동관계조합법 제46조', '민사소송법 제173조 제1항', '민사소송법 제185조', '중등교육법 구 국립 및 공립 초·중등학교 회계규칙(2012. 5. 2. 교육과학기술부령 제146호로 폐지) 제9조(현행 국립 유치원 및 초·중등학교 회계규칙 제10조 참조)', '중등교육법 구 대구광역시립학교 회계 규칙(2006. 10. 13. 대구광역시교육규칙 제498호) 제10조', '지방교육자치에 관한 법 구 지방공무원 보수규정(2013. 12. 11. 대통령령 제24917호로 개정되기 전의 것) 제4조 제7항(현행 삭제)', '지방교육자치에 관한 법 제9조 제1항 제3호', '지방교육자치에 관한 법 제12조 제1항', '지방교육자치에 관한 법 구 공무원보수규정(1997. 12. 27. 대통령령 제15551호로 개정되기 전의 것) 제5조 [별표 8](현행 삭제)', '지방교육자치에 관한 법 제13조', '헌법 제10조', '근로기준법 제70조', '근로기준법 제109조 제1호', '근로기준법 제110조 제1호', '헌법 제32조 제1항', '여객자동차 운수사업법 제1조', '아이돌봄 지원법 제2조 제3호', '아이돌봄 지원법 제4호', '아이돌봄 지원법 제11조 제1항', '아이돌봄 지원법 제14조', '아이돌봄 지원법 시행규칙 제11조', '건강가정기본법 제35조 제1항', '건강가정기본법 제5항', '민법 제168조', '최저임금법 시행령 제5조의3', '기간제 및 단시간근로자 보호 등에 관한 법률 제1조', '기간제 및 단시간근로자 보호 등에 관한 법 제4조', '기간제 및 단시간근로자 보호 등에 관한 법 제8조 제1항', '근로자의 날 제정에 관한 법률', '고용정책 기본법 제18조', '민사소송법 제136조', '소송촉진 등에 관한 특례법 제3조 제2항', '남녀고용평등과 일ㆍ가정 양립 지원에 관한 법률 제8조', '근로자퇴직급여 보장법 제2조 제6호', '근로자퇴직급여 보장법 제17조 제1항', '근로자퇴직급여 보장법 제4항', '근로자퇴직급여 보장법 제20조', '근로자퇴직급여 보장법 제21조 제1항', '근로자퇴직급여 보장법 제22조', '근로자퇴직급여 보장법 제44조 제2호', '근로기준법 공무원보수규정 제5조 [별표 12]', '사립학교법 제60조의3', '근로자퇴직급여 보장법 제10조', '근로자퇴직급여 보장법 제14조', '근로자퇴직급여 보장법 제20조 제1항', '근로자퇴직급여 보장법 제3항', '사립학교법 제30조', '사립학교법 제31조 제1항', '민사소송법 제150조 제1항', '근로기준법 제37조 제1항', '여객자동차 운수사업법 제25조 제1항', '여객자동차 운수사업법 제2항', '여객자동차 운수사업법 제85조 제1항 제23호', '여객자동차 운수사업법 시행규칙 제58조 제1항 제2호', '상법 제4조', '상법 제5조 제1항', '의료법 제2조 제2항 제1호', '의료법 제4조 제1항', '의료법 제2항', '의료법 제5조', '의료법 제8조', '의료법 제9조', '의료법 제10조', '의료법 제11조', '의료법 제12조', '의료법 제13조', '의료법 제15조', '의료법 제19조', '의료법 제23조의5', '의료법 제30조', '의료법 제56조', '의료법 제57조', '근로기준법 시행령 제7조 [별표 1]', '민법 제660조 제1항', '사립학교법 제53조의2 제9항', '교원의 지위 향상 및 교육활동 보호를 위한 특별법 제10조 제2항']

def filter_with_reference(cases, one_laws):
    filtered_cases = []
    law_references = []
    for case in cases:
        references = case['참조조문']
        if references:
            for reference in references.split('/'):
                for data in reference.split(','):
                    cleaned_text = re.sub(r"\[\d+\]", "", data).strip()
                    cleaned_text = re.sub(r"([가-힣]+\.)+", "", cleaned_text).strip()
                    if '법' not in cleaned_text:
                        cleaned_text = re.findall(r'([가-힣\s]+(?:법|법률))', law_references[-1])[0] + ' ' + cleaned_text
                    elif '같은법시행령' in cleaned_text:
                        cleaned_text = re.findall(r'([가-힣\s]+(?:법|법률))', law_references[-1])[
                                           0] + ' ' + cleaned_text.replace('같은법시행령', '')
                    elif '같은법' in cleaned_text:
                        cleaned_text = re.findall(r'([가-힣\s]+(?:법|법률))', law_references[-1])[
                                           0] + ' ' + cleaned_text.replace('같은법', '')
                    law_references.append(cleaned_text)
                    if cleaned_text in one_laws:
                        filtered_cases.append(case)
                        break
    return filtered_cases

# 최종 필터링된 데이터셋 생성
civil_cases_with_wage_excluded = filter_with_reference(civil_cases_with_wage_excluded, one_laws)

df = pd.DataFrame(civil_cases_with_wage_excluded)
# DataFrame을 Hugging Face의 Dataset 객체로 변환
civil_cases_with_wage_excluded = Dataset.from_pandas(df)

# civil_cases_with_wage_excluded.to_csv('civil_cases_with_wage.csv')

# '참조조문'에서 법률 이름 추출
# law_references = []
# for references in civil_cases_with_wage_excluded['참조조문']:
#     if references:
#         for reference in references.split('/'):
#             for data in reference.split(','):
#                 cleaned_text = re.sub(r"\[\d+\]", "", data).strip()
#                 cleaned_text = re.sub(r"([가-힣]+\.)+", "", cleaned_text).strip()
#                 if '법' not in cleaned_text:
#                     cleaned_text = re.findall(r'([가-힣\s]+(?:법|법률))', law_references[-1])[0] + ' ' + cleaned_text
#                 elif '같은법시행령' in cleaned_text:
#                     cleaned_text = re.findall(r'([가-힣\s]+(?:법|법률))', law_references[-1])[0] + ' ' + cleaned_text.replace('같은법시행령', '')
#                 elif '같은법' in cleaned_text:
#                     cleaned_text = re.findall(r'([가-힣\s]+(?:법|법률))', law_references[-1])[0] + ' ' + cleaned_text.replace('같은법', '')
#                 law_references.append(cleaned_text)
#                 # 정규 표현식을 사용하여 '___법 ____조' 및 '___법률 ____조' 형식 추출
#                 # matches = re.findall(r'([가-힣\s]+(?:법|법률))\s제(\d+조)', cleaned_text)
#                 # for match in matches:
#                 #     # '법률 이름 제조번호' 형태로 정리
#                 #     full_reference = f"{match[0]} 제{match[1]}"
#                 #     law_references.append(full_reference)
#
# law_count = Counter(law_references)
# print(law_count)

# 원본 데이터셋에 전처리 함수 적용
processed_dataset = civil_cases_with_wage_excluded.map(preprocess_data)
# processed_dataset = list(map(preprocess_data, final_filtered_dataset))

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