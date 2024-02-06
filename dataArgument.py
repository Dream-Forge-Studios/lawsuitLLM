from datasets import load_dataset
import pandas as pd
import re

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

    split_text = re.split("【이\s*유】", reason, maxsplit=1)
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



# 데이터셋 로드
custom_cache_dir = "D:/huggingface/cache/"
dataset = load_dataset('joonhok-exo-ai/korean_law_open_data_precedents', cache_dir=custom_cache_dir, split="train")

# dataset_name = "joonhok-exo-ai/korean_law_open_data_precedents"
# dataset = load_dataset(dataset_name, split="train")

civil_cases_with_wage_excluded = dataset.filter(
    lambda x: x['사건종류명'] == '민사'
              # x['사건명'] is not None and
              # '임금' in x['사건명']
)

# 원본 데이터셋에 전처리 함수 적용
processed_dataset = civil_cases_with_wage_excluded.map(preprocess_data)

# processed_dataset에는 이미 'input_text'가 추가되어 있음
# 원본 데이터셋의 다른 열을 제거하고 'input_text'만 남깁니다.
final_dataset = processed_dataset.remove_columns([column_name for column_name in processed_dataset.column_names if column_name != 'input_text'])

# '민사' 사건만 필터링
civil_cases_with_wage = dataset.filter(lambda x: x['사건종류명'] == '민사' and x['사건명'] is not None and '임금' in x['사건명'])

# 참조 조문이 존재하는 데이터 필터링
filtered_cases = civil_cases_with_wage.filter(lambda x: x['참조조문'] is not None and len(x['참조조문']) > 0)

# 테스트 데이터로 사용할 비율 설정 (예: 10%)
test_size = 0.1
# 훈련 데이터와 테스트 데이터로 분리
train_test_data = filtered_cases.train_test_split(test_size=test_size)

# 훈련 데이터와 테스트 데이터 할당
train_data = train_test_data['train']
test_data = train_test_data['test']

# 테스트 데이터의 판례번호를 추출
test_case_numbers = test_data['판례정보일련번호']

# 판례번호를 파일에 저장
test_case_file = "/data/llm/test_case_numbers.txt"
with open(test_case_file, 'w') as f:
    for number in test_case_numbers:
        f.write("%s\n" % number)

print(f"Test case numbers saved to {test_case_file}")