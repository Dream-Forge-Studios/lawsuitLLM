from datasets import load_dataset
import pandas as pd

# 데이터셋 로드
# custom_cache_dir = "D:/huggingface/cache/"
# dataset = load_dataset('joonhok-exo-ai/korean_law_open_data_precedents', cache_dir=custom_cache_dir, split="train")

dataset_name = "joonhok-exo-ai/korean_law_open_data_precedents"
dataset = load_dataset(dataset_name, split="train")

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