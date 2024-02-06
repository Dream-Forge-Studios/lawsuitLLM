from datasets import load_dataset
import pandas as pd

# 데이터셋 로드
custom_cache_dir = "D:/huggingface/cache/"
dataset = load_dataset('joonhok-exo-ai/korean_law_open_data_precedents', cache_dir=custom_cache_dir)

# '민사' 사건만 필터링
civil_cases = dataset.filter(lambda x: x['사건종류명'] == '민사')

# pandas DataFrame으로 변환
df = pd.DataFrame(civil_cases)

# df['train']에서 각 사전을 행으로 가지는 새로운 DataFrame 생성
expanded_df = pd.json_normalize(df['train'])

case_counts = expanded_df.groupby('사건명').size()

# 사건명별 데이터 개수를 CSV 파일로 저장
case_counts.to_csv('case_counts.csv', encoding='utf-8-sig')

