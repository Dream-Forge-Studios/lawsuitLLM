import subprocess

# 실행할 모델의 리스트
# new_models = ['lawsuit-7B-wage-lawNotNone-oneLaw-a', 'lawsuit-7B-wage-reason-lawNotNone-oneLaw-a', 'lawsuit-7B-civil-wage-a', 'lawsuit-7B-civil-wage-f', 'lawsuit-7B-wage-reason-a', 'lawsuit-7B-wage-reason-f']
new_models = ['sangbul-e200', 'sangbul-e150', 'sangbul-e100', 'sangbul-e50']

# 각 모델에 대해 test.py 스크립트 실행
for model in new_models:
    cmd = f'python doc_test.py --new_model {model}'
    subprocess.run(cmd, shell=True)
