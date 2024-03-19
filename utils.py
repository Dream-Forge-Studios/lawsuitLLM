# -*- coding: utf-8 -*-

import pandas as pd
from datasets import Dataset
import re
import random
from datasets import load_dataset
import os
import json
import pdfplumber
import requests
from bs4 import BeautifulSoup, NavigableString, Tag
from tqdm import tqdm

# Loading a Gath_baize dataset
custom_cache_dir = "/data/huggingface/cache/"
# custom_cache_dir = "D:/huggingface/cache/"

train_data_root = '/data'
# train_data_root = "D:"

# one_laws = ['근로기준법 제127조의2', '민법 제126조', '근로기준법시행령 제33조 제2항', '민법 제5조', '민사소송법 제51조', '민사소송법 제328조', '근로기준법  제19조', '근로기준법  제31조  근로기준법 제42조', '근로기준법  제31조  근로기준법 제46조', '근로기준법  제48조', '민사소송법 제395조', '민사소송법 제383조', '같은 법 제41조', '같은 법 제42조', '같은 법 제43조', '같은 법 제55조', '같은 법 제49조', '근로기준법  제31조 제3항', '근로기준법 제36조의2', '남녀고용평등법 제6조의2', '노동조합법 제36조', '같은 법 ', '같은 법 제19조', '같은 법 제18조', '민사소송법 제384조', '민사소송법  제183조', '민사소송법 제193조 제2항', '근로기준법  제31조 제1항', '같은 법 제95조', '근로기준법 제98조', '노동조합법 제33조', '민법 제538조 제2항', '최저임금법 제7조', '민법 제460조', '민법 제396조', '민사소송법 제422조 제1항 제10호', '노동조합법 제39조', '노동조합법 제43조', '소득세법 제142조 제1항', '근로기준법 제9조 제2항', '민법 제454조', '상법 제42조', '노동조합법 제33조 제1항', '노동조합법 제34조 제1항', '민사소송법 제714조 제2항', '민법 제748조 제2항', '민법 제395조', '민법 제656조 제2항', '의료보험법 제3조', '국가공무원법 제46조 제2항', '입법 폐지) 제8조', '입법 구 국회공무원수당규정(1995. 7. 22. 국회규정 제423호로 개정되기 전의 것) 제9조', '입법 공무원수당규정 제19조 제7항', '국가공무원법 제48조 제1항', '국가공무원법 공무원보수규정 제4조 제1호', '공무원연금법 제24조 제2항', '의료보험법 제25조', '민법 제166조 제1항', '헌법재판소법 제47조 제2항', '공무원연금법 제24조', '청원경찰법 제6조', '청원경찰법  제8조 제1항', '사립학교법 제58조 제1항', '사립학교교원연금법 제33조', '사립학교교원연금법 제42조', '민법 제107조', '민법 제110조', '민법 제712조', '민법 제713조', '파산법 제50조', '파산법 제154조', '파산법 제50조제154조', '상법 제382조 제2항', '새마을금고법 제24조', '지방재정법 제34조', '지방재정법 제35조', '노동조합및노동관계조정법 제32조', '상법 제235조', '민법 제166조', '노동조합및노동관계조정법 제29조', '민법 제742조', '한국전기통신공사법폐지법률 부칙 제6조', '근로기준법 제95조 제1항', '근로기준법 제57조', '근로기준법 제31조', '상법 제227조', '상법 제517조', '상법 제344조', '상법 제345조', '상법 제370조', '증권거래법 제52조', '상법 제341조', '상법 제625조 제2호', '노동조합및노동관계조정법 제24조 제2항', '노동조합및노동관계조정법 제81조 제4호', '노동조합및노동관계조정법 제90조', '민법 제147조', '근로기준법 제34조 제2항', '직업안정법 제33조 제1항', '근로기준법 제8조', '공무원연금법 제3조 제1항', '공무원연금법 제66조', '지방공무원법 제2조', '국가공무원법 제70조 제1항 제3호', '근로기준법 제34조 제1항', '민사소송법 제248조', '민사소송법 제253조', '민사소송법 제436조', '민사소송법 제451조', '헌법재판소법 제75조 제7항', '여객자동차 운수사업법 제22조 제1항', '［1］ 근로기준법 제14조', '［2］ 근로기준법 제5조', ' 근로기준법 제14조', ' 근로기준법 제34조', '외국인근로자의 고용 등에 관한 법률 제22조', '［3］ 파견근로자보호 등에 관한 법률 제2조', ' 파견근로자보호 등에 관한 법 제5조 제1항', '［1］ 민법 제750조', '［2］ 민법 제2조', ' 민법 제750조', '상법 제171조 제1항', '최저임금법 제3조 제1항', '민사소송법 제1조', '국제사법 제2조 제1항', '국제사법 제28조 제5항', '사립학교교직원연금법 제33조', '사립학교교직원연금법 제42조', '최저임금법 제5조의2', '근로기준법 제20조', '근로기준법 시행령 제6조 제2항 제3호', '최저임금법 제6조 제3항', '최저임금법 제1조', '민법 제398조 제2항', '민법 제680조', '민법 제31조', '민사소송법 제432조', '민사소송법 제142조', '민사소송법 제185조 제1항', '민사소송법 제2항', '민사소송법 제186조', '근로기준법(2007. 4. 11. 법률 제8372호로 전부 개정되기 전의 것) 제54조(현행 제55조 참조)', '근로기준법 제59조(현행 제60조 참조)', '국가공무원법 제56조', '국가공무원법 제63조', '사립학교법 제53조의2 제7항', '교원지위향상을 위한 특별법 제7조 제1항', '교원지위향상을 위한 특별법 제10조 제2항', '근로기준법 제18조 제1호', '임금채권보장법 시행령 제4조', '기업구조조정촉진법 제2조 제5호', '기업구조조정촉진법 제7조 제1항', '석탄산업법 제39조의3 제1항', '민사소송 등 인지법 제1조', '민사소송법 제216조', '교원지위 향상을 위한 특별법 제1조', '상법 제376조', '지방공무원법 제2조 제1항', '지방공무원법 제2항 제2호', '지방공무원법 제44조 제4항', '지방공무원법 제45조 제1항 제2호', '지방공무원법 지방공무원 보수규정 제30조', '지방공무원법 지방공무원 수당 등에 관한 규정 제15조', '지방공무원법 제16조', '지방공무원법 제17조', '행정소송법 제3조 제2호', '근로기준법 시행령 제30조', '상법 제530조의10', '헌법 제15조', '근로기준법 제7조', '민법 제106조', '헌법 제33조', '노동조합 및 노동관계조정법 제44조', '남녀고용평등과 일·가정 양립 지원에 관한 법률 제19조', '남녀고용평등과 일·가정 양립 지원에 관한 법률 제8조 제1항', '채무자 회생 및 파산에 관한 법률 제1조', '채무자 회생 및 파산에 관한 법 제382조 제1항', '채무자 회생 및 파산에 관한 법 제384조', '채무자 회생 및 파산에 관한 법 제423조', '채무자 회생 및 파산에 관한 법 제424조', '채무자 회생 및 파산에 관한 법 제446조 제1항 제2호', '채무자 회생 및 파산에 관한 법 제473조 제4호', '채무자 회생 및 파산에 관한 법 제10호', '채무자 회생 및 파산에 관한 법 제475조', '채무자 회생 및 파산에 관한 법 제476조', '채무자 회생 및 파산에 관한 법 제505조', '근로자퇴직급여 보장법 제12조', '근로자퇴직급여 보장법 제8조 제2항', '민법 제497조', '민사집행법 제246조 제1항 제4호', '민법 제476조', '민법 제477조', '민법 제499조', '민사소송법 제216조 제2항', '민법 제406조 제2항', '민법 제766조 제1항', '소득세법 제127조 제1항', '국민연금법 제88조의2 제1항', '국민건강보험법 제79조 제1항', '고용보험 및 산업재해보상보험의 보험료징수 등에 관한 법률 제16조의2 제1항', '파견근로자보호 등에 관한 법률 제6조의2 제3항 제1호', '채무자 회생 및 파산에 관한 법률 제193조 제2항 제6호', '채무자 회생 및 파산에 관한 법률 제179조', '변호사법 제34조 제1항', '변호사법 제2항', '변호사법 제109조 제2호', '변호사법 제116조', '민법 제168조 제3호', '최저임금법 제6조 제5항', '근로기준법 제6항 제1호', '근로기준법 제60조 제6항 제1호', '파견근로자보호 등에 관한 법률 제2조 제1항', '파견근로자보호 등에 관한 법 제34조 제1항', '근로기준법 제7호', '근로기준법 제73조', '근로기준법 제78조', '근로기준법 제79조', '근로기준법 제109조', '근로기준법 제110조', '산업재해보상보험법 제52조', '산업재해보상보험법 제80조 제1항', '민법 제110조 제1항', '노동조합 및 노동관계조합법 제41조 제1항', '노동조합 및 노동관계조합법 제46조', '민사소송법 제173조 제1항', '민사소송법 제185조', '중등교육법 구 국립 및 공립 초·중등학교 회계규칙(2012. 5. 2. 교육과학기술부령 제146호로 폐지) 제9조(현행 국립 유치원 및 초·중등학교 회계규칙 제10조 참조)', '중등교육법 구 대구광역시립학교 회계 규칙(2006. 10. 13. 대구광역시교육규칙 제498호) 제10조', '지방교육자치에 관한 법 구 지방공무원 보수규정(2013. 12. 11. 대통령령 제24917호로 개정되기 전의 것) 제4조 제7항(현행 삭제)', '지방교육자치에 관한 법 제9조 제1항 제3호', '지방교육자치에 관한 법 제12조 제1항', '지방교육자치에 관한 법 구 공무원보수규정(1997. 12. 27. 대통령령 제15551호로 개정되기 전의 것) 제5조 [별표 8](현행 삭제)', '지방교육자치에 관한 법 제13조', '헌법 제10조', '근로기준법 제70조', '근로기준법 제109조 제1호', '근로기준법 제110조 제1호', '헌법 제32조 제1항', '여객자동차 운수사업법 제1조', '아이돌봄 지원법 제2조 제3호', '아이돌봄 지원법 제4호', '아이돌봄 지원법 제11조 제1항', '아이돌봄 지원법 제14조', '아이돌봄 지원법 시행규칙 제11조', '건강가정기본법 제35조 제1항', '건강가정기본법 제5항', '민법 제168조', '최저임금법 시행령 제5조의3', '기간제 및 단시간근로자 보호 등에 관한 법률 제1조', '기간제 및 단시간근로자 보호 등에 관한 법 제4조', '기간제 및 단시간근로자 보호 등에 관한 법 제8조 제1항', '근로자의 날 제정에 관한 법률', '고용정책 기본법 제18조', '민사소송법 제136조', '소송촉진 등에 관한 특례법 제3조 제2항', '남녀고용평등과 일ㆍ가정 양립 지원에 관한 법률 제8조', '근로자퇴직급여 보장법 제2조 제6호', '근로자퇴직급여 보장법 제17조 제1항', '근로자퇴직급여 보장법 제4항', '근로자퇴직급여 보장법 제20조', '근로자퇴직급여 보장법 제21조 제1항', '근로자퇴직급여 보장법 제22조', '근로자퇴직급여 보장법 제44조 제2호', '근로기준법 공무원보수규정 제5조 [별표 12]', '사립학교법 제60조의3', '근로자퇴직급여 보장법 제10조', '근로자퇴직급여 보장법 제14조', '근로자퇴직급여 보장법 제20조 제1항', '근로자퇴직급여 보장법 제3항', '사립학교법 제30조', '사립학교법 제31조 제1항', '민사소송법 제150조 제1항', '근로기준법 제37조 제1항', '여객자동차 운수사업법 제25조 제1항', '여객자동차 운수사업법 제2항', '여객자동차 운수사업법 제85조 제1항 제23호', '여객자동차 운수사업법 시행규칙 제58조 제1항 제2호', '상법 제4조', '상법 제5조 제1항', '의료법 제2조 제2항 제1호', '의료법 제4조 제1항', '의료법 제2항', '의료법 제5조', '의료법 제8조', '의료법 제9조', '의료법 제10조', '의료법 제11조', '의료법 제12조', '의료법 제13조', '의료법 제15조', '의료법 제19조', '의료법 제23조의5', '의료법 제30조', '의료법 제56조', '의료법 제57조', '근로기준법 시행령 제7조 [별표 1]', '민법 제660조 제1항', '사립학교법 제53조의2 제9항', '교원의 지위 향상 및 교육활동 보호를 위한 특별법 제10조 제2항']
#
def filter_with_reference(cases, target):
    filtered_cases = []
    law_references = []
    for case in cases:
        references = case['참조조문']
        if references:
            checker = True
            for reference in references.split('/'):
                for data in reference.split(','):
                    cleaned_text = re.sub(r"\[\d+\]", "", data).strip()
                    cleaned_text = re.sub(r"([가-힣]+\.)+", "", cleaned_text).strip()
                    if '법' not in cleaned_text:
                        if '법령 제149호 대외무역규칙' in law_references[-1]:
                            cleaned_text = re.sub(r"제\d+조", "", law_references[-1]) + cleaned_text
                        else:
                            cleaned_text = re.findall(r'([가-힣\s]+(?:법|법률|규정|규칙))', law_references[-1])[0] + ' ' + cleaned_text
                    elif '같은법시행령' in cleaned_text:
                        cleaned_text = re.findall(r'([가-힣\s]+(?:법|법률))', law_references[-1])[
                                           0] + ' ' + cleaned_text.replace('같은법시행령', '')
                    elif '같은법' in cleaned_text:
                        cleaned_text = re.findall(r'([가-힣\s]+(?:법|법률))', law_references[-1])[
                                           0] + ' ' + cleaned_text.replace('같은법', '')
                    elif '같은 법' in cleaned_text:
                        cleaned_text = re.findall(r'([가-힣\s]+(?:법|법률))', law_references[-1])[
                                           0] + ' ' + cleaned_text.replace('같은 법', '')
                    law_references.append(cleaned_text)
                    if type(target) is list:
                        if cleaned_text in target:
                            filtered_cases.append(case)
                            break
                    if target == 2 and cleaned_text[0] == '구':
                        checker = False
                        break
            if target == 2 and checker:
                filtered_cases.append(case)
    if target == 1:
        return law_references
    else:
        df = pd.DataFrame(filtered_cases)
        return Dataset.from_pandas(df), law_references

def sample_selcet(input_file, out_file, num):
    # 파일 로드
    with open(input_file, "r") as file:
        case_numbers = file.read().splitlines()

    # 100개의 숫자를 랜덤으로 선택
    random_selected_numbers = random.sample(case_numbers, num)

    with open(out_file, 'w') as f:
        for number in random_selected_numbers:
            f.write("%s\n" % number)

def format_date(numeric_date):
    # 숫자 형식의 날짜를 문자열로 변환
    str_date = str(numeric_date)

    # YYYY-MM-DD 형식으로 변환
    formatted_date = f"{str_date[:4]}-{str_date[4:6]}-{str_date[6:]}"

    return formatted_date
def precedents_preprocess_data(examples):
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
        final_text = reason_text
        # # final_text = re.split("대법원\s+|판사\s+|대법원판사\s+|대법관\s+", reason_text, maxsplit=1)
        # final_text = re.split("\n\s*대법원\s+|판사\s+|대법원판사\s+|대법관\s+", reason_text)
        # if len(final_text) > 1:
        #     print()
        # else:
        #     print()
        # final_text = final_text[0] if final_text else ""
        # if len(final_text) <= 60:
        #     print(final_text)
        #     print('========' * 10)
        if final_text:
            combined_parts.append(f'이유: {final_text}')

    # if decision:
    #     combined_parts.append(f'판시사항: {decision}')
    # if summary:
    #     combined_parts.append(f'판결요지: {summary}')
    # if laws:
    #     combined_parts.append(f'참조조문: {laws}')
    #
    # if precedents:
    #     precedents += ', ' + examples['법원명'] + " " + format_date(examples['선고일자']) + " " + examples['선고'] + " " + examples['사건번호'] + " " + '판결'
    # else:
    #     precedents += examples['법원명'] + " " + format_date(examples['선고일자']) + " " + examples['선고'] + " " + examples[
    #         '사건번호'] + " " + '판결'
    # combined_parts.append(f'참조판례: {precedents}')


    combined_text = "\n".join(combined_parts)

    return {'input_text': combined_text}

#허깅페이스 precedents 데이터에 랜덤 샘플을 추출하여 txt 파일로 저장한다.
def precedents_sample_txt(dataset, test_case_file):
    civil_cases_not_with_wage_excluded = dataset.filter(
        lambda x: x['사건종류명'] == '민사' and (x['사건명'] is None or '임금' not in x['사건명'])
    )

    # civil_cases_not_with_wage_excluded에서 랜덤으로 100개 샘플 선택
    random_samples = civil_cases_not_with_wage_excluded.shuffle().select(range(100))
    #
    # 테스트 데이터의 판례번호를 추출
    test_case_numbers = random_samples['판례정보일련번호']

    # 판례번호를 파일에 저장
    with open(test_case_file, 'w') as f:
        for number in test_case_numbers:
            f.write("%s\n" % number)

def hugging_precedents():
    dataset_name = "joonhok-exo-ai/korean_law_open_data_precedents"
    dataset = load_dataset(dataset_name, cache_dir=custom_cache_dir, split="train")

    civil_cases_with_wage_excluded = dataset.filter(
        lambda x: x['사건종류명'] == '민사'
                  and
                  x['사건명'] is not None
                  and
                  '임금' in x['사건명']
        # and
        # (str(x['판례정보일련번호']) in test_case_numbers or (x['사건명'] is not None and '임금' in x['사건명']))
        # x['참조조문'] is not None
        # str(x['판례정보일련번호']) in test_case_numbers
        # str(x['판례정보일련번호']) not in test_case_numbers
    )

    # 최종 필터링된 데이터셋 생성
    civil_cases_with_wage_excluded, law_references = filter_with_reference(civil_cases_with_wage_excluded, 2)
    processed_dataset = civil_cases_with_wage_excluded.map(civil_cases_with_wage_excluded)
    return processed_dataset

def hugging_precedents():
    dataset_name = "joonhok-exo-ai/korean_law_open_data_precedents"
    dataset = load_dataset(dataset_name, cache_dir=custom_cache_dir, split="train")

    civil_cases_with_wage_excluded = dataset.filter(
        lambda x: x['사건종류명'] == '민사'
                  # and
                  # x['사건명'] is not None
                  # and
                  # '임금' in x['사건명']
        # and
        # (str(x['판례정보일련번호']) in test_case_numbers or (x['사건명'] is not None and '임금' in x['사건명']))
        # x['참조조문'] is not None
        # str(x['판례정보일련번호']) in test_case_numbers
        # str(x['판례정보일련번호']) not in test_case_numbers
    )

    # 최종 필터링된 데이터셋 생성
    civil_cases_with_wage_excluded, law_references = filter_with_reference(civil_cases_with_wage_excluded, 2)
    processed_dataset = civil_cases_with_wage_excluded.map(precedents_preprocess_data)
    return processed_dataset

def ko_wikidata_QA(sample_num):
    dataset = load_dataset('maywell/ko_wikidata_QA', cache_dir=custom_cache_dir, split="train")
    random_samples = dataset.select(range(sample_num))
    qa_dataset = random_samples.map(ko_wikidata_QA_preprocess_data)
    return qa_dataset
def ko_wikidata_QA_preprocess_data(examples):
    # '참조조문'이 None이면 빈 문자열로 처리
    instruction = examples['instruction'] if examples['instruction'] is not None else ""
    output = examples['output'] if examples['output'] is not None else ""

    combined_text = f"{instruction}\n".join(output)

    # output = examples['text'] if examples['text'] is not None else ""
    return {'input_text': combined_text}

def korean_textbooks(sample_num, subset):
    dataset = load_dataset('maywell/korean_textbooks', subset, cache_dir=custom_cache_dir, split="train")
    random_samples = dataset.select(range(sample_num))
    textbooks_dataset = random_samples.map(korean_textbooks_preprocess_data)
    return textbooks_dataset
def korean_textbooks_preprocess_data(examples):
    output = examples['text'] if examples['text'] is not None else ""
    return {'input_text': output}


def ai_hub_precedents():
    # 해당 경로 설정
    directory_path = r"D:\019.법률, 규정 (판결서, 약관 등) 텍스트 분석 데이터\01.데이터\1.Training\라벨링데이터_230510_add\TL_1.판결문\TL_1.판결문\1.Training\라벨링데이터\TL_1.판결문\01.민사"

    # 'courtDcss' 값을 저장할 리스트 초기화
    courtDcss_values = []

    # directory_path 아래의 모든 폴더 및 파일 순회
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            # .json 파일만 처리
            if file.endswith(".json"):
                full_path = os.path.join(root, file)
                # 파일 열기 및 JSON 로드
                with open(full_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 'courtDcss' 값 추출 및 리스트에 추가
                    # 'info', 'dcss', 'courtDcss' 경로에 유효한 데이터가 있는지 확인
                    if 'dcss' in data and 'courtDcss' in data['dcss']:
                        if data['dcss']['courtDcss']:
                            courtDcss_values.append({'input_text': data['dcss']['courtDcss'][0]})

    df = pd.DataFrame(courtDcss_values)
    return Dataset.from_pandas(df)

def law_qa_datas():
    results = []
    full_path = train_data_root + r'\law_train_data\law_qa_data.json'

    with open(full_path, 'r', encoding='utf-8') as f:
        datas = json.load(f)
        for data in datas:
            results.append({'input_text': data['question'] + '\n' + data['answer']})

    df = pd.DataFrame(results)
    return Dataset.from_pandas(df)
def law_translate_datas():
    results = []
    full_path = train_data_root + r'\law_train_data\law_translate_data.json'

    with open(full_path, 'r', encoding='utf-8') as f:
        datas = json.load(f)
        for data in datas:
            results.append({'input_text': data['text']})

    df = pd.DataFrame(results)
    return Dataset.from_pandas(df)
def law_translate_pdf():
    # PDF 파일 경로
    pdf_path = r'C:\Users\tyflow\Downloads\2023_상반기_법령해석사례집(상).pdf'

    data = []
    extracted_text = ''
    chapter_started = False
    pattern_bigo = r'\n\d+\).+?(?=\n\d+\)|$)'
    pattern_list = r'^\d+\n･\n'
    # PDF 파일 열기
    with pdfplumber.open(pdf_path) as pdf:
        # PDF의 각 페이지를 순회
        for page in pdf.pages:
            # 현재 페이지의 텍스트 추출
            text = page.extract_text()
            if text:  # 텍스트가 있는 경우에만 추가
                if re.match(pattern_list, text) is None:
                    text = text.split('\n', maxsplit=1)[1]
                    if "1. 질의요지" in text:
                        chapter_started = True
                        if extracted_text:
                            extracted_text = re.sub(r'2\. 회답', '', extracted_text)
                            extracted_text = re.sub(r'3\. 이유', '', extracted_text)
                            data.append({'input_text': extracted_text})
                        extracted_text = re.sub(pattern_bigo, '', text.split('1. 질의요지')[1], flags=re.DOTALL)

                    elif chapter_started:
                        extracted_text += re.sub(pattern_bigo, '', text, flags=re.DOTALL)
                elif '편집ㆍ발행' in text:
                    extracted_text = re.sub(r'2\. 회답', '', extracted_text)
                    extracted_text = re.sub(r'3\. 이유', '', extracted_text)
                    data.append({'input_text': extracted_text})
    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def law_translate_href(start_page=1):
    base_url = "https://www.moleg.go.kr/lawinfo/nwLwAnList.mo"
    links = []

    while True:
        # 현재 페이지 URL 구성
        url = f"{base_url}?mid=a10106020000&currentPage={start_page}&pageCnt=10&keyField=&keyWord=&sort=date"
        response = requests.get(url)

        # 요청이 성공적이지 않은 경우 반복 중단
        if response.status_code != 200:
            break

        # BeautifulSoup 객체 생성
        soup = BeautifulSoup(response.text, 'html.parser')

        # tbody 내의 모든 a 태그 찾기
        tbody = soup.find('tbody')
        if tbody:
            a_tags = tbody.find_all('a')
            for a in a_tags:
                href = a.get('href')
                if href:
                    links.append(href.replace('¤', '&curren'))

        # 링크가 없거나 특정 조건에 따라 반복 중단
        if not a_tags:
            break

        # 다음 페이지로
        start_page += 1

    with open('law_translate_links.json', 'w', encoding='utf-8') as file:
        json.dump(links, file, ensure_ascii=False, indent=4)

def get_custom_text(div):
    text_parts = []  # 최종 텍스트 조각들을 저장할 리스트
    for content in div.contents:  # div의 자식 요소들을 순회
        if isinstance(content, NavigableString):
            text_parts.append(content.strip())
        elif isinstance(content, Tag):
            if content.name == 'strong':  # 'strong' 태그인 경우
                text_parts.append(content.get_text(strip=True) + ' ')  # 텍스트 뒤에 띄어쓰기 추가
            else:
                text_parts.append(content.get_text(strip=True))
    return ' '.join(text_parts)  # 모든 텍스트 조각들을 공백으로 연결

def law_translate_crawling():
    # links.json 파일에서 링크들 읽기
    with open('law_translate_links.json', 'r', encoding='utf-8') as file:
        links = json.load(file)

    base_url = "https://www.moleg.go.kr/"
    extracted_data = []  # URL, 추출된 텍스트 및 추가 데이터를 저장할 리스트

    for link in tqdm(links):
        # 완전한 URL 생성
        full_url = base_url + link
        # URL로부터 HTML 내용 가져오기
        response = requests.get(full_url)
        # 응답이 성공적이면, BeautifulSoup으로 HTML 파싱
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            # 'tb_contents' 클래스를 가진 div 태그들 찾기
            divs = soup.find_all('div', class_='tb_contents')
            page_text = ''.join(get_custom_text(div) + '\n' for div in divs[:-1])

            # 특정 요소에서 추가 데이터 추출
            specific_element = soup.find('li', class_='date').find('span')
            data_text = specific_element.get_text(strip=True) if specific_element else 'Not found'

            # 현재 페이지의 URL, 텍스트 및 추가 데이터를 딕셔너리로 저장
            extracted_data.append({'url': full_url, 'text': re.sub(r'\s{2,}', ' ', page_text), 'data': data_text})

    # 추출된 데이터를 JSON 파일로 저장
    with open('law_translate_data.json', 'w', encoding='utf-8') as file:
        json.dump(extracted_data, file, ensure_ascii=False, indent=4)

def trim_text_after_warning(text):
    # '※ 주의' 문자열이 나오는 위치를 찾습니다.
    index = text.find('\n\n\n※ 주의')

    # 문자열이 발견되면, 해당 부분부터 모든 텍스트를 제거합니다.
    if index != -1:
        # '※ 주의' 문자열이 포함된 부분부터 시작하여 그 이전의 텍스트만 반환합니다.
        return text[:index]
    else:
        # '※ 주의' 문자열이 없는 경우, 원본 텍스트를 그대로 반환합니다.
        return text

def law_qa_crawling():
    base_url = "https://www.klac.or.kr/legalinfo/counselView.do"
    bNum = 1
    sNum = 1
    extracted_data = []

    # tqdm 객체 초기화. 전체 데이터 개수를 10037로 설정.
    pbar = tqdm(total=10037, desc='크롤링 진행률')

    while True:
        # bNum과 sNum을 사용하여 caseId를 생성
        caseId = f"case-{bNum:03}-{sNum:05}"

        # 완성된 URL
        url = f"{base_url}?pageIndex=8&folderId=000&caseId={caseId}&listNm=전체&searchCnd=0&searchWrd=&scdFolderId="

        # 페이지 요청 및 BeautifulSoup 객체 생성
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        dds = soup.find_all('dd')
        print('bNum: ' + str(bNum))
        print('sNum: ' + str(sNum))
        if dds[4].getText() != '':
            extracted_data.append({'url': url, 'gubun': dds[3].getText(), 'title': dds[4].getText(), 'question': dds[5].getText(), 'answer': trim_text_after_warning(dds[6].getText())})
            pbar.update(1)
            sNum += 1
        else:
            if sNum == 1:
                print("첫 번째 sNum에서 데이터 없음, 크롤링 종료.")
                break
            # 이전 bNum에서 데이터를 찾았으면 sNum을 리셋하고 bNum을 증가
            sNum = 1
            bNum += 1

    with open('law_qa_data.json', 'w', encoding='utf-8') as file:
        json.dump(extracted_data, file, ensure_ascii=False, indent=4)
