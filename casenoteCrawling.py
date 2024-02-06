import requests
from time import sleep
from bs4 import BeautifulSoup
import logging
import requests
import uuid
import time
import json
import os
import re

api_url = 'https://qv7j3dq3l2.apigw.ntruss.com/custom/v1/27629/6151372a0ad6c8086a38e159917dfdef42047aa2c9e7e6405befb19e32fe851d/general'
secret_key = 'dGZXUmx2eVdzZWRkSVNxd0VmQ2FjTk5DV1l1ZFJtZ0c='
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
def naver_ocr(image_url):
    # 이미지를 다운로드하여 저장할 로컬 파일 경로
    local_image_path = "image.jpg"

    # 이미지 다운로드
    response = requests.get(image_url)

    # 이미지를 로컬 파일로 저장
    with open(local_image_path, "wb") as file:
        file.write(response.content)


    request_json = {
        'images': [
            {
                'format': 'jpg',
                'name': 'demo'
            }
        ],
        'requestId': str(uuid.uuid4()),
        'version': 'V2',
        'timestamp': int(round(time.time() * 1000))
    }

    payload = {'message': json.dumps(request_json).encode('UTF-8')}
    files = [
      ('file', open(local_image_path,'rb'))
    ]
    ocr_headers = {
      'X-OCR-SECRET': secret_key
    }
    while True:
        try:
            response = requests.request("POST", api_url, headers=ocr_headers, data = payload, files = files)
            break
        except:
            time.sleep(1)
            pass

    ocr_text = ''
    jsonData = json.loads(response.text)
    if 'images' in jsonData and isinstance(jsonData['images'], list) and len(jsonData['images']) > 0:
        for a in json.loads(response.text)['images'][0]['fields']:
            ocr_text += ' ' + a['inferText']
    return ocr_text

def setup_logger():
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


hrefs = []
results = []
def crawl_hrefs(url_queue, file_name):
    page_num = 1
    while True:
    # while page_num <= 10:
        print(page_num)
        try:
            url = url_queue + str(page_num)
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                items = soup.find_all('div', class_='searched-item')
                for item in items:
                    a_tag = item.find('a')
                    if item and a_tag.has_attr('href'):
                        hrefs.append(a_tag['href'])
                page_num += 1
            else:
                logging.warning(f"Error accessing {url}: {response.status_code}")
                break
        except Exception as e:
            logging.error(f"Exception occurred for {url}: {e}")

        sleep(1)  # 서버 부하를 줄이기 위해 1초 간격으로 요청

    with open(file_name, "w") as file:
        for item in hrefs:
            file.write(item + "\n")

def crawl_data(input_file, output_file, option):
    line_count = sum(1 for _ in open(input_file))

    with open(input_file, "r") as file:
        for i, line in enumerate(file, start=1):
            # if i > 2:
            #     break
            if option == 1:
                results.append(one_two_caseLaw(line))
            elif option == 3:
                results.append(three_caseLaw(line))

            # 진행률 계산 및 출력
            progress = (i / line_count) * 100
            print(f"Current Progress: {progress:.2f}%")

            sleep(1)  # 서버 부하를 줄이기 위해 1초 간격으로 요청

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=4)

def three_caseLaw(line):
    try:
        data = {}
        url = 'https://casenote.kr' + line.strip()
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            data['url'] = url
            # 사건 제목
            data['cn-case-title'] = soup.find('div', class_='cn-case-title').find('h1').text.strip()

            # 이유
            data['reason'] = []
            data['law'] = []
            data['caseLaw'] = []

            #판시사항
            issueDiv = soup.find('div', class_='issue')
            if issueDiv:
                data['reason'].append(issueDiv.text.strip())

            #판결요지
            summaryDiv = soup.find('div', class_='summary')
            if summaryDiv:
                data['reason'].append(summaryDiv.text.strip())

            #참조조문
            reflawsDiv = soup.find('div',class_='reflaws')
            if reflawsDiv:
                for a_tag in reflawsDiv.find_all('a'):
                    lawData = {}
                    href = a_tag['href']
                    lawCrwalData = lawCrwal(href)
                    if lawCrwalData:
                        lawData['text'], lawData['date'], lawData['href'] = lawCrwalData
                        data['law'].append(lawData)
                        reflawsDiv.find('a').replace_with('')
                cleaned_text = re.sub(r'\[\d+\]', '', reflawsDiv.text)
                cleaned_text = cleaned_text.replace('/',',')
                for text in cleaned_text.split(','):
                    text = text.strip()
                    if text:
                        lawData['text'] = text
                        data['law'].append(lawData)

            #참조판례
            refcasesDiv = soup.find('div', class_='refcases')
            if refcasesDiv:
                for a_tag in refcasesDiv.find_all('a'):
                    lawCaseData = {}
                    lawCaseData['text'] = a_tag.text
                    lawCaseData['href'] = a_tag['href']
                    data['caseLaw'].append(lawCaseData)

            # 주문
            checker = False
            for q_tag in soup.find_all('p'):
                if 'title' in q_tag.get('class', []):
                    if not checker and data.get('request'):
                        data['judgment'] = temp
                        break
                    checker = not checker
                    temp = []
                elif checker:
                    temp.append(q_tag.text)


            pattern1 = r'^(\d+|[가-힣]+)\.'
            pattern2 = r'^\((\d+|[가-힣]+)\)$'
            for p_tag in soup.find('div', class_='reason').find_all('p'):
                p_text = p_tag.text.strip()
                if bool(re.match(pattern1, p_text)) or bool(re.match(pattern2, p_text)):
                    data['reason'].append(temp)
                    temp = [p_text]
                else:
                    for sup_tag in p_tag.find_all('sup'):
                        name = sup_tag.find('a')['name']
                        checker = False
                        for small_tag in soup.find_all('small'):
                            if small_tag.find('a')['href'] == '#' + name:
                                fix_text = '(' + small_tag.text[2:].strip() + ')'
                                checker = True
                                break
                        if checker:
                            p_tag.find('sup').replace_with(fix_text)
                            p_text = p_tag.text.strip()

                    img_tag = p_tag.find('img')
                    if img_tag and 'src' in img_tag.attrs:
                        p_text = naver_ocr('https://casenote.kr' + img_tag['src'])

                    # 관련 법률
                    for a_tag in p_tag.find_all('a'):
                        if a_tag.text[-2:] == '판결':
                            lawCaseData = {}
                            lawCaseData['text'] = a_tag.text
                            lawCaseData['href'] = a_tag['href']
                            data['caseLaw'].append(lawCaseData)
                        else:
                            lawData = {}
                            href = a_tag['href']
                            lawCrwalData = lawCrwal(href)
                            if isinstance(lawCrwalData, tuple):
                                lawData['text'], lawData['date'], lawData['href'] = lawCrwalData
                                data['law'].append(lawData)
                            else:
                                pass
                    temp.append(p_text)

            if temp:
                data['reason'].append(temp)

        else:
            logging.warning(f"Error accessing {url}: {response.status_code}")
    except Exception as e:
        logging.error(f"Exception occurred for {url}: {e}")

def one_two_caseLaw(line):
    # try:
        data = {}
        url = 'https://casenote.kr' + line.strip()
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            data['url'] = url
            # 사건 제목
            data['cn-case-title'] = soup.find('div', class_='cn-case-title').find('h1').text.strip()

            for tr_tag in soup.find('table', class_='table-in-abstract').find_all('tr'):
                textList = tr_tag.text.strip().split('\n')
                checkerText = textList[0].strip().replace(' ','')

                if checkerText == '사건':
                    if not data.get('case'):
                        data['case'] = []
                    for text in textList[1:]:
                        if text:
                            data['case'].append(text.strip().replace('\xa0', ' '))

                elif checkerText[:2] == '원고':
                    if not data.get('plaintiffs'):
                        data['plaintiffs'] = []
                    for text in textList[1:]:
                        if text:
                            data['plaintiffs'].append(text.strip().replace('\xa0', ' '))

                elif checkerText[:2] == '피고':
                    if not data.get('defendant'):
                        data['defendant'] = []
                    for text in textList[1:]:
                        if text:
                            data['defendant'].append(text.strip().replace('\xa0', ' '))

                elif checkerText == '변론종결':
                    if not data.get('conclusion_date'):
                        data['conclusion_date'] = []
                    for text in textList[1:]:
                        if text:
                            data['conclusion_date'].append(text.strip().replace('\xa0', ' '))

                elif checkerText == '판결선고':
                    if not data.get('judgment_date'):
                        data['judgment_date'] = []
                    for text in textList[1:]:
                        if text:
                            data['judgment_date'].append(text.strip().replace('\xa0', ' '))

            # 주문, 청구취지 (list)
            qs = ['judgment', 'request']
            checker = False
            for q_tag in soup.find_all('p'):
                if 'title' in q_tag.get('class', []):
                    if data.get('judgment'):
                        data[qs[i]] = temp
                        break
                    checker = True
                    temp = []
                    i = 0
                elif 'claim-title' in q_tag.get('class', []):
                    data[qs[i]] = temp
                    temp = []
                    i = 1
                elif checker:
                    temp.append(q_tag.text)
            temp = []
            # 이유
            data['reason'] = {
                'fcat': [],
                'claim': [],
                'judgement': [],
                'conclusion': [],
            }
            data['law'] = []
            data['caseLaw'] = []
            pattern1 = r'^(\d+|[가-힣]+)\.' #1., 가.
            pattern2 = r'^\((\d+|[가-힣]+)\)$' #(1), (가)
            pattern3 = r'^\[(\d+|[가-힣]+)+\]' #[1], [가]
            pattern4 = r'^(\d+|[가-힣]+)\)' #1), 가)
            checker = False
            key = ''
            for p_tag in soup.find('div', class_='reason').find_all('p'):
                p_text = p_tag.text.strip()
                if bool(re.match(pattern1, p_text)) or bool(re.match(pattern2, p_text)) or bool(re.match(pattern3, p_text)) or bool(re.match(pattern4, p_text)):
                    if not checker:
                        checker = True

                    if '사실' in p_text.replace(' ','') or '개요' in p_text.replace(' ',''):
                        if temp:
                            data['reason'][key].append(temp)
                        key = 'fcat'
                        temp = []
                    elif '주장' in p_text.replace(' ','') or '요지' in p_text.replace(' ',''):
                        if temp:
                            data['reason'][key].append(temp)
                        key = 'claim'
                        temp = []
                    elif '판단' in p_text.replace(' ','') or '법리' in p_text.replace(' ',''):
                        if temp:
                            data['reason'][key].append(temp)
                        key = 'judgement'
                        temp = []
                    elif '결론' in p_text.replace(' ',''):
                        if temp:
                            data['reason'][key].append(temp)
                        key = 'conclusion'
                        temp = []
                    else:
                        temp.append(p_text)
                else:
                    if not checker:
                        pass
                    for sup_tag in p_tag.find_all('sup'):
                        name = sup_tag.find('a')['name']
                        checker = False
                        for small_tag in soup.find_all('small'):
                            if small_tag.find('a')['href'] == '#' + name:
                                fix_text = '(' + small_tag.text[2:].strip() + ')'
                                checker = True
                                break
                        if checker:
                            p_tag.find('sup').replace_with(fix_text)
                            p_text = p_tag.text.strip()

                    img_tag = p_tag.find('img')
                    if img_tag and 'src' in img_tag.attrs:
                        p_text = naver_ocr('https://casenote.kr' + img_tag['src'])

                    # 관련 법률
                    for a_tag in p_tag.find_all('a'):
                        if a_tag.text[-2:] == '판결':
                            lawCaseData = {}
                            lawCaseData['text'] = a_tag.text
                            lawCaseData['href'] = a_tag['href']
                            data['caseLaw'].append(lawCaseData)
                        else:
                            lawData = {}
                            href = a_tag['href']
                            lawCrwalData = lawCrwal(href)
                            if isinstance(lawCrwalData, tuple):
                                lawData['text'], lawData['date'], lawData['href'] = lawCrwalData
                                data['law'].append(lawData)
                            else:
                                pass
                    temp.append(p_text)

            if temp:
                if key:
                    data['reason'][key].append(temp)
                else:
                    data['reason']['judgement'].append(temp)
            return data
        else:
            logging.warning(f"Error accessing {url}: {response.status_code}")
    # except Exception as e:
    #     logging.error(f"Exception occurred for {url}: {e}")

def lawCrwal(href):
    response2 = requests.get('https://casenote.kr' + href, headers=headers)
    if response2.status_code == 200:
        soup2 = BeautifulSoup(response2.text, 'html.parser')
        try:
            return soup2.find('h2', class_='title').text + soup2.find('div', class_='law_article').text, soup2.find('p', class_='abstract').text, href
        except:
            return False
    else:
        return False

# 사용 예시
# url_queue = 'https://casenote.kr/%EC%84%9C%EC%9A%B8%EC%A4%91%EC%95%99%EC%A7%80%EB%B0%A9%EB%B2%95%EC%9B%90/2020%EA%B0%80%ED%95%A9575470#ref_1'
url_queue = 'https://casenote.kr/search/?q=%EA%B0%80%EB%93%B1%EA%B8%B0%EB%A7%90%EC%86%8C&sort=0&period=1&court=2&case=0&partial=0&oc=1&page='
setup_logger()
# crawl_hrefs(url_queue, 'D:\판례학습데이터\가등기말소\hrefs\가등기말소_지방법원_민사_원고패_최근3년_hrefs.txt')
crawl_data('D:\판례학습데이터\가등기말소\hrefs\가등기말소_지방법원_민사_원고패_최근3년_hrefs.txt', r"D:\판례학습데이터\가등기말소\rawJson\가등기말소_지방법원_민사_원고패_최근3년.json", 1)
