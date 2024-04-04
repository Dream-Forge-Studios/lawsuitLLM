# -*- coding: utf-8 -*-
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, BitsAndBytesConfig
from peft import PeftModel
import torch
import json
import click

# 커맨드 라인 인터페이스 생성
@click.command()
@click.option('--new_model', help='새로운 모델의 이름을 입력합니다.')
def main(new_model):
    base_model = "/data/llm/Synatra-7B-v0.3-dpo"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)


    def stream(user_prompt, key):
        runtimeFlag = "cuda:0"
        B_INST, E_INST = "[INST]", "[/INST]"

        prompt = f"{B_INST}{user_prompt.strip()}\n{E_INST}"

        inputs = tokenizer([prompt], return_tensors="pt").to(runtimeFlag)

        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        generated_ids = model.generate(**inputs, streamer=streamer, max_new_tokens=400)
        # generated_ids = model.generate(**inputs, streamer=streamer, max_new_tokens=1000)

        decoded = tokenizer.batch_decode(generated_ids)

        # '[/INST]'를 기준으로 문자열 분리 및 예외 처리
        split_text = decoded[0].split("[/INST]")
        if len(split_text) >= 2:
            # 첫 번째 '[/INST]' 앞 부분을 질문으로, 첫 번째 '[/INST]' 뒷 부분을 답변으로 가정
            question = split_text[0] + "[/INST]"
            answer = "[/INST]".join(split_text[1:])  # 만약 여러 개의 [/INST]가 있다면, 나머지는 답변으로 간주
        else:
            # 예외 처리: 분리된 텍스트가 기대한 형태가 아닌 경우
            question = "[INST] 분리 오류"
            answer = "답변을 생성할 수 없습니다."

        cleaned_question = question.replace("[INST]", "").strip()
        cleaned_answer = answer.strip()
        results[key].append({'질문': cleaned_question, '답변': cleaned_answer})
        print(f"질문:", cleaned_question)
        print("\n답변:", cleaned_answer)


    testData = {
        '제1장': [
            "이 규정의 목적은 무엇인가요?",
            "상벌에 관한 규정의 적용 범위는 어떻게 되나요?",
            "포상을 행할 때 필요한 절차는 무엇인가요?",
            "징계를 결정하는 과정에서 어떤 단계를 거치나요?",
            "상벌 기록은 어디에 보관되며, 어떤 경우에 학적부 기재사항의 삭제가 가능한가요?",
            "동일한 사항에 대해 이중으로 상벌을 할 수 있나요?",
            "징계심의 과정에서 학생의 의견을 듣기 위한 조치는 무엇인가요?",
            "포상과 징계의 통보 방법은 공개를 원칙으로 하나요?",
            "졸업사정을 위한 교수회의에서 졸업예정자 중 벌을 받은 사실이 있는 학생의 기록을 삭제할 수 있는 기준은 무엇인가요?",
            "포상과 징계결과는 누구에게 어떻게 통보되나요?"
        ],
        '제2장': [
              "강원대학교의 졸업포상을 수여받을 수 있는 학생은 어떤 조건을 만족해야 하나요?",
              "재학 중 징계처분을 받은 학생은 졸업포상을 수여받을 수 있는지 여부를 설명해 주세요.",
              "강원대학교에서 수여하는 우등상의 대상은 누구인가요?",
              "우수학술연구상은 어떤 기준으로 수여되나요?",
              "강원대학교에서는 어떤 유형의 특별포상을 수여하나요?",
              "포상 대상자 추천 시 필요한 서류는 무엇인가요?",
              "졸업포상의 추천 기한은 언제까지인가요?",
              "졸업식에서 시상되는 상의 종류는 무엇인가요?",
              "입학포상의 시상 시기는 언제 인가요?",
              "경우에 따라 외부인의 상을 접수하여 시상할 수 있나요?"
        ],
        '제3장': [
            '강원대학교 학생 상벌규정에 따르면, 징계의 종류에는 어떤 것들이 있나요?',
            '근신 징계의 최소 및 최대 기간은 각각 몇 일인가요?',
            '유기정학 징계의 기간은 최소 및 최대 몇 일로 정해져 있나요?',
            '무기정학 징계의 경우, 해제 가능 기간이 있나요? 있다면, 그 기간은 몇 일인가요?',
            '징계 발의 시, 어떤 서류들이 필요한가요?',
            '제명 처분을 제외한 징계를 받은 학생이 징계 기간 중 받아야 하는 상담은 누구로부터 받아야 하나요?',
            '유기정학 또는 무기정학 징계를 받은 학생은 징계 기간 중 몇 회 이상 상담을 받아야 하나요?',
            '근신 처분을 받은 학생에게 금지되는 활동은 무엇인가요?',
            '근신 이상의 처분을 받은 학생의 권리는 언제부터 언제까지 정지되나요?',
            '무기정학의 해제는 누구의 의견을 듣고 결정할 수 있나요?',
        ],
        '제4장': [
            '강원대학교 학생이 교내 환경을 더럽히거나 훼손한 경우 어떤 종류의 징계를 받을 수 있나요?',
            '정당한 공고물을 무단으로 제거한 행위에 대한 강원대학교의 징계 규정은 무엇인가요?',
            '강원대학교에서는 학생들이 교내외에서 금지된 장소에 출입했을 때 어떤 처벌을 할 수 있나요?',
            '교직원의 지시에 불응하는 행위에 대해 강원대학교는 어떤 방식으로 징계할 수 있나요?',
            '강원대학교에서는 허가받지 않은 유인물, 영상물의 소지나 배포에 대해 어떻게 대응하나요?',
            '강원대학교 규정상 집단행위로 수업을 방해한 학생에게 어떤 징계가 가능한가요?',
            '강원대학교 학생이 수험부정행위를 했을 때, 어떤 처벌을 받게 되나요?',
            '타인의 답안을 보거나 받아쓴 행위에 대해 강원대학교에서는 어떤 조치를 취하나요?',
            '강원대학교에서 부정행위를 한 학생의 해당 과목 성적 처리 방법은 무엇인가요?',
            '강원대학교에서 수험 부정행위를 적발한 경우 감독관의 의무는 무엇인가요?',
        ],
    }

    results = {}
    model = PeftModel.from_pretrained(model, f"/data/docLLM/{new_model}")

    for key in testData.keys():
        results[key] = []

    for key in testData.keys():
        for prompt in testData[key]:
            stream(prompt, key)



    # JSON 파일 경로
    file_path = f"/data/doc_results/{new_model}_results.json"
    # file_path = f"results/lawsuit-7B-wage-100-c_results.json"

    # JSON 파일에 데이터 쓰기
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()