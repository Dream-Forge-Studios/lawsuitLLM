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
    base_model = "/data/Synatra-7B-v0.3-dpo"
    # Windows 사용자의 경우 경로
    base_model = r"D:\Synatra-7B-v0.3-dpo"

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

        generated_ids = final_model.generate(**inputs, streamer=streamer, max_new_tokens=400)
        # generated_ids = final_model.generate(**inputs, streamer=streamer, max_new_tokens=4000)

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
        '근로기준법 제18조': [
            '마케팅 회사에서 일하는 단시간 근로자 리나는 주 14시간 근무합니다. 리나는 같은 부서의 통상 근로자들과 비교해 현저히 낮은 시간당 임금을 받고 있습니다.',
            '소프트웨어 개발자로 일하는 단시간 근로자 진수는 계약상 주 10시간 근무합니다. 진수는 연차 유급휴가를 요구했지만, 회사는 그의 근로시간이 15시간 미만이라는 이유로 거부했습니다.',
            '파트타임 근로자인 현지는 주 12시간 근무하며, 그녀의 근로조건 결정 시, 회사는 통상 근로자와의 비교 없이 임의로 결정했습니다.'
        ],
        '민법 제741조': [
            '개인 A는 이웃 B의 빈집을 무단으로 사용하여 홈스테이 사업을 운영했습니다. B는 이 사실을 알고 A에게 사업으로 얻은 이익의 반환을 요구했습니다.',
            '회사 C는 경쟁 회사 D의 영업 비밀을 불법적으로 사용하여 이익을 얻었습니다. D는 C에게 이로 인해 얻은 이익의 반환을 요구했습니다.',
            '개인 E는 친구 F의 차량을 무단으로 사용하여 택시 서비스를 제공했습니다. F는 차량 사용으로 인한 마모 및 E가 얻은 수익에 대한 보상을 요구합니다.'
        ],
        '외국인근로자의 고용 등에 관한 법률 제22조': [
            '회사 A에서 근무하는 외국인 근로자 B는 같은 업무를 수행하는 내국인 동료들보다 낮은 임금을 받고 있습니다. B는 이에 대해 문제를 제기합니다.',
            '외국인 근로자 C는 회사 D에서 근무하면서, 회사 내에서 제공되는 교육 프로그램 참여 기회가 내국인 근로자들에 비해 제한적입니다. C는 이로 인해 자신의 업무 스킬 향상에 불이익을 받고 있다고 느낍니다.',
            '외국인 근로자 E는 그의 출신 국가 때문에 회사 F의 사회적 활동에서 배제되고 있습니다. E는 이를 차별적인 처우로 여기고 이의를 제기합니다.'
        ],
        '공공기관의 운영에 관한 법률 제48조 제1항': [
            '공기업 A는 최근 준정부기관으로 지정되었습니다. 이에 따라, 기획예산처장관은 A의 지정 첫 해에 경영실적 평가를 계획하고 있습니다. 그러나 A는 법률에 따라 첫 해에는 경영실적 평가를 받지 않아도 된다고 주장합니다.',
            '준정부기관 B는 계약 이행에 관한 보고서와 경영목표 및 경영실적보고서를 제출했습니다. 그러나 기획예산처장관은 이를 기초로 한 경영실적 평가를 실시하지 않았습니다. B는 이에 대한 설명을 요구합니다.',
            '공기업 C는 제31조제3항 및 제4항에 따른 계약의 이행 보고서와 제46조에 따른 경영목표 및 경영실적보고서를 제출하지 않았습니다. 이에 기획예산처장관은 C의 경영실적 평가를 보류하였습니다.'
        ],
        '근로기준법 제79조': [
            '직원 A는 업무 중 다친 후 요양 중입니다. 회사 B는 A에게 요양 기간 동안의 평균임금을 전혀 지급하지 않았습니다. A는 자신의 권리를 주장하며 적절한 휴업보상을 요구합니다.',
            '직원 C는 업무상 부상으로 요양 중에 있으며, 이 기간 동안 일부 임금을 지급받았습니다. 그러나 회사 D는 이미 지급된 임금을 고려하지 않고 추가 휴업보상을 거부하고 있습니다.',
            '직원 E는 업무상 질병으로 장기 요양 중입니다. 그러나 회사 F는 휴업보상의 시기와 관련된 대통령령을 몰라 휴업보상 지급을 지연시키고 있습니다.'
        ],
    }

    results = {}
    final_model = PeftModel.from_pretrained(model, f"D:\lawsuit-7B\{new_model}")

    for key in testData.keys():
        results[key] = []

    for key in testData.keys():
        for prompt in testData[key]:
            # stream("'" + prompt + "'와 관련된 법은?", key)
            stream("'" + prompt + "'이에 대한 법적 판단은?", key)



    # JSON 파일 경로
    file_path = f"results/{new_model}_results.json"

    # JSON 파일에 데이터 쓰기
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()