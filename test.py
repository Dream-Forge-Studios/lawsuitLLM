from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import PeftModel
import torch
def stream(user_prompt):
    runtimeFlag = "cuda:0"
    # system_prompt = 'The conversation between Human and AI assisatance named Gathnex\n'
    B_INST, E_INST = "[INST]", "[/INST]"

    # prompt = f"{system_prompt}{B_INST}{user_prompt.strip()}\n{E_INST}"
    prompt = f"{B_INST}{user_prompt.strip()}\n{E_INST}"

    inputs = tokenizer([prompt], return_tensors="pt").to(runtimeFlag)

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generated_ids = model.generate(**inputs, streamer=streamer, max_new_tokens=4096)

    decoded = tokenizer.batch_decode(generated_ids)
    # 질문과 답변 분리
    question, answer = decoded[0].split("[/INST]")

    print(f"질문:", question.replace("[INST]", "").strip())
    print("\n답변:", answer.strip())


model = AutoModelForCausalLM.from_pretrained(
    "/data/Synatra-7B-v0.3-dpo", low_cpu_mem_usage=True,
    return_dict=True,torch_dtype=torch.bfloat16,
    device_map= {"": 0})

# LORA로 미세조정한 모델을 로드합니다.
new_model = "/data/llm/lawsuit-7B-civil-wage-a"
# new_model = r"D:\lawsuit-7B-easylaw_kr-e3"
model = PeftModel.from_pretrained(model, new_model)

# tokenizer = AutoTokenizer.from_pretrained("/data/llm/Synatra-7B-v0.3-dpo")
tokenizer = AutoTokenizer.from_pretrained("/data/Synatra-7B-v0.3-dpo")

stream("판결 요지: '지방자치단체의 장의 보조기관인 건설과장의 요청에 의하여 노무자가 일정한 노무를 제공한 경우, 이 계약의 체결이 위 과장의 권한에 속하는 한 이는 유효하게 성립한 것으로 위 과장이 노무자를 고용함에 있어서 행정상의 내부적인 절차를 경유하지 않았다고 할지라도 그것은 행정상의 내부관계에 불과하여 이로써 위 노무계약이 위법이거나 무효라고 단정할 수 없다.' "
       "참조 조문: 민법 제655조 "
       "판결 요지: '근로자가 유급휴일에 근로한 경우에는 유급으로서 당연히 지급되는 임금과 그 유급휴일의 근로에 대한 소정의 통상임금을 포함한 임금을 지급하여야 된다.' "
       "참조 조문: 근로기준법 제47조 "
       "판결 요지: '피고회사가 근로자들을 월남국으로 파병시킬 당시 미국인회사로부터 앞으로 일년간 공사시공을 함에 상당한 하도급을 받았으나 그 후의 사정변경으로 하도급받은 작업량이 줄어들게 되었다는 사실로써는 민법 제661조나 본조 제1항 단서 소정 부득이한 사유라 할 수 없으므로 해고기간의 정함이 있는 위 근로자들과의 본건 해고계약은 30일 전의 예고로도 해지시킬 수 없다.' "
       "참조 조문:")

# class BankAccount:
#     def __init__(self, account: str):
#         self.money = 0
#         try:
#             if len(self.account) != 6:
#                 raise Exception('올바른 계좌번호를 입력하세요.')
#             else:
#                 self.account = account
#         except Exception as e:
#             print(e)
#     def deposit(self, money: int):
#         try:
#             if money < 0:
#                 raise Exception('음수값은 처리될 수 없습니다. 다시 시도해주세요.')
#             else:
#                 self.money += money
#         except Exception as e:
#             print(e)
#
#     def withdraw(self, money: int):
#         try:
#             if self.money < money:
#                 raise Exception('잔액이 충분하지 않습니다. 확인 후 다시 시도해주세요.')
#             else:
#                 self.money -= money
#         except Exception as e:
#             print(e)
#
# def transfer(a: BankAccount, b: BankAccount, money):
#     a.withdraw(money)
#     b.deposit(money)
#
# def transaction_processing(account, data):
#     for key, value in account.items():
#         if data[0] == 'deposit':
#             value(key)
#             value.deposit(data[1])
#         elif data[0] == 'withdraw':
#             value(key)
#             value.withdraw(data[1])
#         elif data[0] == 'transfer':
#             a = value(key)
#             b = BankAccount(data[1])
#             transfer(a, b, data[2])