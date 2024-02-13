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

    generated_ids = model.generate(**inputs, streamer=streamer, max_new_tokens=200)

    decoded = tokenizer.batch_decode(generated_ids)
    # 질문과 답변 분리
    question, answer = decoded[0].split("[/INST]")

    print(f"질문:", question.replace("[INST]", "").strip())
    print("\n답변:", answer.strip())


model = AutoModelForCausalLM.from_pretrained(
    "maywell/Synatra-7B-v0.3-dpo", low_cpu_mem_usage=True,
    return_dict=True,torch_dtype=torch.bfloat16,
    device_map= {"": 0})

# LORA로 미세조정한 모델을 로드합니다.
# new_model = "/data/llm/lawsuit-7B-easylaw_kr-e3"
# new_model = r"D:\lawsuit-7B-easylaw_kr-e3"
# model = PeftModel.from_pretrained(model, new_model)

# tokenizer = AutoTokenizer.from_pretrained("/data/llm/Synatra-7B-v0.3-dpo")
tokenizer = AutoTokenizer.from_pretrained("maywell/Synatra-7B-v0.3-dpo")

stream("신호를 어겨서 벌점을 받았는데 이거는 평생가는거야?")

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