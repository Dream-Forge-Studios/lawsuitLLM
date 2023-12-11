from transformers import AutoTokenizer
from peft import PeftModel, PeftConfig

device = "cuda" # the device to load the model onto

# 먼저 원래 모델의 구성을 불러옵니다.
config = PeftConfig.from_pretrained("maywell/Synatra-7B-v0.3-dpo")

# LORA로 미세조정한 모델을 로드합니다.
model_path = "/data/data/lawsuit-7B-easylaw_kr-v0.1"
model = PeftModel.from_pretrained(model_path, config=config)

tokenizer = AutoTokenizer.from_pretrained("maywell/Synatra-7B-v0.3-dpo")

messages = [
    {"role": "user", "content": "신호를 어겨서 벌점을 받았는데 이거는 평생가는거야?"}
]

encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

model_inputs = encodeds.to(device)
model.to(device)

generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded[0])
