from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained("/data/data/lawsuit-7B-easylaw_kr-v0.1/adapter_model.safetensors")
tokenizer = AutoTokenizer.from_pretrained("/data/data/lawsuit-7B-easylaw_kr-v0.1/adapter_model.safetensors")

messages = [
    {"role": "user", "content": "신호를 어겨서 벌점을 받았는데 이 벌점은 계속 유지되는거야?"},
]

encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

model_inputs = encodeds.to(device)
model.to(device)

generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded[0])