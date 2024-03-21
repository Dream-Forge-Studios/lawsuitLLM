# -*- coding: utf-8 -*-

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,DataCollatorForLanguageModeling,TrainingArguments, Trainer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import os, wandb
from datasets import concatenate_datasets
import torch
from utils import hugging_precedents, korean_textbooks, ai_hub_precedents, law_qa_datas, law_translate_datas
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler

def main():
    base_model = "/data/llm/Synatra-7B-v0.3-dpo"

    cutoff_len = 4096

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    print(os.environ.get("CUDA_VISIBLE_DEVICES", "Not Set"))

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
    )
    model.config.use_cache = False # silence the warnings. Please re-enable for inference!
    # model.config.pretraining_tp = 1

    # 그래디언트 체크포인팅 활성화
    model.gradient_checkpointing_enable()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.add_eos_token = True
    tokenizer.add_bos_token, tokenizer.add_eos_token

    processed_dataset = hugging_precedents()
    processed_dataset = processed_dataset.remove_columns([column_name for column_name in processed_dataset.column_names if column_name != 'input_text'])
    ai_hub_precedents_dataset = ai_hub_precedents()
    law_qa_dataset = law_qa_datas()
    law_translate_dataset = law_translate_datas()

    # qa_dataset = ko_wikidata_QA(300)
    textbooks_dataset = korean_textbooks(945, 'tiny-textbooks')

    # 48168개
    combined_dataset = concatenate_datasets([processed_dataset, ai_hub_precedents_dataset, law_qa_dataset, law_translate_dataset, textbooks_dataset])

    def tokenize_function(examples):
        return tokenizer(examples['input_text'], truncation=True, padding=True, max_length=cutoff_len)

    # 데이터셋 토큰화 적용
    tokenized_dataset = combined_dataset.map(tokenize_function, batched=True)

    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]
        )
    model = get_peft_model(model, peft_config)

    with open('/data/llm/wandbKey_js.txt', 'r') as file:
        wandb_key = file.read().strip()

    wandb.login(key = wandb_key)
    wandb.init(project='Fine tuning mistral 7B civil wage', job_type="training", anonymous="allow")

    # Training Arguments
    # Hyperparameters should beadjusted based on the hardware you using
    output_dir = "./results"
    num_train_epochs = 1
    gradient_accumulation_steps = 4
    save_steps = 1500
    logging_steps = 1
    learning_rate = 2e-4
    weight_decay = 0.001
    max_grad_norm = 0.3
    warmup_ratio = 0.3
    lr_scheduler_type = "cosine_with_restarts"

    data_collator = DataCollatorForLanguageModeling(
            tokenizer, mlm=False, pad_to_multiple_of=8, return_tensors="pt"
        )
    # DataLoader 설정
    batch_size = 16  # 배치 사이즈 설정
    train_dataloader = DataLoader(
        tokenized_dataset,  # 학습 데이터셋 사용
        shuffle=True,  # 데이터셋 셔플
        batch_size=batch_size,  # 배치 사이즈 설정
        collate_fn=data_collator,  # 데이터 콜레이터 사용
    )
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # Accelerator 초기화
    accelerator = Accelerator()

    # 모델, 옵티마이저, 데이터 로더 준비
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    num_epochs = 1
    num_training_steps = num_epochs * len(train_dataloader)
    # 웜업 스텝 계산
    num_warmup_steps = int(warmup_ratio * num_training_steps)
    lr_scheduler = get_scheduler(
        lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    for epoch in range(num_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            # 그래디언트 누적을 사용하여 일정 스텝마다 가중치를 업데이트합니다.
            if (step + 1) % gradient_accumulation_steps == 0:
                # 최대 그래디언트 노름을 사용하여 그래디언트 클리핑을 적용합니다.
                accelerator.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Weights & Biases에 학습 손실 등의 메트릭을 로깅합니다.
            if accelerator.is_main_process and (step + 1) % logging_steps == 0:
                wandb.log({"loss": loss.item(), "global_step": step + 1, "epoch": epoch + 1})

            # 주어진 스텝마다 체크포인트를 저장합니다.
            if (step + 1) % save_steps == 0 and step > 0:
                if accelerator.is_main_process:
                    model_save_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch + 1}_step_{step + 1}")
                    model.save_pretrained(model_save_path)
                    tokenizer.save_pretrained(model_save_path)

    if accelerator.is_main_process:
        # 최종 모델 저장
        model_save_path = os.path.join(output_dir, "final_model")
        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)

        # Weights & Biases 세션 종료
        wandb.finish()


if __name__ == "__main__":
    main()