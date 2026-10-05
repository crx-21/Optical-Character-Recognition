import torch
import os
from typing import Optional
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from transformers import TrainingArguments

def train_qwen_vl():
    # --- Configuration ---
    MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
    TRAIN_DATA_PATH = "amd-ocr-challenge/app/dataset/qwen_formatted/train_data.jsonl"
    VALID_DATA_PATH = "amd-ocr-challenge/app/dataset/qwen_formatted/valid_data.jsonl"
    OUTPUT_DIR = "amd-ocr-challenge/app/qwen_finetuned_lora"

    # ROCm Optimized Quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    print("Loading model and processor...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    # LoRA Configuration
    # Target the linear layers of the attention and MLP blocks
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Dataset loading
    train_dataset = load_dataset("json", data_files=TRAIN_DATA_PATH, split="train")
    valid_dataset = load_dataset("json", data_files=VALID_DATA_PATH, split="train")

    # SFTTrainer formatting function
    def formatting_prompts_func(example):
        # The dataset is already in conversation format, SFTTrainer handles this
        return example['messages']

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        bf16=True, # ROCm bfloat16
        optim="paged_adamw_32bit",
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        report_to="tensorboard",
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        peft_config=lora_config,
        # Qwen-2.5 VL specific: the dataset should match the chat template
        dataset_text_field="messages",
        max_seq_length=512,
    )

    print("Starting training...")
    trainer.train()

    # Save the final adapter
    trainer.save_model(OUTPUT_DIR)
    print(f"Training complete. Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    train_qwen_vl()
