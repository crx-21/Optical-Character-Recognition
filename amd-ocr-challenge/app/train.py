import torch
import os
from typing import Optional
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig, TrainingArguments
from peft import PeftModel, LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DATA_PATH = os.path.join(SCRIPT_DIR, "dataset", "qwen_formatted", "train_data.jsonl")
VALID_DATA_PATH = os.path.join(SCRIPT_DIR, "dataset", "qwen_formatted", "valid_data.jsonl")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "qwen_finetuned_lora")

def train_qwen_vl():
    # ROCm Optimized Quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )

    print("Loading model and processor...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": 0}, # Explicitly map everything to GPU 0 to avoid meta-tensor dispatch errors
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    # LoRA Configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Dataset loading
    train_dataset = load_dataset("json", data_files=TRAIN_DATA_PATH, split="train")
    valid_dataset = load_dataset("json", data_files=VALID_DATA_PATH, split="train")

    # Pre-process images to bypass broken torchvision jpeg decoder
    def preprocess_images(example):
        # The SFTTrainer internally calls apply_chat_template.
        # When the input is a conversation, it expects the 'image'
        # field in the content to be the image itself (PIL Image)
        # or a path, but not the whole content dictionary.
        # We ensure that if the 'image' key exists in a content dict,
        # it's passed correctly.

        # Actually, the error 'got type=<class 'dict'>' suggests that
        # the image_processor is receiving the content dictionary
        # instead of just the image.

        # To fix this and also the libjpeg issue, we pre-load images
        # and ensure the data format strictly matches what the
        # processor expects.
        for message in example['messages']:
            if message['role'] == 'user':
                for content in message['content']:
                    if content['type'] == 'image':
                        # Load as PIL and keep it in the dictionary
                        # The processor should be able to handle this
                        try:
                            img_path = content['image']
                            content['image'] = Image.open(img_path).convert("RGB")
                        except Exception as e:
                            print(f"Warning: Could not load image {content.get('image')}: {e}")
        return example

    train_dataset = train_dataset.map(preprocess_images)
    valid_dataset = valid_dataset.map(preprocess_images)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        bf16=True, # ROCm bfloat16
        optim="paged_adamw_32bit",
        lr_scheduler_type="linear",
        report_to="tensorboard",
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        peft_config=lora_config,
    )

    print("Starting training...")
    trainer.train()

    # Save the final adapter
    trainer.save_model(OUTPUT_DIR)
    print(f"Training complete. Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    train_qwen_vl()
