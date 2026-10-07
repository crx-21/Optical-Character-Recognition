import os

import torch
from PIL import Image
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)
from trl import SFTConfig, SFTTrainer

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DATA_PATH = os.path.join(SCRIPT_DIR, "dataset", "qwen_formatted", "train_data.jsonl")
VALID_DATA_PATH = os.path.join(SCRIPT_DIR, "dataset", "qwen_formatted", "valid_data.jsonl")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "qwen_finetuned_lora")

# Caps visual tokens per image (28*28 px = 1 token). Lower = less VRAM.
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1024 * 28 * 28


class QwenVLCollator:
    """Builds batches on the fly: loads images, runs the processor, builds labels.

    Doing this in the collator (instead of dataset.map) avoids two problems:
    1. datasets re-serializes PIL images to Arrow as {"bytes", "path"} dicts,
       which is what the processor choked on.
    2. Storing pixel_values for every sample in Arrow is huge and slow.
    """

    def __init__(self, processor):
        self.processor = processor
        tok = processor.tokenizer
        self.pad_id = tok.pad_token_id
        # Everything up to and including this marker is prompt -> masked from loss
        self.assistant_marker = tok.encode("<|im_start|>assistant\n", add_special_tokens=False)
        self.vision_ids = [
            tok.convert_tokens_to_ids(t)
            for t in ("<|image_pad|>", "<|vision_start|>", "<|vision_end|>")
        ]

    def _find_last(self, seq, marker):
        n = len(marker)
        for i in range(len(seq) - n, -1, -1):
            if seq[i : i + n] == marker:
                return i + n
        return 0

    def __call__(self, examples):
        texts, images = [], []
        for ex in examples:
            clean_messages, sample_images = [], []
            for msg in ex["messages"]:
                content = msg["content"]
                if isinstance(content, str):
                    clean_messages.append({"role": msg["role"], "content": content})
                    continue
                parts = []
                for c in content:
                    if c["type"] == "image":
                        path = c["image"]
                        if not os.path.isabs(path):
                            path = os.path.join(SCRIPT_DIR, path)
                        sample_images.append(Image.open(path).convert("RGB"))
                        # Placeholder only: no extra keys, so the chat template
                        # emits exactly one <|image_pad|> block per image.
                        parts.append({"type": "image"})
                    else:
                        parts.append({"type": "text", "text": c["text"]})
                clean_messages.append({"role": msg["role"], "content": parts})

            texts.append(
                self.processor.apply_chat_template(
                    clean_messages, tokenize=False, add_generation_prompt=False
                )
            )
            images.append(sample_images)

        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )

        labels = batch["input_ids"].clone()
        for i in range(labels.size(0)):
            ids = batch["input_ids"][i].tolist()
            start = self._find_last(ids, self.assistant_marker)
            labels[i, :start] = -100  # mask system/user/image prompt
        labels[batch["input_ids"] == self.pad_id] = -100
        for vid in self.vision_ids:
            labels[batch["input_ids"] == vid] = -100
        batch["labels"] = labels
        return batch


def train_qwen_vl():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    print("Loading model and processor...")
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        # Regex restricts LoRA to the language model; plain names would also
        # match the vision tower's attention/MLP layers.
        target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Raw dataset only; no .map() and no PIL objects stored in Arrow
    train_dataset = load_dataset("json", data_files=TRAIN_DATA_PATH, split="train")
    valid_dataset = load_dataset("json", data_files=VALID_DATA_PATH, split="train")

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        bf16=True,
        optim="paged_adamw_32bit",
        lr_scheduler_type="linear",
        report_to="tensorboard",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Required for custom multimodal collator:
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        peft_config=lora_config,
        data_collator=QwenVLCollator(processor),
        processing_class=processor,
    )

    print("Starting training...")
    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"Training complete. Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    train_qwen_vl()