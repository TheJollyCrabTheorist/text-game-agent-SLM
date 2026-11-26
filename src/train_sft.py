from prepare_sft_dataset import (
    SYSTEM_PROMPT,
    build_user_prompt,
)

from data_integrity import validate_history
import json
import math
import random
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

MAX_LENGTH = 1024
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16

EPOCHS = 10
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

PATIENCE = 2
SEED = 42

DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("models/qwen25-05b-textworld-qlora")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TextWorldDataset(Dataset):

    def __init__(self, path, tokenizer, max_length, prompt_version="ordered"):
        if prompt_version == "legacy":
            from evaluate_games import LEGACY_SYSTEM_PROMPT as system_prompt, build_legacy_user_prompt as builder
        else:
            system_prompt, builder = SYSTEM_PROMPT, build_user_prompt
        self.examples = []

        total_trimmed = 0
        total_removed_steps = 0
        max_original_length = 0
        max_final_length = 0

        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]

        for row in rows:
            messages = row["messages"]

            if (
                len(messages) != 3
                or [m["role"] for m in messages]
                != ["system", "user", "assistant"]
            ):
                raise ValueError(
                    f"Unexpected message structure in {path}"
                )

            if "prompt_step" not in row or "previous_steps" not in row:
                raise ValueError(
                    f"Missing structured prompt data in {path}. "
                    "Regenerate the dataset using prepare_sft_dataset.py."
                )

            system_message = {
                "role": "system",
                "content": messages[0]["content"],
            }

            if system_message["content"] != system_prompt:
                raise ValueError(
                    f"Unexpected system prompt in {path}, "
                    f"game={row.get('game_id')}, step={row.get('step')}"
                )

            action = messages[-1]["content"]

            bounded_history = list(row["previous_steps"])
            validate_history(bounded_history, row["step"])
            if builder(row["prompt_step"], bounded_history) != messages[1]["content"]:
                raise ValueError("Stored prompt differs from structured data")
            removed_steps = 0

            while True:

                user_prompt = builder(
                    row["prompt_step"],
                    bounded_history,
                )

                current_messages = [
                    system_message,
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                    {
                        "role": "assistant",
                        "content": action,
                    },
                ]

                # Prompt: system + user + assistant generation prefix.
                prefix = tokenizer.apply_chat_template(
                    current_messages[:-1],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                )["input_ids"]

                # Full sequence: system + user + assistant target.
                full = tokenizer.apply_chat_template(
                    current_messages,
                    tokenize=True,
                    add_generation_prompt=False,
                    return_dict=True,
                )["input_ids"]

                # Ensure that the prompt is exactly the prefix
                # of the full sequence.
                if full[:len(prefix)] != prefix:
                    raise ValueError(
                        f"Token alignment error in {path}, "
                        f"game={row.get('game_id')}, "
                        f"step={row.get('step')}"
                    )

                if len(full) <= max_length:
                    break

                # Preserve the current state and remove only
                # the oldest previous interaction.
                if not bounded_history:
                    raise ValueError(
                        f"Current example exceeds MAX_LENGTH={max_length} "
                        f"even without interaction history: "
                        f"{len(full)} tokens. "
                        f"Game={row.get('game_id')}, "
                        f"step={row.get('step')}."
                    )

                bounded_history.pop(0)
                removed_steps += 1

            if removed_steps:
                total_trimmed += 1
                total_removed_steps += removed_steps

            max_original_length = max(
                max_original_length,
                len(tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=False,
                )),
            )

            max_final_length = max(
                max_final_length,
                len(full),
            )

            # Ignore system and user tokens in the loss.
            labels = full.copy()
            labels[:len(prefix)] = [-100] * len(prefix)

            # Ensure that at least one target token is trained.
            target_ids = [
                token for token in labels
                if token != -100
            ]

            if not target_ids:
                raise ValueError(
                    f"No assistant tokens available for training: "
                    f"{row.get('game_id')}, step={row.get('step')}"
                )

            target_text = tokenizer.decode(
                target_ids,
                skip_special_tokens=True,
            ).strip()

            if not target_text:
                raise ValueError(
                    f"Empty assistant target in {path}, "
                    f"game={row.get('game_id')}, "
                    f"step={row.get('step')}"
                )

            if target_text not in action:
                raise ValueError(
                    f"Unexpected target decoding in {path}, "
                    f"game={row.get('game_id')}, "
                    f"step={row.get('step')}: {target_text!r}"
                )

            self.examples.append({
                "input_ids": full,
                "labels": labels,
            })

        print(f"Loaded {len(self.examples)} examples from {path}")
        print(
            f"Examples requiring history truncation: "
            f"{total_trimmed}/{len(rows)}"
        )
        print(f"Oldest interactions removed: {total_removed_steps}")
        print(f"Maximum original sequence length: {max_original_length}")
        print(f"Maximum final sequence length: {max_final_length}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]

class DataCollator:

    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch):
        max_len = max(len(item["input_ids"]) for item in batch)

        input_ids = []
        attention_masks = []
        labels = []

        for item in batch:
            padding_length = max_len - len(item["input_ids"])

            input_ids.append(
                item["input_ids"]
                + [self.pad_token_id] * padding_length
            )

            attention_masks.append(
                [1] * len(item["input_ids"])
                + [0] * padding_length
            )

            labels.append(
                item["labels"]
                + [-100] * padding_length
            )

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(
                attention_masks, dtype=torch.long
            ),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()

    total_loss = 0.0
    total_target_tokens = 0

    for batch in dataloader:
        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        outputs = model(**batch)

        target_tokens = (batch["labels"] != -100).sum().item()

        total_loss += outputs.loss.item() * target_tokens
        total_target_tokens += target_tokens

    model.train()

    return total_loss / max(total_target_tokens, 1)


def train(args):

    set_seed(SEED)

    model_name = getattr(args, "model_name", MODEL_NAME)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device("cuda:0")

    print("\n=== DEVICE ===")
    print(torch.cuda.get_device_name(0))
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    print("\n=== LOADING TOKENIZER ===")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    print("\n=== LOADING DATASETS ===")

    train_dataset = TextWorldDataset(
        args.data_dir / "train.jsonl",
        tokenizer,
        MAX_LENGTH,
        args.prompt_version,
    )

    validation_dataset = TextWorldDataset(
        args.data_dir / "validation.jsonl",
        tokenizer,
        MAX_LENGTH,
        args.prompt_version,
    )

    collator = DataCollator(tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )

    print("\n=== LOADING 4-BIT MODEL ===")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map={"": 0},
        dtype=torch.float16,
    )

    model.config.use_cache = False

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)

    print("\n=== TRAINABLE PARAMETERS ===")
    model.print_trainable_parameters()

    model.train()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    updates_per_epoch = math.ceil(
        len(train_loader) / GRADIENT_ACCUMULATION_STEPS
    )

    total_steps = updates_per_epoch * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print("\n=== TRAINING CONFIGURATION ===")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Optimizer updates per epoch: {updates_per_epoch}")
    print(f"Total planned updates: {total_steps}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = []

    for epoch in range(EPOCHS):

        model.train()
        optimizer.zero_grad(set_to_none=True)

        running_loss = 0.0
        target_tokens_seen = 0

        num_batches = len(train_loader)

        print(f"\n{'=' * 60}")
        print(f"EPOCH {epoch + 1}/{EPOCHS}")
        print(f"{'=' * 60}")

        for batch_idx, batch in enumerate(train_loader):

            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
            }

            outputs = model(**batch)
            loss = outputs.loss

            target_tokens = (batch["labels"] != -100).sum().item()

            running_loss += loss.item() * target_tokens
            target_tokens_seen += target_tokens

            # Handle the final incomplete accumulation group correctly
            group_start = (
                batch_idx // GRADIENT_ACCUMULATION_STEPS
            ) * GRADIENT_ACCUMULATION_STEPS

            group_size = min(
                GRADIENT_ACCUMULATION_STEPS,
                num_batches - group_start,
            )

            (loss / group_size).backward()

            is_update_step = (
                (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0
                or batch_idx + 1 == num_batches
            )

            if is_update_step:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if (batch_idx + 1) % 10 == 0 or batch_idx + 1 == num_batches:
                print(
                    f"Batch {batch_idx + 1}/{num_batches} | "
                    f"Loss: {loss.item():.4f} | "
                    f"VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
                )

        train_loss = running_loss / max(target_tokens_seen, 1)

        val_loss = evaluate(
            model,
            validation_loader,
            device,
        )

        current_lr = scheduler.get_last_lr()[0]

        print(f"\nTrain loss: {train_loss:.4f}")
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Learning rate: {current_lr:.7f}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "learning_rate": current_lr,
        })

        with open(OUTPUT_DIR / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0

            print("New best validation loss. Saving checkpoint...")

            model.save_pretrained(OUTPUT_DIR / "best_adapter")
            tokenizer.save_pretrained(OUTPUT_DIR / "best_adapter")

        else:
            epochs_without_improvement += 1

            print(
                f"No improvement: "
                f"{epochs_without_improvement}/{PATIENCE}"
            )

        if epochs_without_improvement >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

    print("\n=== TRAINING COMPLETED ===")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best adapter: {OUTPUT_DIR / 'best_adapter'}")

    # Save final training history
    with open(OUTPUT_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR,
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default=MODEL_NAME,
        help=(
            "Base model name, e.g. Qwen/Qwen2.5-0.5B-Instruct "
            "or Qwen/Qwen2.5-1.5B-Instruct."
        ),
    )

    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--prompt-version", choices=["legacy", "ordered"], default="ordered")
    args = parser.parse_args()

    OUTPUT_DIR = args.output

    train(args)
