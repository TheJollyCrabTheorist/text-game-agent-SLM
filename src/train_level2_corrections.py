import argparse
from data_integrity import validate_history, validate_adapter_base
from evaluate_games import PROMPT_CONFIGS
import json
import math
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)
from peft import PeftModel, prepare_model_for_kbit_training


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

INITIAL_ADAPTER = (
    PROJECT_ROOT
    / "models/qwen25-05b-textworld-qlora-objective/best_adapter"
)

TRAIN_PATH = PROJECT_ROOT / "data/level2/finetuning/train.jsonl"
VALIDATION_PATH = (
    PROJECT_ROOT / "data/level2/finetuning/validation.jsonl"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT / "models/level2/qlora-corrective"
)

MAX_LENGTH = 1024
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
EPOCHS = 2
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05
CORRECTION_WEIGHT = 10.0
MAX_GRAD_NORM = 1.0
SEED = 42


def set_seed(seed):
    """Set the random seed."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path):
    """Read a JSONL file into a list of dicts."""
    with path.open(encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


class TextWorldDataset(Dataset):
    def __init__(self, path, tokenizer):
        """Initialize the level 2 corrective dataset."""
        self.examples = []

        rows = read_jsonl(path)

        for index, row in enumerate(rows):
            messages = row["messages"]

            if len(messages) != 3:
                raise ValueError(
                    f"Example {index}: expected exactly 3 messages."
                )

            if [m["role"] for m in messages] != [
                "system", "user", "assistant"
            ]:
                raise ValueError(
                    f"Example {index}: unexpected message roles."
                )

            history = list(row["previous_steps"])
            validate_history(history, row["step"])
            builders = [c["user_prompt_builder"] for c in PROMPT_CONFIGS.values()
                        if c["system_prompt"] == messages[0]["content"]]
            if len(builders) != 1:
                raise ValueError(f"Example {index}: unrecognized system prompt")
            builder = builders[0]
            if builder(row["prompt_step"], history) != messages[1]["content"]:
                raise ValueError(f"Example {index}: stored prompt differs from structured data")
            while True:
                bounded_messages = [messages[0], {"role": "user", "content":
                    builder(row["prompt_step"], history)}, messages[2]]
                prompt_ids = tokenizer.apply_chat_template(
                    bounded_messages[:-1], tokenize=True,
                    add_generation_prompt=True, return_dict=False)
                full_ids = tokenizer.apply_chat_template(
                    bounded_messages, tokenize=True,
                    add_generation_prompt=False, return_dict=False)
                if full_ids[:len(prompt_ids)] != prompt_ids:
                    raise ValueError(f"Example {index}: prompt/full token alignment failed")
                if len(full_ids) <= MAX_LENGTH:
                    break
                if not history:
                    raise ValueError(f"Example {index}: current example exceeds MAX_LENGTH")
                history.pop(0)
            input_ids = full_ids
            labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

            if len(input_ids) != len(labels):
                raise ValueError(
                    f"Example {index}: input/label length mismatch."
                )

            if all(label == -100 for label in labels):
                raise ValueError(
                    f"Example {index}: no assistant target remains."
                )

            self.examples.append({
                "input_ids": input_ids,
                "labels": labels,
                "source": row.get("source", "unknown"),
                "was_correction": row.get(
                    "was_correction", False
                ),
                "game_id": row["game_id"],
            })

        if not self.examples:
            raise ValueError(f"No examples found in {path}")

        print(f"Loaded {len(self.examples)} examples from {path}")

    def __len__(self):
        """Return the number of correction examples."""
        return len(self.examples)

    def __getitem__(self, index):
        """Return one correction example as model inputs."""
        return self.examples[index]


class DataCollator:
    def __init__(self, tokenizer):
        """Initialize the level 2 corrective dataset."""
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, examples):
        """Collate correction examples into model batches."""
        max_length = max(
            len(example["input_ids"])
            for example in examples
        )

        input_ids = []
        attention_masks = []
        labels = []

        for example in examples:
            padding = max_length - len(example["input_ids"])

            input_ids.append(
                example["input_ids"]
                + [self.pad_token_id] * padding
            )

            attention_masks.append(
                [1] * len(example["input_ids"])
                + [0] * padding
            )

            labels.append(
                example["labels"]
                + [-100] * padding
            )

        return {
            "input_ids": torch.tensor(
                input_ids, dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                attention_masks, dtype=torch.long
            ),
            "labels": torch.tensor(
                labels, dtype=torch.long
            ),
        }


def evaluate(model, dataloader, device):
    """Evaluate the model on a dataloader."""
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            outputs = model(**batch)

            target_tokens = (
                batch["labels"] != -100
            ).sum().item()

            total_loss += outputs.loss.item() * target_tokens
            total_tokens += target_tokens

    model.train()

    return total_loss / max(total_tokens, 1)


def save_adapter(model, tokenizer, output_dir):
    """Save the LoRA adapter."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def train(args):
    """Run the level 2 corrective fine-tuning."""
    set_seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this QLoRA training script."
        )

    if not args.initial_adapter.exists():
        raise FileNotFoundError(args.initial_adapter)
    if (args.output / "best_adapter").resolve() == args.initial_adapter.resolve():
        raise ValueError("Output would overwrite the initial adapter")
    validate_adapter_base(args.initial_adapter, args.base_model)

    print("=" * 60)
    print("LEVEL 2 CORRECTIVE FINE-TUNING")
    print("=" * 60)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Initial adapter:", args.initial_adapter)
    print("Output:", args.output)

    tokenizer = AutoTokenizer.from_pretrained(
        args.initial_adapter
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = TextWorldDataset(
        args.train_data, tokenizer
    )

    validation_dataset = TextWorldDataset(
        args.validation_data, tokenizer
    )

    print("Training data:", args.train_data)
    print("Validation data:", args.validation_data)
    print("Training examples:", len(train_dataset))
    print("Validation examples:", len(validation_dataset))

    # Give correction examples more sampling probability.
    sample_weights = [
        CORRECTION_WEIGHT
        if example["was_correction"]
        else 1.0
        for example in train_dataset.examples
    ]

    samples_per_epoch = (
        math.ceil(
            len(train_dataset) / GRADIENT_ACCUMULATION_STEPS
        )
        * GRADIENT_ACCUMULATION_STEPS
    )

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=samples_per_epoch,
        replacement=True,
    )

    collator = DataCollator(tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        collate_fn=collator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
    )

    print("\n=== LOADING BASE MODEL ===")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quantization_config,
        device_map="auto",
    )

    base_model = prepare_model_for_kbit_training(
        base_model
    )

    print("\n=== LOADING EXISTING LEVEL 1 ADAPTER ===")

    model = PeftModel.from_pretrained(
        base_model,
        args.initial_adapter,
        is_trainable=True,
    )

    model.config.use_cache = False
    model.print_trainable_parameters()

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if not trainable_parameters:
        raise RuntimeError("No trainable parameters found.")

    device = model.get_input_embeddings().weight.device

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=WEIGHT_DECAY,
    )

    total_steps = (
        math.ceil(len(train_loader) / GRADIENT_ACCUMULATION_STEPS)
        * args.epochs
    )

    warmup_steps = int(total_steps * WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print("\n=== INITIAL VALIDATION ===")

    initial_val_loss = evaluate(
        model, validation_loader, device
    )

    print(f"Initial validation loss: {initial_val_loss:.6f}")

    best_val_loss = initial_val_loss
    best_epoch = 0

    # Preserve the initial adapter if training does not improve validation.
    save_adapter(model, tokenizer, args.output / "best_adapter")

    history = {
        "base_model": args.base_model,
        "initial_adapter": str(args.initial_adapter),
        "train_data": str(args.train_data),
        "validation_data": str(args.validation_data),
        "epochs_requested": args.epochs,
        "learning_rate": args.learning_rate,
        "seed": SEED,
        "initial_validation_loss": initial_val_loss,
        "epochs": [],
        "best_epoch": 0,
        "best_validation_loss": initial_val_loss,
    }

    print("\n=== TRAINING ===")

    for epoch in range(args.epochs):
        model.train()

        optimizer.zero_grad(set_to_none=True)

        total_train_loss = 0.0
        optimizer_steps = 0

        for step, batch in enumerate(train_loader):
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            outputs = model(**batch)
            loss = outputs.loss

            total_train_loss += loss.item()

            (loss / GRADIENT_ACCUMULATION_STEPS).backward()

            should_update = (
                (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0
                or (step + 1) == len(train_loader)
            )

            if should_update:
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    MAX_GRAD_NORM,
                )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                optimizer_steps += 1

        average_train_loss = (
            total_train_loss / len(train_loader)
        )

        validation_loss = evaluate(
            model, validation_loader, device
        )

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train loss: {average_train_loss:.6f} | "
            f"validation loss: {validation_loss:.6f} | "
            f"optimizer steps: {optimizer_steps}"
        )

        history["epochs"].append({
            "epoch": epoch + 1,
            "train_loss": average_train_loss,
            "validation_loss": validation_loss,
            "optimizer_steps": optimizer_steps,
        })

        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            best_epoch = epoch + 1

            save_adapter(
                model,
                tokenizer,
                args.output / "best_adapter",
            )

            print("New best adapter saved.")

        torch.cuda.empty_cache()

    history["best_epoch"] = best_epoch
    history["best_validation_loss"] = best_val_loss

    args.output.mkdir(parents=True, exist_ok=True)

    with (args.output / "training_history.json").open("w") as file:
        json.dump(history, file, indent=2)

    print("\n=== TRAINING COMPLETE ===")
    print(f"Initial validation loss: {initial_val_loss:.6f}")
    print(f"Best validation loss:    {best_val_loss:.6f}")
    print(f"Best epoch:              {best_epoch}")
    print(f"Best adapter:            {args.output / 'best_adapter'}")
    print(f"History:                 {args.output / 'training_history.json'}")


def main():
    """Run the level 2 corrective fine-tuning pipeline."""
    parser = argparse.ArgumentParser(
        description="Fine-tune the Level 1 adapter on Level 2 corrections."
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=LEARNING_RATE,
    )
    
    parser.add_argument(
        "--train-data",
        type=Path,
        default=TRAIN_PATH,
        help="Path to the training JSONL dataset.",
    )

    parser.add_argument(
        "--validation-data",
        type=Path,
        default=VALIDATION_PATH,
        help="Path to the validation JSONL dataset.",
    )

    parser.add_argument("--initial-adapter", type=Path, default=INITIAL_ADAPTER)
    parser.add_argument("--base-model", default=MODEL_NAME)
    args = parser.parse_args()
    if args.epochs < 1 or args.learning_rate <= 0:
        parser.error("epochs and learning rate must be positive")
    train(args)


if __name__ == "__main__":
    main()
