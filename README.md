Training the model


!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
!pip install -U git+https://github.com/Sakib323/matmulfreellm.git
!pip install transformers
!pip install triton==2.2
!pip install datasets
!pip install wandb
import wandb
wandb.login()
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset

from mmfreelm.models import ( HGRNBitForCausalLM,HGRNBitModel, HGRNBitConfig)

import torch
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("ridger/MMfreeLM-370M")
tokenizer.pad_token = tokenizer.eos_token

demo_data = load_dataset("fka/awesome-chatgpt-prompts")

if "act" in demo_data["train"].column_names:
    demo_data = demo_data.remove_columns(["act"])

def tokenize_function(examples):
    tokens = tokenizer(examples["prompt"], truncation=True, padding=True, max_length=2048)
    tokens["labels"] = tokens["input_ids"].copy()  # Labels should be the same as input_ids
    return tokens

tokenized_dataset = demo_data.map(tokenize_function, batched=True, remove_columns=["prompt"])

split_datasets = tokenized_dataset["train"].train_test_split(test_size=0.1)


config = HGRNBitConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=1024,
    num_hidden_layers=24,
    max_position_embeddings=2048,
    attn_mode="fused_recurrent",
    use_short_conv=False,
    conv_size=4,
    rms_norm_eps=1e-6,
    pad_token_id=tokenizer.pad_token_id,
    rope_theta=10000.0,          
    use_ternary_rope=True,
    rotary_embeddings=False,     
)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HGRNBitForCausalLM(config).to(device)


training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    per_device_train_batch_size=4,
    remove_unused_columns=False,
    num_train_epochs=50,
    learning_rate=4e-3,
    weight_decay=0.01,
    logging_steps=100,
    save_steps=1000,
    gradient_accumulation_steps=4,
    fp16=False,
    run_name="HGRNBit-MMfreeLM-370M-with-rotary-embedding",
    report_to="wandb",
)



data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=split_datasets["train"],
    eval_dataset=split_datasets["test"],
    data_collator=data_collator,
)

trainer.train()
