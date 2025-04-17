Any idea on what is the most costly operation on an LLM model? It’s matrix multiplication, which is the most dominant operation in most neural networks, including vector-matrix multiplication in the dense layer as well as matrix-matrix multiplication in the self-attention mechanism. We are pretty much familiar with the terms Q (query), K (key), and V (value) from the self-attention mechanism, where Q (query) and K (key) are multiplied to form the attention map:





Minimize image
Edit image
Delete image


Attention(Q, K, V) = softmax( (Q × Kᵀ) / √dₖ ) × V

Q = X* W_Q, K = X W_K, V = X * W_V

Where:

X = Input matrix (sequence of token embeddings)

W_Q, W_K, W_V = Learnable weight matrices for Q, K, and V



Minimize image
Edit image
Delete image


In a standard dense layer, the matrix multiplication (MatMul) between the input x ∈ ℝ¹ˣᵈ and the weight matrix W ∈ ℝᵈˣᵐ is expressed as:

y = xW = ∑ (from j = 1 to d) xⱼ × Wᵢⱼ, for i = 1, 2, ..., m

Where→



y ∈ ℝ¹ˣᵐ is the output vector.

Let’s now do an initial cost analysis:

For full-precision matrix multiplication where weights are random real numbers (x ∈ ℝ¹ˣᵈ):

Each element multiplication: y = x * w





1 multiply (x * w) = 1 FLOP

1 addition (accumulating to output) = 1 FLOP

Total per element: 2 FLOPs (1 multiply + 1 add)




So what if we could construct a new architecture without matrix multiplication that reduces computational cost and memory utilization while preserving the expressiveness of the network? Introducing a Matrix Multiplication-free Language Model with ternary weights {-1, 0, 1} in order to develop a new LLM architecture that is more cost-effective, lightweight, and scalable, and can run on edge devices, IoT systems, or ultra-low-power applications while also maintaining performance comparable to that of state-of-the-art language models.



MatMul-free dense layers (BitLinear layers) use ternary weights ∈ {-1, 0, +1} to replace full-precision matrix multiplication. By constraining the weights to the set {-1, 0, +1} and applying additional quantization techniques, MatMul operations are replaced with addition and negation operations.



That was the basic concept behind the model architecture, but the architecture also includes a token mixer for capturing sequential dependencies and a channel mixer for integrating information across embedding dimensions. We also made changes to other components of a typical LLM architecture to make them compatible with the ternary weight architecture. For example, It introduced Rotary Positional Embeddings (RoPE) in order to better capture token order and preserve contextual information across longer sequences, but in this case, the parameters were ternary.



To avoid standard MatMul-based dense layers, BitNet replaces them with BitLinear modules, which use ternary weights to transform MatMul operations into pure addition (ternary accumulation).

In this case, the weight matrix elements Wᵢⱼ are constrained to values from the set {−1, 0, +1}. Let W be the ternary weight matrix. The ternary MatMul can be written as:

Y = x ⊛ W = ∑ (from j = 1 to d) xⱼ × Wᵢⱼ, where Wᵢⱼ ∈ {−1, 0, +1}, for i = 1, 2, ..., m

Here→

Y ∈ ℝ¹ˣᵐ is the output.

⊛ denotes ternary MatMul, simplified to accumulation.



Since Wᵢⱼ ∈ {−1, 0, +1}, the multiplication operation can be replaced with simple logic:

If Wᵢⱼ = +1, then xⱼ × Wᵢⱼ = xⱼ → just pass the value as-is (no computation needed)

If Wᵢⱼ = 0, then xⱼ × Wᵢⱼ = 0 → ignore it (no computation needed)

If Wᵢⱼ = −1, then xⱼ × Wᵢⱼ = −xⱼ → just change the sign (no computation needed)

Thus, the output Yᵢ becomes:

Yᵢ = ∑ (xⱼ for Wᵢⱼ = +1) − ∑ (xⱼ for Wᵢⱼ = −1)



No multiplication needed; uses only addition, subtraction, or skipping.

Total per element: 0 FLOPs


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
Minimize image
Edit image
Delete image


📉 train/loss

The model is learning well — the loss starts around 6 and steadily drops to below 1, which is a strong sign of convergence and effective training.

📉 train/learning_rate

The learning rate is gradually decreasing, which is common practice in training LLMs — high learning rate early for fast learning, then lower to fine-tune.

📉 train/grad_norm

The gradient norm is decreasing, indicating that the model is stabilizing and updates are becoming smaller — again a good sign.

📈 train/global_step

Steadily increasing from 100 to 550. This just confirms that training progressed over time.

📈 train/epoch

Starts from 10 and goes beyond 45 — multiple epochs mean the model is getting multiple exposures to the full dataset, which is typical for deep training.



Let's try the model out

from huggingface_hub import login
login(token="Hugging Face Token")

model.save_pretrained("MMfreeLM-370M")
tokenizer.save_pretrained("MMfreeLM-370M")

from huggingface_hub import HfApi, HfFolder
from transformers import AutoModelForCausalLM, AutoTokenizer

from huggingface_hub import create_repo
create_repo("MMfreeLM-370M", private=False)

model.push_to_hub("MMfreeLM-370M")
tokenizer.push_to_hub("MMfreeLM-370M")
from mmfreelm.models import HGRNBitForCausalLM
import torch
model = HGRNBitForCausalLM.from_pretrained("Sakib323/MMfreeLM-370M")
model.to("cuda" if torch.cuda.is_available() else "cpu")
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Sakib323/MMfreeLM-370M")
tokenizer.pad_token = tokenizer.eos_token 
def generate_text(prompt, max_length=100):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
 
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example prompt
prompt = "I want you to act as a UX/UI developer."
generated_text = generate_text(prompt)
print(generated_text)



Access the model from my hugging face repo: Sakib323/MMfreeLM-370M · Hugging Face

