Finetuning LLMs using Quantization and LoRA

This project demonstrates fine-tuning large language models (LLMs), specifically Meta's LLaMA-2, using quantization and LoRA (Low-Rank Adaptation) to optimize for memory and computational efficiency. The project also leverages the Hugging Face Transformers and PEFT libraries.

🔹 Project Overview

Large language models are highly powerful but resource-intensive. This project focuses on techniques to efficiently fine-tune these models for custom tasks:

LoRA (Low-Rank Adaptation)

Fine-tunes large models efficiently using low-rank matrices.

Reduces computation and memory usage while maintaining performance.

Quantization

Converts high-precision model weights (32-bit) into lower-precision (4-bit) for faster inference and smaller model size.

Ideal for deployment on limited-resource devices.

Trainer & SFTTrainer

Uses Hugging Face Trainer and SFTTrainer for supervised fine-tuning.

Supports gradient accumulation, learning rate scheduling, and push-to-Hub functionality.

🔹 Key Features

Fine-tuning LLaMA-2 7B model on custom datasets.

Integration of 4-bit quantization using bitsandbytes.

Efficient parameter tuning via LoRA.

Uploading fine-tuned models directly to Hugging Face Hub.

🔹 Prerequisites
pip install accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 transformers==4.30.2 trl==0.4.7 datasets


Hugging Face account & token (Sign Up
)

Access to Meta’s LLaMA-2 (Download
)

🔹 Project Structure

tokenizer_setup.py – Initialize and configure tokenizer.

model_quantization.py – Load model with 4-bit quantization.

lora_setup.py – Configure LoRA for efficient fine-tuning.

train.py – Training script using SFTTrainer.

README.md – Project documentation.

🔹 Code Snippets
Tokenizer
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_auth_token=True)
tokenizer.pad_token = tokenizer.eos_token

Model with Quantization
from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype="float16"
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map={"": 0}
)

LoRA Configuration
from peft import LoraConfig

peft_config = LoraConfig(
    r=64,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

Training
from transformers import TrainingArguments
from trl import SFTTrainer

training_arguments = TrainingArguments(
    output_dir="llama2_finetuned_chatbot",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    optim="paged_adamw_4bit",
    learning_rate=2e-4,
    lr_scheduler_type="linear",
    save_strategy="epoch",
    logging_steps=10,
    num_train_epochs=1,
    max_steps=10,
    push_to_hub=True
)

trainer = SFTTrainer(
    model=model,
    train_dataset=data,
    peft_config=peft_config,
    dataset_text_field="text",
    args=training_arguments,
    tokenizer=tokenizer,
    max_seq_length=512
)

trainer.train()
trainer.push_to_hub()

🔹 References

LLaMA-2 on Hugging Face

LoRA (Low-Rank Adaptation) Paper

BitsAndBytes Quantization

Hugging Face Transformers

🔹 Outcome

Successfully fine-tuned a 7B parameter LLaMA-2 model with 4-bit quantization and LoRA.

Reduced memory footprint and computation time while retaining performance.

Model available for inference or further fine-tuning.


