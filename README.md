# 🚀 Fine-Tuning LLaMA-2 with 4-bit Quantization & LoRA

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-Transformers-orange)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-Educational-green)](#license)

This project demonstrates **fine-tuning a large language model (LLaMA-2 7B)** using **4-bit quantization** and **LoRA (Low-Rank Adaptation)** for **memory-efficient supervised fine-tuning**.

---

## 📌 Table of Contents

1. [Installation](#installation)
2. [Hugging Face Authentication](#hugging-face-authentication)
3. [Dataset Loading](#dataset-loading)
4. [Tokenizer Setup](#tokenizer-setup)
5. [Quantization Setup](#quantization-setup)
6. [Model Loading with Quantization](#model-loading-with-quantization)
7. [Model Configuration Adjustments](#model-configuration-adjustments)
8. [LoRA Configuration](#lora-configuration)
9. [Training Arguments](#training-arguments)
10. [SFTTrainer Setup](#sfttrainer-setup)
11. [Training and Push to Hub](#training-and-pushing-to-hugging-face-hub)
12. [Notes](#notes)
13. [License](#license)

---

## 🛠 Installation

```bash
pip install accelerate==0.21.0 \
            peft==0.4.0 \
            bitsandbytes==0.40.2 \
            transformers==4.30.2 \
            trl==0.4.7 \
            datasets
```

---

## 🔑 Hugging Face Authentication

```python
from huggingface_hub import notebook_login
notebook_login()
```

---

## 📚 Dataset Loading

```python
from datasets import load_dataset

data = load_dataset("timdettmers/openassistant-guanaco", split="train")
print(data)
```

---

## 📝 Tokenizer Setup

### For LLaMA-2:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_auth_token=True)
tokenizer.pad_token = tokenizer.eos_token
```

---

## ⚡ Quantization Setup

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype="float16"
)
```

---

## 🖥 Model Loading with Quantization

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map={"": 0}
)
```

---

## ⚙️ Model Configuration Adjustments

```python
model.config.use_cache = False
model.config.pretraining_tp = 1
```

---

## 🔗 LoRA Configuration

```python
from peft import LoraConfig

peft_config = LoraConfig(
    r=64,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

---

## 🎯 Training Arguments

```python
from transformers import TrainingArguments

training_arguments = TrainingArguments(
    output_dir="llama2_finetuned_chatbot",
    per_device_train_batch_size=8,
    gradi
```
