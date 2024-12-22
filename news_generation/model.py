import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from datasets import load_dataset, Dataset
import pandas as pd
import re

model_name = "taide/TAIDE-LX-7B-Chat"

# LoRA attention dimension
lora_r = 64

# Alpha parameter for LoRA scaling
lora_alpha = 16

# Dropout probability for LoRA layers
lora_dropout = 0.1

################################################################################
# bitsandbytes parameters
################################################################################

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

################################################################################
# TrainingArguments parameters
################################################################################

# Output directory where the model predictions and checkpoints will be stored
output_dir = "./results"

# Number of training epochs
num_train_epochs = 20

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = False

# Batch size per GPU for training
per_device_train_batch_size = 1

# Batch size per GPU for evaluation
per_device_eval_batch_size = 1

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 4

# Enable gradient checkpointing
gradient_checkpointing = True

# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3

# Initial learning rate (AdamW optimizer)
learning_rate = 1e-4

# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001

# Optimizer to use
optim = "paged_adamw_32bit"

# Learning rate schedule
lr_scheduler_type = "cosine"

# Number of training steps (overrides num_train_epochs)
max_steps = -1

# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03

# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True

# Save checkpoint every X updates steps
save_steps = 0

# Log every X updates steps
logging_steps = 25

################################################################################
# SFT parameters
################################################################################

# Maximum sequence length to use
max_seq_length = None

# Pack multiple short examples in the same input sequence to increase efficiency
packing = False

# Load the entire model on the GPU 0
device_map = {"": 0}


# 讀取 CSV 檔案
file_path = './data/news.csv'  # 替換為您的檔案路徑
df = pd.read_csv(file_path)

# 檢查資料格式
# print(df.head())  # 確認資料有 `question` 和 `answer` 欄位

# 將資料轉換為 Hugging Face Dataset 格式
dataset = Dataset.from_pandas(df)
sys_prompt = '你是一個專業的運動新聞編輯，我會給你文字轉播資料，請幫我用繁體中文撰寫新聞稿。'

# 定義轉換函數
def transform_conversation(example):
    # 取出問題與生成內容
    human_text = example['content'].strip()
    assistant_text = example['news'].strip()

    # 應用格式化模板
    formatted_text = f'<<SYS>>{sys_prompt}<</SYS>><s>[INST] {human_text} [/INST] {assistant_text} </s>'
    return {'text': formatted_text}

# 應用轉換函數
transformed_dataset = dataset.map(transform_conversation)

train_test_dataset = transformed_dataset.train_test_split(test_size=0.1, seed=12)

# Load dataset (you can process it here)
dataset = train_test_dataset['train']

# Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

# Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard"
)

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)

# Train model
trainer.train()
# Save trained model
# trainer.model.save_pretrained(new_model)

# Save trained model
new_model = './output/'
trainer.model.save_pretrained(new_model)


# Ignore warnings
logging.set_verbosity(logging.CRITICAL)

# 測試資料集
dataset = train_test_dataset['test']
print(dataset)

# 定義生成管道
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=1024)


# 定義轉換函數
def transform_conversation(example):
    # 提取 [INST] [/INST] 中的 human_text 和答案 assist_text
    human_text = example['content'].strip()  # 假設 human_text 存在 'content' 欄位
    assist_text = example['news'].strip()   # 假設 assist_text 存在 'news' 欄位
    
    # 準備格式化的提示
    prompt = f'<<SYS>>{sys_prompt}<</SYS>><s>[INST] {human_text} [/INST]'
    
    # 使用生成模型產生結果
    generated_text = pipe(prompt, max_new_tokens=1024, do_sample=True, top_k=50, top_p=0.9)[0]["generated_text"]
    after_inst_text = generated_text.split("[/INST]")[-1].strip()

    # result = pipe(prompt)
    
    # 提取生成的文字
    # generated_text = result[0]['generated_text']
    
    # 返回原始輸入和生成結果
    return {
        'generated_text': after_inst_text
    }

# 應用轉換函數，生成結果
transformed_results = dataset.map(transform_conversation)

# 轉換為 DataFrame
df = pd.DataFrame(transformed_results)

# 將結果存為 CSV
df[['content','news','generated_text']].to_csv("generated_results.csv", index=False)

print("生成完成，結果已存入 'generated_results.csv'")

