from transformers import pipeline
import torch
torch.cuda.empty_cache()

import pandas as pd
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
import re
from datasets import load_dataset, Dataset
import pandas as pd


model_name = "taide/TAIDE-LX-7B-Chat"

# Load the entire model on the GPU 0
device_map = {"": 0}

# 讀取 CSV 檔案
file_path = './data/news.csv'  # 替換為您的檔案路徑
df = pd.read_csv(file_path)

device_map = {"": 0}

# 將資料轉換為 Hugging Face Dataset 格式
dataset = Dataset.from_pandas(df)

# 定義轉換函數
def transform_conversation(example):
    sys_prompt = '你是一個專業的運動新聞編輯，我會給你文字轉播資料，請幫我用繁體中文撰寫新聞稿。'
    # 取出問題與生成內容
    human_text = example['content'].strip()
    assistant_text = example['news'].strip()

    # 應用格式化模板
    formatted_text = f'<<SYS>>{sys_prompt}<</SYS>><s>[INST] {human_text} [/INST] {assistant_text} </s>'
    return {'text': formatted_text}

# 應用轉換函數
transformed_dataset = dataset.map(transform_conversation)

train_test_dataset = transformed_dataset.train_test_split(test_size=0.1, seed=12)

# 設定 System Prompt 和使用者的問題
sys_prompt = "你是一個專業的運動新聞編輯，我會給你文字轉播資料，請幫我用繁體中文撰寫新聞稿。"

# 測試資料集
dataset = train_test_dataset['test']
print(dataset)

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

# 定義生成管道
# pipe = pipeline(task="text-generation", model=model_name, tokenizer=tokenizer, max_new_tokens=1024,device_map=device_map)
pipe = pipeline("text-generation",
                model_name, 
                torch_dtype=torch.bfloat16, 
                device_map="auto", 
                return_full_text=False)

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
