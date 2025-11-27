# %% [markdown]
# # Machine Translation - Inference

# %% [markdown]
# Uncomment this if you are using Colab. Please reserve at least 80GB storage.

# %%
# from google.colab import drive
# drive.mount('/content/drive')

# %%
# cd /content/drive/MyDrive/Machine Learning/Project

# %%
import os, json, math, time
import dotenv
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
    AutoModelForSeq2SeqLM,
)
import torch
import numpy as np
import random
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import jsonlines

dotenv.load_dotenv('.env')

# %% [markdown]
# ## Load Dataset WMT17
# 
# Device space limitation.

# %%
SUPPORTED_LANGS = ['zh-en'] #'cs-en', 'de-en', 'fi-en', 'lv-en', 'ru-en', 'tr-en', ]
lang_map = {
        'en': 'en_XX',
        'zh': 'zh_CN',
        'cs': 'cs_CZ',
        'de': 'de_DE',
        'fi': 'fi_FI',
        'lv': 'lv_LV',
        'ru': 'ru_RU',
        'tr': 'tr_TR',
    }
model_map = {
    'mbart': 'facebook/mbart-large-50-many-to-many-mmt',
    'opus-mt': 'Helsinki-NLP/opus-mt-zh-en',
    'nllb': 'facebook/nllb-200-distilled-1.3B',
    't5-small': 'google-t5/t5-small',
}
# Set the best model parameters here!
model_params = {
    'mbart': {'num_train': 2048},
    'opus-mt': {'num_train': 2048},
    'nllb': {'num_train': 2048},
    't5-small': {'num_train': 2048},
}

# %% [markdown]
# ## Load Pretrained Model.
# 
# | Model key | HF model id                              | # Params | Arch.                     | Languages             | Pre-train data        | Tokeniser           | Requires prompt prefix?    | Size on disk |
# | --------- | ---------------------------------------- | -------- | ------------------------- | --------------------- | --------------------- | ------------------- | -------------------------- | ------------ |
# | mbart     | facebook/mbart-large-50-many-to-many-mmt | 610 M    | Transformer-large (12-12) | 50 langs, any→any     | CC25 + mined data     | SentencePiece 250 k | No (src/tgt codes in call) | ~2.3 GB      |
# | opus-mt   | Helsinki-NLP/opus-mt-zh-en               | 74 M     | Transformer-base (6-6)    | zh → en only          | OPUS corpus           | SentencePiece 32 k  | No                         | ~298 MB      |
# | nllb      | facebook/nllb-200-distilled-1.3B         | 1.3 B    | Transformer (12-12, MoE)  | 200 langs, any→any    | NLLB-200 corpus       | SentencePiece 256 k | No                         | ~5 GB        |
# | t5-small  | google-t5/t5-small                       | 60 M     | Encoder-decoder (6-6)     | any pair in pre-train | C4 + downstream tasks | SentencePiece 32 k  | Yes ("translate X to Y:")  | ~242 MB      |
# 

# %%
def build_model(model_name, src_lang, tgt_lang):
    model_name = model_map[model_name]
    tok = AutoTokenizer.from_pretrained(model_name, src_lang=src_lang, tgt_lang=tgt_lang, cache_dir='./cache')
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir='./cache')
    return tok, model


# %%
import random
import os
import numpy as np
import torch

def seed_all(seed: int = 42):
    # 1. Python builtin
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 2. NumPy
    np.random.seed(seed)

    # 3. PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[seed_all] Global random seed set to {seed}")

# %% [markdown]
# ## Load MT Metrics

# %%
from evaluate import load

bleu = load("sacrebleu", cache_dir='./cache')
chrf = load("chrf", cache_dir='./cache')


# %% [markdown]
# ## Preprocess Datasets

# %%
def split_train_test(full_train, num_trains, num_test, num_val):
    indices = random.sample(range(len(full_train)), num_trains + num_test + num_val)
    raw_train = full_train.select(indices[:num_trains])
    raw_test = full_train.select(indices[num_trains:num_trains + num_test])
    raw_val = full_train.select(indices[num_trains + num_test:])
    return raw_train, raw_test, raw_val


# %%
def flip_dataset(raw, src, tgt):
    def flip(batch):
        batch["translation"] = {src: batch["translation"][tgt],
                                tgt: batch["translation"][src]}
        return batch
    return raw.map(flip, batched=False)


# %%
def clean_memory():
    import torch, gc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # 把未用缓存还给 CUDA
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()   # 把未用缓存还给 CUDA
    gc.collect()               # 再让 Python 回收一次

# %%
def train_model(model_name: str, src: str, tgt: str, num_train=10, num_test=10, num_val=10):
    need_prompt = model_name.startswith('t5')
    data = load_dataset("wmt17", name="-".join([src, tgt]), split="train", cache_dir='./cache')

    lang_src = lang_map[src]
    lang_tgt = lang_map[tgt]
    tok, model = build_model(model_name, lang_src, lang_tgt)

    num_train_epochs = 10
    max_src, max_tgt = 128, 128

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        labels = np.where(labels != -100, labels, tok.pad_token_id)
        pred_str = tok.batch_decode(preds, skip_special_tokens=True)
        label_str = tok.batch_decode(labels, skip_special_tokens=True)
        bleu_score  = bleu.compute(predictions=pred_str,
                                    references=[[r] for r in label_str])["score"]
        chrf_score  = chrf.compute(predictions=pred_str,
                                    references=label_str)["score"]

        return {"bleu": bleu_score, "chrf": chrf_score}

    def encode(ex):
        en_sent = [item[src] for item in ex["translation"]]
        zh_sent = [item[tgt] for item in ex["translation"]]
        if need_prompt:
            # T5 is the google model that needs a prompt.
            en_sent = ["Translate from English to Chinese: " + item for item in en_sent]

        # 一次调用同时编码源端和目标端
        model_inputs = tok(
            en_sent,
            text_target=zh_sent,
            max_length=max_src,
            truncation=True,
            padding=False,          # 动态 padding，由 data_collator 完成
        )
        return model_inputs

    raw_train, raw_test, raw_val = split_train_test(data,
                                                    num_trains=num_train, num_test=num_test, num_val=num_val)
    # Why we tokenize the split instead of data? Since data is very large, and we only use a small part of it!!
    # So 3 times tokenizations is OK.
    tokenised_train = raw_train.map(encode, batched=True)
    tokenised_val = raw_val.map(encode, batched=True)
    tokenised_test = raw_test.map(encode, batched=True)

    data_coll = DataCollatorForSeq2Seq(tok, model=model)

    args = Seq2SeqTrainingArguments(
        # output_dir='/content/output',
        eval_strategy="steps",
        eval_steps=500,
        logging_steps=100,
        save_steps=500,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=3e-5,
        num_train_epochs=num_train_epochs,
        predict_with_generate=True,
        generation_max_length=max_tgt,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        report_to="none",
        fp16=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenised_train,
        eval_dataset=tokenised_val,
        processing_class=tok,
        data_collator=data_coll,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()
    return model, tok


# %% [markdown]
# ## Run Inference

# %% [markdown]
# Obtain the input setences.

# %%
dataset = load_dataset('wmt17', 'zh-en', split='test', cache_dir='./cache')

# %%
seed_all(111) # Set a seed to 111.

# %%
len(dataset)

# %%
num_inputs = 5 # Fixed to 5.

input_data = dataset.select(random.choices(range(len(dataset)), k=num_inputs))
len(input_data)

# %%
input_data

# %%
def get_avail_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device

# %%
import torch
from tqdm.auto import tqdm

def translate(model, tok, input_data, src='zh'):
    model.eval()
    device = get_avail_device()

    # 1. 生成参数（可按需调）
    gen_kwargs = {
        "max_length": 64,
        "num_beams": 4,
        "early_stopping": True,
    }

    out_f = []

    for sample in tqdm(input_data, desc="Translating"):
        zh = sample["translation"][src]
        inputs = tok(zh, return_tensors="pt").to(device)

        with torch.no_grad():
            pred = model.generate(**inputs, **gen_kwargs)

        hyp = tok.decode(pred[0], skip_special_tokens=True)
        out_f.append(hyp.strip())

    return out_f


# %%
tgt, src = 'en', 'zh'

translation_results = {'tgt': [sample['translation'][tgt] for sample in input_data],
                       'src': [sample['translation'][src] for sample in input_data]}

for key in model_map:
    num_train = model_params[key].get("num_train", 32)
    model, tok = train_model(key, src='zh', tgt='en', num_train=num_train, num_test=20)
    translation_results[key] = translate(model, tok, input_data, src)

# %%
df = pd.DataFrame(translation_results)
df.to_csv('./translation_results.csv', index=False)


