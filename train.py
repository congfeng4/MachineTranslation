# %% [markdown]
# # Machine Translation

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
#    'nllb': 'facebook/nllb-200-distilled-1.3B',
 #   't5-small': 'google-t5/t5-small',
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
def train_evaluate(model_name: str, src: str, tgt: str, seed: int, num_train=10, num_test=10, num_val=10):
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

    eval_result = trainer.evaluate(tokenised_test)
    eval_result.update(src=src, tgt=tgt, num_train=num_train, num_test=num_test, num_val=num_val, model_name=model_name)

    writer = jsonlines.open('wmt17_results.jsonl', mode='a')
    writer.write(eval_result)
    writer.close()
    return eval_result



# %% [markdown]
# ## Run Experiments

# %%
def parameter_grid():
    lang_pair = 'zh-en'
    for model_name in model_map.keys():
        for seed in [111, 112, 113]:
            src, tgt = lang_pair.split('-')
            for num_train in [16, 32, 64, 128, 256, 512, 1024, 2048]:
                for num_test in [20, 200]:
                    num_val = num_train // 2
                    yield dict(src=src, tgt=tgt, model_name=model_name, seed=seed,
                                num_train=num_train, num_test=num_test, num_val=num_val)

params_list = list(parameter_grid())
random.shuffle(params_list)


# %%
for params in params_list:
    clean_memory()
    print(params)
    try:
        res = train_evaluate(**params)
        print('OK', res)
    except Exception as e:
        print(f'Error processing {e}')
        raise e



