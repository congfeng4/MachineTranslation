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
    GenerationConfig,
)
import torch
import numpy as np
import random
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import jsonlines
from joblib import delayed, Parallel

dotenv.load_dotenv('.env')

# %% [markdown]
# ## Load Dataset WMT17
# 
# | Field                                   | Value                                                                                            |
# | --------------------------------------- | ------------------------------------------------------------------------------------------------ |
# | Release year                            | 2017                                                                                             |
# | Language pair                           | Chinese ↔ English (zh-en)                                                                        |
# | Parallel sentences – Train              | 25 136 609                                                                                       |
# | Parallel sentences – Dev (valid)        | 2 002                                                                                            |
# | Parallel sentences – Test (newstest-17) | 2 001                                                                                            |
# | Approx. download size                   | 884 MiB                                                                                          |
# | De-compressed size                      | 6.4 GiB                                                                                          |
# | Main sources                            | UN Parallel Corpus v1.0, News Commentary v12, CWMT corpus, WikiTitles, etc.                      |
# | Domain                                  | News + mixed web crawl                                                                           |
# | Tokenisation                            | raw text + SGML/TXT parallel files; no forced segmentation provided                              |
# | Evaluation metric reported              | BLEU (baseline ≈ 17–18 on test set)                                                              |
# | Hosted links                            | [TFDS catalog](https://www.tensorflow.org/datasets/catalog/wmt17_translate#wmt17_translatezh-en) |
# | HF compatibility                        | load via `datasets.load_dataset("wmt17", "zh-en")`                                               |
# 

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
    # 'mbart': 'facebook/mbart-large-50-many-to-many-mmt',
    # 'opus-mt': 'Helsinki-NLP/opus-mt-zh-en',
    'nllb': 'facebook/nllb-200-distilled-600M',
}

# %%
data = load_dataset("wmt17", name='zh-en', split="train", cache_dir='./cache')


# %% [markdown]
# ## Load Pretrained Model.
# 
# | Model key | HF model id                              | # Params | Arch.                     | Languages             | Pre-train data        | Tokeniser           | Requires prompt prefix?    | Size on disk |
# | --------- | ---------------------------------------- | -------- | ------------------------- | --------------------- | --------------------- | ------------------- | -------------------------- | ------------ |
# | mbart     | facebook/mbart-large-50-many-to-many-mmt | 610 M    | Transformer-large (12-12) | 50 langs, any→any     | CC25 + mined data     | SentencePiece 250 k | No (src/tgt codes in call) | ~2.3 GB      |
# | opus-mt   | Helsinki-NLP/opus-mt-zh-en               | 74 M     | Transformer-base (6-6)    | zh → en only          | OPUS corpus           | SentencePiece 32 k  | No                         | ~298 MB      |
# | nllb      | facebook/nllb-200-distilled-600M         | 1.3 B    | Transformer (12-12, MoE)  | 200 langs, any→any    | NLLB-200 corpus       | SentencePiece 256 k | No                         | ~5 GB        |
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

bleu = load("sacrebleu", cache_dir='./cache')  # Word-level similarity
chrf = load("chrf", cache_dir='./cache')  # Character-level similarity


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
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()

# %%
def train_evaluate(model_name: str, src: str, tgt: str, seed: int, num_train=10, num_test=10, num_val=10):
    seed_all(seed=seed)
    need_prompt = model_name.startswith('t5')
    # data = load_dataset("wmt17", name="-".join([src, tgt]), split="train", cache_dir='./cache')

    lang_src = lang_map[src]
    lang_tgt = lang_map[tgt]
    tok, model = build_model(model_name, lang_src, lang_tgt)

    num_train_epochs = 10
    max_src, max_tgt = 128, 128
    batch_size = 64

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

    generation_config = GenerationConfig(
        max_length=max_tgt,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        decoder_start_token_id=tok.pad_token_id,
        # 显式禁止输出 extra_id
        suppress_tokens=[i for i in range(32128, 32228)]
    )

    args = Seq2SeqTrainingArguments(
        # output_dir='/content/output',
        eval_strategy="steps",
        eval_steps=500,
        logging_steps=100,
        save_steps=500,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=batch_size,
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
        generation_config=generation_config,
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



# %%
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import numpy as np
import evaluate

def train_evaluate(model_name: str, src: str, tgt: str, seed, num_train=32, num_test=20, num_val=20):
    
    # --- 1. Setup Model & Tokenizer ---
    # Ensure you are using the correct NLLB codes
    src_code = "zho_Hans" 
    tgt_code = "eng_Latn" 
    model_url = model_map[model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_url, src_lang=src_code, tgt_lang=tgt_code, cache_dir='./cache')
    model = AutoModelForSeq2SeqLM.from_pretrained(model_url, cache_dir='./cache')

    # --- 2. Fix Configuration for Training ---
    # Convert target code to ID
    target_lang_id = tokenizer.convert_tokens_to_ids(tgt_code)
    
    # This tells the model: "During training, start decoding with this token"
    model.config.decoder_start_token_id = target_lang_id
    
    # Optional: Fix forced_bos_token_id in config so it persists for inference
    model.config.forced_bos_token_id = target_lang_id

    # --- 3. Prepare Data (Using Corrected Encode) ---
    def encode(ex):
        # ... (Insert the corrected encode function from above) ...
        # For this snippet, assuming simpler inputs:
        inputs = [x[src] for x in ex["translation"]]
        targets = [x[tgt] for x in ex["translation"]]
        
        tokenizer.src_lang = src_code
        model_inputs = tokenizer(inputs, max_length=128, truncation=True)

        tokenizer.src_lang = tgt_code
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=128, truncation=True)
            
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    raw_train, raw_test, raw_val = split_train_test(data,
                                                    num_trains=num_train, num_test=num_test, num_val=num_val)
    # Why we tokenize the split instead of data? Since data is very large, and we only use a small part of it!!
    # So 3 times tokenizations is OK.
    tokenised_train = raw_train.map(encode, batched=True)
    tokenised_val = raw_val.map(encode, batched=True)
    tokenised_test = raw_test.map(encode, batched=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # --- 4. Training Arguments ---
    args = Seq2SeqTrainingArguments(
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=10,
        weight_decay=0.01,
        save_total_limit=2,
        predict_with_generate=True,
        fp16=True,
        logging_steps=10,
        eval_strategy="no", # or "steps" if you have eval data
        # REMOVE generation_config here. Rely on model.config.
    )

    # 2. **CRITICAL STEP**: Specify the target language ID
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        pred_str = tokenizer.batch_decode(preds, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)
        bleu_score  = bleu.compute(predictions=pred_str,
                                    references=[[r] for r in label_str])["score"]
        chrf_score  = chrf.compute(predictions=pred_str,
                                    references=label_str)["score"]
        print('preds', pred_str)
        return {"bleu": bleu_score, "chrf": chrf_score}
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenised_train, # Replace with your dataset
        eval_dataset=tokenised_val,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
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
# random.shuffle(params_list)

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




