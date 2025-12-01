# %% [markdown]
# # Machine Translation - Inference

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
    GenerationConfig
)
import torch
import numpy as np
import random
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import jsonlines

dotenv.load_dotenv('.env')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# %% [markdown]
# ## Load Dataset WMT17
# 

# %%
target_language_code = "eng_Latn" # BCP-47 code for English


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
    'nllb': 'facebook/nllb-200-distilled-600M',
}
# Set the best model parameters here!
model_params = {
    'mbart': {'num_train': 2048},
    'opus-mt': {'num_train': 2048},
    'nllb': {'num_train': 2048},
}

# %% [markdown]
# ## Load Pretrained Model.
# 
# 

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
def clean_memory():
    import torch, gc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # 把未用缓存还给 CUDA
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()   # 把未用缓存还给 CUDA
    gc.collect()               # 再让 Python 回收一次

# %% [markdown]
# ## Run Inference

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

@torch.no_grad()
def translate(model, model_name, tok, input_data, src='zh'):
    model.eval()
    device = get_avail_device()

    out_f = []

    for sample in tqdm(input_data, desc="Translating"):
        zh = sample["translation"][src]
        print('zh', zh)
        inputs = tok(zh, return_tensors="pt").to(device)
        if model_name == 'nllb':
            target_lang_id = tok.convert_tokens_to_ids(target_language_code)
            pred = model.generate(**inputs, forced_bos_token_id=target_lang_id)
        else:
            pred = model.generate(**inputs)

        hyp = tok.decode(pred[0], skip_special_tokens=True)
        print('en', hyp)
        out_f.append(hyp.strip())

    return out_f


# %%
def build_model(model_name, src_lang, tgt_lang):
    model_name = model_map[model_name]
    tok = AutoTokenizer.from_pretrained(model_name, src_lang=src_lang, tgt_lang=tgt_lang, cache_dir='./cache')
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir='./cache')
    return tok, model

def train_model(model_name: str, src: str, tgt: str, num_train=10, num_test=10, num_val=10):
    need_prompt = model_name.startswith('t5')
    target_language_code = "eng_Latn" # BCP-47 code for English

    lang_src = lang_map[src]
    lang_tgt = lang_map[tgt]
    tok, model = build_model(model_name, lang_src, lang_tgt)

    target_lang_id = tok.convert_tokens_to_ids(target_language_code)
    model.config.decoder_start_token_id = target_lang_id
    # return model, tok

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
            zh_sent = ["translate Chinese to English:" + item for item in zh_sent]

        # 一次调用同时编码源端和目标端
        model_inputs = tok(
            en_sent,
            text_target=zh_sent,
            max_length=max_src,
            truncation=True,
            padding=False,          # 动态 padding，由 data_collator 完成
        )
        return model_inputs

    raw_train, raw_test, raw_val = split_train_test(dataset,
                                                    num_trains=num_train, num_test=num_test, num_val=num_val)
    # Why we tokenize the split instead of data? Since data is very large, and we only use a small part of it!!
    # So 3 times tokenizations is OK.
    tokenised_train = raw_train.map(encode, batched=True)
    tokenised_val = raw_val.map(encode, batched=True)
    tokenised_test = raw_test.map(encode, batched=True)

    data_coll = DataCollatorForSeq2Seq(tok, model=model)

    # 2. **CRITICAL STEP**: Specify the target language ID
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
        generation_config=GenerationConfig(
            forced_bos_token_id=target_lang_id,
        ) if 'nllb' in model_name else None,
        fp16=True,
    )
    
    # 2. **CRITICAL STEP**: Specify the target language ID
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


# %%
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import numpy as np

def train_model_nllb(model_name: str, src: str, tgt: str, num_train=32, num_test=20, num_val=20):
    
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

    raw_train, raw_test, raw_val = split_train_test(dataset,
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
    return model, tokenizer

# %% [markdown]
# ## Run Inference

# %%
tgt, src = 'en', 'zh'

translation_results = {'tgt': [sample['translation'][tgt] for sample in input_data],
                       'src': [sample['translation'][src] for sample in input_data]}

for key in model_map:
    print('model', key)
    num_train = model_params[key].get("num_train", 32)
    if key == 'nllb':
        model, tok = train_model_nllb(key, src='zh', tgt='en', num_train=num_train, num_test=20)
    else:
        model, tok = train_model(key, src='zh', tgt='en', num_train=num_train, num_test=20)

    translation_results[key] = translate(model, key, tok, input_data, src)

# %%
df = pd.DataFrame(translation_results)
df.to_csv('./translation_results.csv', index=False)

# %%
translation_results

# %% [markdown]
# ## Dimension Reduction & Clustering

# %%
df = pd.read_csv('./translation_results.csv')

# %%
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import umap
import matplotlib.pyplot as plt
import seaborn as sns

# | tgt | pred_modelA | pred_modelB | pred_modelC | ...

sentences, labels, markers = [], [], []
for idx, row in df.iterrows():
    sentences.append(row["tgt"])
    lb = row['tgt'][:20]
    labels.append(lb)
    markers.append("gt")          # ground-truth
    for model_name in model_map:
        if model_name in df.columns:
            sentences.append(row[model_name])
            labels.append(lb)
            markers.append(model_name)

# %%
# 2. TF-IDF
vectorizer = TfidfVectorizer(lowercase=True, stop_words=None, max_features=20_000)
X = vectorizer.fit_transform(sentences)   # shape: (n_sent, n_features)
fontsize = 10
S = 150

# 3. UMAP
reducer = umap.UMAP(n_neighbors=5, min_dist=0.1, metric="cosine", random_state=42)
XY = reducer.fit_transform(X)             # shape: (n_sent, 2)

# 4. Plot
plt.figure(figsize=(8, 5))
unique_samples = sorted(set(labels))
palette = sns.color_palette("husl", n_colors=len(unique_samples))
color_map = {sam: palette[i] for i, sam in enumerate(unique_samples)}

for i, (x, y) in enumerate(XY):
    color = color_map[labels[i]]
    if markers[i] == "gt":      
        plt.scatter(x, y, marker="*", s=S*2, c=[color], edgecolor="black")
    else:                       
        plt.scatter(x, y, c=[color], s=S, alpha=0.8)
        plt.text(x, y, markers[i], fontsize=fontsize, ha="left")

legend_elements = [plt.Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=color_map[sam], markersize=8, label=sam)
                   for sam in unique_samples]
plt.legend(handles=legend_elements, title="tgt", bbox_to_anchor=(1.05, 1), fontsize=fontsize, loc='best')
plt.title("UMAP projection of TF-IDF vectors (GT (*) vs. model predictions)")
plt.tight_layout()
plt.grid(True)
plt.savefig('./cluster.jpg')

# %% [markdown]
# For each source sentences, we run all three models with finetuning to generate a prediction.
# 
# The predictions of all models and the ground truth target sentence are first vectorized using TF-IDF and then reduced to two dimension with UMAP.
# 
# The two-dimensional space thereby is referred to the semantic space.
# 
# Finally, the points which correspond to the same source sentences are in the same color.
# 
# We can observe that the predictions of the same source sentences form clusters in the semantic space.
# 
# The predictions of opus-mt are also closer to the ground truth in more cases, followed by mbart, which coincidents with the models' performances.


