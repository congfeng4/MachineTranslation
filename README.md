# Machine Translation - CS5489 Machine Learning Project

This repository contains code for finetuning and evaluating machine translation models on the WMT17 dataset, 
with a focus on Chinese-English translation. 
The project supports multiple pre-trained models and provides functionality for model finetuning, inference, and result analysis.

## Requirements

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

The main dependencies include:
- transformers: For loading pre-trained models and tokenizers.
- datasets: For loading and processing the WMT17 dataset.
- sacrebleu & chrf: For evaluating translation quality.
- sentencepiece: For tokenization.
- accelerate: For training acceleration.
- evaluate: For loading the NLP metrics.
- pandas & seaborn & matplotlib: For data visualization.
- dotenv: For loading local environmental variables.
- jsonlines: For result storage.
- ipywidgets: For progressbar in notebooks.
- protobuf: Dependency of huggingface's libraries.
- scikit-learn: For TF-IDF feature extraction.
- umap-learn: For dimension reduction.
- wordcloud: For plotting word clouds.
- jieba: For tokenization of Chinese words.


## Project Structure

### Scripts and Notebooks
- `train.py` and `train.ipynb`: Script and notebook for training and evaluating models with different parameters.
- `infer.py` and `infer.ipynb`: Script and notebook for running inference on trained models and generating translations.
- `visualize.ipynb`: Notebook for visualizing the quantative experimental results (metrics).
- `data.ipynb`: Notebook for exploring and visualizing the WMT17 dataset (subset).

### Results
- `bilignual_cloud.png`: Word cloud visualization of a subset of WMT17 (zh-en) produced by `data.ipynb`.
- `cluster.jpg`: UMAP projections of 5 source sentences, corresponding target sentences, and predicted sentences by 3 models.
- `data-samples.csv`: The samples of WMT17 (zh-en) we use to perform data analysis in `data.ipynb`.
- `translation_results.csv`: The source data of `cluster.jpg` produced by `infer.ipynb`.
- `wmt17_results.jsonl`: JSON line file produced by `train.py`, containing the raw experimental setup and metric values.

### Misc
- `requirements.txt`: List of required Python packages.
- `README.md`: The description file of this project.
- `.env`: Environmental variables local to the running machine. You should prepare the following fields:
```.dotenv
HF_TOKEN=your huggingface token
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
CUDA_VISIBLE_DEVICES=gpu_0,gpu_1,gpu_2
HF_ENDPOINT=your huggingface mirror link
```

## Hardware Platform
- CUDA 12.8, NVIDIA A800, 80GB.
- Diskspace used: 18GB.
- Main memory: 881GB.
- CPU: single-thread program, N/A.

## Data Preparation

The data are downloaded from Hugging Face and preprocessed automatically as you run one of
`train.py`/`train.ipynb`/`infer.py`/`infer.ipynb`.

- Where to download the data: WMT17 (name of the dataset).
- How to preprocess it: No additional script is needed. Just run one of the above 4 scripts/notebooks.
- Where to place it: The dataset will be place in `./cache/wmt17`. Again, you don't need to download the data. The above
scripts/notebooks will download the data for you.


## Supported Models

The project supports the following pre-trained models:

| Model Key | Hugging Face Model ID                     | Parameters | Architecture              |
|-----------|-------------------------------------------|------------|---------------------------|
| mbart     | facebook/mbart-large-50-many-to-many-mmt  | 610M       | Transformer-large (12-12) |
| opus-mt   | Helsinki-NLP/opus-mt-zh-en                | 74M        | Transformer-base (6-6)    |
| nllb      | facebook/nllb-200-distilled-600M          | 600M       | Transformer (12-12, MoE)  |


## Training

To train models, run the `train.py` script:

```bash
python train.py
```

Or interactively, run `train.ipynb`

The training process:
1. Loads the WMT17 dataset for Chinese-English translation.
2. Sample a subset from WMT17 (zh-en) train split.
3. Splits the dataset subset into training, validation, and test sets.
4. Tokenizes the data using the model's tokenizer
5. Loads the pretrained models' weights from Hugging Face.
6. Finetunes the pretrained models using the Seq2SeqTrainer from Hugging Face.
7. Evaluates model performance using BLEU and chrF metrics.
8. Saves evaluation results to `wmt17_results.jsonl`.


The training parameters include:
- Number of training examples (16, 32, 64, 128, 256, 512, 1024, 2048)


The testing setup include:
- Number of test examples (20, 200)


## Inference

To generate translations using trained models, run the `infer.py` script:

```bash
python infer.py
```

Or interactively, run `infer.ipynb`

The inference process:
1. Loads a subset of the WMT17 test set.
2. Generates translations using each supported model.
3. Saves the results to `translation_results.csv`.
4. Dimension reduction and clustering analysis.


## Visualization

Run `visualize.ipynb` for interactive visualization.

## Evaluation Metrics

The project uses two common machine translation evaluation metrics:
- BLEU (Bilingual Evaluation Understudy): Measures n-gram overlap between translations and references.
- chrF (Character F-score): Measures character n-gram overlap, useful for languages with complex tokenization.
- Evaluation Loss: The cross-entropy loss on the testing set, averaging 3 independent samples.

## Notes

- The code supports CUDA, MPS, and CPU devices automatically.
- Model and dataset caching is enabled to avoid repeated downloads.
- Early stopping is implemented to prevent overfitting.
- Results are saved in JSON Lines format for easy analysis.

For best performance, it is recommended to run the code on a GPU with sufficient memory (at least 8GB for smaller models, 
more for larger models like nllb).
