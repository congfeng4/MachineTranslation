# Machine Translation Project

This repository contains code for training and evaluating machine translation models on the WMT17 dataset, with a focus on Chinese-English translation. The project supports multiple pre-trained models and provides functionality for model training, inference, and result analysis.

## Requirements

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

The main dependencies include:
- transformers: For loading pre-trained models and tokenizers
- datasets: For loading and processing the WMT17 dataset
- sacrebleu & chrf: For evaluating translation quality
- sentencepiece: For tokenization
- accelerate: For training acceleration
- pandas & seaborn & matplotlib: For data visualization
- jsonlines: For result storage

## Project Structure

- `train.py`: Script for training and evaluating models with different parameters
- `infer.py`: Script for running inference on trained models and generating translations
- `requirements.txt`: List of required Python packages

## Supported Models

The project supports the following pre-trained models:

| Model Key | Hugging Face Model ID                     | Parameters | Architecture               |
|-----------|-------------------------------------------|------------|---------------------------|
| mbart     | facebook/mbart-large-50-many-to-many-mmt  | 610M       | Transformer-large (12-12) |
| opus-mt   | Helsinki-NLP/opus-mt-zh-en                | 74M        | Transformer-base (6-6)    |
| nllb      | facebook/nllb-200-distilled-600M          | 600M       | Transformer (12-12, MoE)  |


## Training

To train models, run the `train.py` script:

```bash
python train.py
```

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

The inference process:
1. Loads a subset of the WMT17 test set
2. Generates translations using each supported model
3. Saves the results to `translation_results.csv`
4. Includes code for dimension reduction and clustering analysis (incomplete)

## Evaluation Metrics

The project uses two common machine translation evaluation metrics:
- BLEU (Bilingual Evaluation Understudy): Measures n-gram overlap between translations and references
- chrF (Character F-score): Measures character n-gram overlap, useful for languages with complex tokenization

## Notes

- The code supports CUDA, MPS, and CPU devices automatically
- Model caching is enabled to avoid repeated downloads
- Early stopping is implemented to prevent overfitting
- Results are saved in JSON Lines format for easy analysis
- Seed fixing ensures reproducibility

For best performance, it is recommended to run the code on a GPU with sufficient memory (at least 8GB for smaller models, more for larger models like nllb).
