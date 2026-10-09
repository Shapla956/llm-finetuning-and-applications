# Large Language Models: evaluation, fine-tuning and applications

Three notebooks on working with large language models in Python (Hugging Face Transformers,
PyTorch, PEFT/LoRA). They were written for a university course in the Master of Data Science,
Adelaide University, and run on a GPU (Google Colab).

| Notebook | What it does |
|---|---|
| [`01_llm_evaluation_classification_summarisation.ipynb`](01_llm_evaluation_classification_summarisation.ipynb) | Evaluates pretrained models on two tasks. GPT-2 classifies Yelp reviews (`Yelp/yelp_review_full`). T5-small and BART-large-CNN summarise news articles (`abisee/cnn_dailymail`) and are compared with ROUGE. |
| [`02_lora_finetuning_classification_dialogue.ipynb`](02_lora_finetuning_classification_dialogue.ipynb) | Fine-tunes models with LoRA. DistilBERT and RoBERTa classify news topics (`fancyzhx/ag_news`). GPT-2 and DialoGPT-small generate dialogue (`AlekseyKorshuk/persona-chat`). Compares encoder and decoder models before and after fine-tuning. |
| [`03_llm_ecommerce_applications.ipynb`](03_llm_ecommerce_applications.ipynb) | Three eCommerce uses of LLMs: advertisement generation from product images (BLIP-2 and Qwen2-VL), product recommendation over tables (TableLLM and Mistral-7B-Instruct), and customer review analysis with sentiment and summaries (Qwen3-4B). |

## Tools
Python, PyTorch, Hugging Face Transformers and Datasets, PEFT (LoRA), bitsandbytes (4-bit quantisation),
evaluate, ROUGE, BLEU, pandas.

## Running
Open a notebook in Google Colab, choose a GPU runtime, and run the cells in order. Datasets and
models download automatically from the Hugging Face Hub. Some models are large, so a GPU with
enough memory is needed.

## Author
Shapla Akter, Master of Data Science, Adelaide University.
