# ATHENA

**Purpose**

Athena is a specialized LLM trained primarily on Wikipedia’s high‑quality, structured encyclopedia content. Its goal is to serve as a reliable knowledge assistant, providing accurate summaries and answers based on the latest wiki data.

---

## Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/dsbudziwojski/athena.git
   cd athena
   ```
2. **Install Athena and its dependencies**

   ```bash
   pip install .
   ```

> **Note:** Dependencies are declared in `pyproject.toml` under `[project].dependencies`.

---

## Features

* **Data Scraper**: `download_dump()` streams a Wikipedia XML dump (sample or full) as a BZ2File; `crawl(dump, max_pages)` parses MediaWiki XML, extracts revision text, and prints prettified HTML for up to `max_pages` pages.
* **Tokenizer**: `GeneralTokenizer(text_type, tokenizer_type)` wraps HuggingFace’s BPE (or stub) to train from lists or JSON files (`train_tokenizer`) and encode/decode sample strings (`test_tokenizer`).
* **Model (Future Work)**: Placeholder PyTorch modules; TODO: implement forward-pass, loss computation, backpropagation, and optimizer integration.

---

## Usage Examples

### Crawling Wikipedia

```python
from athena.crawler import download_dump, crawl

dump = download_dump()
crawl(dump, max_pages=5)
```

### Training the Tokenizer

```python
from athena.tokenizer import GeneralTokenizer

tkn = GeneralTokenizer(text_type="files", tokenizer_type="library")
tkn.train_tokenizer(["data/page1.json", "data/page2.json"])
```

### Testing the Tokenizer

```python
tkn.test_tokenizer("Hello, Athena!")
```

---

## Project Roadmap

| Focus                                        | Status         |
| -------------------------------------------- | -------------- |
| Data collection & Tokenizer                  | ✅ Completed    |
| NumPy forward-pass & loss computation        | 🔜 In progress |
| Manual backprop & optimizer                  | Pending        |
| Training loop, validation & sampling         | Pending        |
| Profiling & first CUDA kernel                | Pending        |
| Additional kernels, fine-tuning & deployment | Pending        |

---
