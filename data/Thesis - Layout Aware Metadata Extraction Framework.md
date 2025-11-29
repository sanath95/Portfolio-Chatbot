# Layout-Aware Metadata Extraction Framework

This is a theoretical documentation of the thesis research project. For executing the experiments, refer to: [CODE](./CODE.md).

---

## Table of contents
1. [Introduction](#introduction)
2. [Project Overview](#project-overview)
   1. [Gold Standard Metadata Curation](#gold-standard-metadata-curation)
   2. [Metadata Extraction with GROBID](#metadata-extraction-with-grobid)
   3. [Metadata Extraction with Language Models](#metadata-extraction-with-language-models)
   4. [Ground Truth for PDF Text Extraction](#ground-truth-for-pdf-text-extraction)
3. [Results](#results)
   1. [Evaluation of Metadata Extraction](#evaluation-of-metadata-extraction)
   2. [Evaluation of PDF Parsers](#evaluation-of-pdf-parsers)
4. [Error Analysis](#error-analysis)
5. [Conclusions](#conclusions)

---

## Introduction

**Motivation**

Academic publications are the cornerstone of scientific communication, yet their widespread use of PDF formats poses significant challenges for automated metadata extraction. Unlike structured formats such as XML or HTML, PDFs prioritize visual presentation over semantic structure, making it difficult to reliably capture critical bibliographic fields like titles, authors, affiliations, and DOIs.

**Overview**

This project investigates metadata extraction from scholarly PDFs by comparing two paradigms: layout-aware systems, which leverage structural and visual cues, and small-scale language models (SLMs), which rely on contextual reasoning. The work focuses on the construction of high-quality ground-truth datasets, and systematic evaluation of both approaches across accuracy, robustness, and computational efficiency.

Since accurately extracting the linear reading order of text from PDF documents is a prerequisite for downstream metadata extraction, this work also includes benchmarking of PDF parsers.

**Highlights**

* **Ground-Truth Resources** – Provides two curated datasets:
  * A page-level benchmark from DocBank for PDF parser evaluation.
  * A gold-standard collection of metadata for scholarly PDFs.
* **Parser Benchmarking** – Assesses five open-source PDF parsers on text fidelity, completeness, and logical reading order.
* **SLM Pipeline** – Demonstrates metadata extraction using prompt-engineered, schema-constrained outputs without model retraining.
* **Baseline Comparison** – Benchmarks layout-aware [GROBID](https://github.com/kermitt2/grobid) against SLM-based pipelines for accuracy, efficiency, and robustness.

**Metadata Fields**

Title, Authors, Affiliations, Email IDs, Publisher, Publication Date, DOI, Keywords, and Abstract.

---

## Project Overview

### Gold Standard Metadata Curation

To evaluate metadata extraction methods reliably, this project introduces a **novel multi-model consensus framework** for building a high-quality gold-standard dataset. Instead of relying on manual annotation or a single model, the pipeline systematically leverages diverse LLMs, schema validation, and adjudication to produce accurate and robust metadata records.

* **Document Selection** – 61 diverse born-digital PDFs sampled from publishers including PLOS, Elsevier, Springer, arXiv, PMLR, MDPI, and Frontiers Media to capture heterogeneous layouts and styles.
* **Text Extraction** – PDFs parsed with PyMuPDF into plain text for standardized processing.
* **Structured Outputs** – A [Pydantic](https://docs.pydantic.dev/latest/) response schema was passed via the API call to enforce JSON-formatted responses with mandatory bibliographic fields.
* **Multi-Model Annotations** – Metadata independently extracted by three LLMs: [o3-mini](https://openai.com/index/openai-o3-mini/), [Gemini-2.5-flash-lite](https://deepmind.google/models/gemini/flash-lite/) (Google), and [Grok-3-mini](https://grok.com/) (X-AI).
* **Consensus & Adjudication** – Fields with full agreement were accepted; disagreements escalated to [GPT-4.1](https://platform.openai.com/docs/models/gpt-4.1), which resolved only the contested fields using contextual reasoning.
* **Validation** – All outputs passed through Pydantic schema checks to enforce type correctness, field coverage, and structural consistency.
* **Final Benchmark** – Unified, machine-readable JSON records serving as a reliable reference for downstream evaluation.

This framework demonstrates how **AI diversity, consensus, and selective adjudication** can replace labor-intensive manual annotation while improving robustness and generalizability. The resulting dataset offers a scalable and reproducible benchmark for metadata extraction research.

<p align="center">
<img src="./assets/methodology_2.png" alt="multi model consensus framework" width="80%"/>
</p>

---

### Metadata Extraction with GROBID

This project integrates GROBID for **layout-aware metadata extraction** from scholarly PDFs. GROBID is deployed in Docker and accessed via its REST API using the official [Python client](https://github.com/kermitt2/grobid_client_python).

**Deployment**

Two Docker images are supported:

* **Full image** (`grobid/grobid:0.8.2-full`, \~8GB) – includes deep learning + CRF models, higher accuracy, supports GPU (but CPU-only in this setup).
* **Lightweight image** (`lfoppiano/grobid:0.8.2-crf`, \~500MB) – CRF-only, faster and smaller, lower accuracy for references and citations.

**Workflow**

1. **Input PDFs** are submitted to the GROBID server via the Python client.
2. **Metadata extraction** uses `processHeaderDocument`, returning results in TEI-XML format.
3. **Parsing** with [`lxml`](https://lxml.de/) + XPath extracts fields (title, authors, affiliations, publication date, publisher, DOI, keywords, abstract).
   * Dates are normalized to `DD-MM-YYYY`.
   * Missing fields (e.g., emails not supported by GROBID) are filled as empty strings.
4. **Output** is written to structured JSON files for downstream analysis.

---

### Metadata Extraction with Language Models

This project implements **language model–based metadata extraction** to compare against layout-aware systems like GROBID. Transformer-based models were used to extract bibliographic fields directly from raw PDF text.

**Models used:**
* [`Qwen/Qwen2.5-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
* [`microsoft/Phi-4-mini-instruct`](https://huggingface.co/microsoft/Phi-4-mini-instruct)
* [`meta-llama/Llama-3.2-3B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
* [`GPT-OSS-20B`](https://ollama.com/library/gpt-oss:20b) (served via [Ollama](https://ollama.com/))

**Inference optimizations:**
* 4-bit quantization via [`BitsAndBytesConfig`](https://github.com/bitsandbytes-foundation/bitsandbytes) for memory-efficient GPU usage.
* [PyTorch](https://pytorch.org/) monitoring of memory and execution time (`torch.cuda.max_memory_reserved`).

**PDF text extraction:**
* First page extracted with PyMuPDF, since it reliably contains bibliographic metadata while reducing token overhead.
* Only gpt-oss was able to handle two pages of extracted text and reliably give structured response.

**Workflow**

* **System prompts** included an explicit JSON schema defining required metadata fields.
* **User prompts** provided the extracted extracted text from PDF in a chat-style template.
* **Model outputs** were expected in structured JSON, validated against a Pydantic schema to ensure type correctness and required field coverage.
* **Error handling:** Incomplete or malformed outputs were caught and replaced with empty defaults to maintain consistency.

<p align="center">
<img src="./assets/implementation_2.png" alt="metadata extraction with language models" width="50%"/>
</p>

---

### Ground Truth for PDF Text Extraction

Accurate metadata extraction from scholarly PDFs requires a reliable textual representation, yet challenges such as publisher-specific layouts, multi-column formatting, and parser limitations make this non-trivial. To enable rigorous benchmarking of open-source PDF parsers, a ground-truth dataset that captures real-world document variability is constructed.

A curated subset of **101 first pages of arXiv papers (2014–2018)** was selected across disciplines (CS, statistics, mathematics, EE, economics).

* **Source Dataset** – Built on the [DocBank corpus](https://doc-analysis.github.io/docbank-page) with token-level annotations (fonts, bounding boxes, textual content).
* **Sampling Strategy** – Papers were chosen as the intersection of arXiv metadata and DocBank annotations, ensuring coverage of diverse layouts and styles while maintaining token-level precision.
* **Text Construction** – Token annotations were aggregated into structural blocks via a [YOLO-DocLayNet](https://github.com/ppaanngggg/yolo-doclaynet) document layout detector, then ordered using [LayoutReader](https://github.com/ppaanngggg/layoutreader) (based on LayoutLM) to reconstruct a natural reading sequence.
* **Evaluation Benchmark** – This dataset serves as the reference transcript for assessing five open-source parsers: [`PyMuPDF`](https://pymupdf.readthedocs.io/en/latest/), [`pypdfium2`](https://github.com/pypdfium2-team/pypdfium2), [`pdfminer.six`](https://github.com/pdfminer/pdfminer.six), [`PyPDF2`](https://pypi.org/project/PyPDF2/), and [`pdfalto`](https://github.com/kermitt2/pdfalto).

The resulting benchmark enables **quantitative, reproducible comparison** of text extraction fidelity, providing a foundation for downstream metadata extraction tasks.

<p align="center">
<img src="./assets/methodology_1.png" alt="construct ground truth for pdf extraction" width="60%"/>
</p>

---

## Results

### Evaluation of Metadata Extraction

Evaluated **layout-aware** (GROBID) vs **language model** approaches across nine metadata fields: *title, authors, affiliations, email addresses, publication date, publisher, DOI, keywords, abstract*.

**Systems Compared**

* **GROBID**: `BiLSTM-CRF (DL)` and `CRF (Wapiti)`.
* **SLMs**: `Qwen/Qwen2.5-3B-Instruct`, `microsoft/Phi-4-mini-instruct`, `meta-llama/Llama-3.2-3B-Instruct`, [`Qwen/Qwen3-4B-Base`](https://huggingface.co/Qwen/Qwen3-4B-Base).
* **LLM**: `gpt_oss_20b`.

**Metrics**

* **Levenshtein Distance**: short strings(*title, date, publisher, DOI*).
* **F1 Score**: lists (*authors, affiliations, emails, keywords*).
* **Cosine Similarity**: long string (*abstract*).

| fields           | metric           | grobid_dl | grobid_crf |    gpt_oss | phi4_mini | qwen3b |  qwen4b | llama3b |
|:-----------------|:-----------------|----------:|-----------:|-----------:|----------:|-------:|--------:|--------:|
| title            | LD |      0.67 |      3.04  |  0.78  |      4.26 |  0.75  | 24.11   |    **0.62** |
| doi              | LD |      2.81 |      2.81  |  **2.09**  |      6.03 |  4.42  |  7.45   |    3.24 |
| publication_date | LD |      1.70 |      1.68  |  **1.27**  |      6.72 |  1.31  |  3.49   |    1.75 |
| publisher        | LD |     14.34 |     14.34  |  **3.62**  |      8.44 | 10.88  |  6.70   |    7.18 |
| abstract         | CosSim |      **0.97** |      0.96  |  **0.97**  |      0.90 |  0.87  |  0.93   |    0.67 |
| authors          | F1         |      0.95 |      0.95  |  **0.96**  |      0.91 |  **0.96**  |  0.74   |    0.92 |
| affiliations     | F1         |      0.82 |      0.80  |  **0.91**  |      0.87 |  **0.91**  |  0.71   |    0.89 |
| keywords         | F1         |      **0.82** |      0.78  |  0.78  |      0.61 |  0.63  |  0.63   |    0.70 |
| email_ids        | F1         |      0    |     0      |  **0.98**  |      0.88 |  0.86  |  0.69   |    0.81 |

**Efficiency**

|    | model       | total time taken | max gpu used | max cpu used |
|---:|------------:|-----------------:|-------------:|-------------:|
|  1 | grobid_dl   | 90s              | -            | -            |
|  2 | grobid_crf  | 15s              | -            | -            |
|  3 | qwen4b      | 3,453s           | 4673 MiB     | 227 MiB      |
|  4 | qwen3b      | 1,777s           | 6380 MiB     | 1848 MiB     |
|  5 | phi4_mini   | 3,245s           | 4263 MiB     | 225 MiB      |
|  6 | llama3b     | 1,713s           | 3416 MiB     | 1500 MiB     |
|  7 | gpt_oss     | 19,851s          | -            | -            |

* GROBID is highly efficient compared to language models.
* Small language models required 30–55 minutes with significant GPU usage.
* GPT-OSS-20B exceeded five hours, making it impractical for large-scale deployment.

**Distribution of Scores**

![dist-group1](./assets/group1_boxplot.png)
![dist-group2](./assets/group3_boxplot.png)
![dist-abstract](./assets/group2_boxplot.png)

**Observations**
* **GPT-OSS-20B**
   * Best performance across most fields.
   * Ex: Abstract 0.97, Authors F1 0.96, Emails 0.98.
* **GROBID**
   * Strong on structured fields: Titles (Lev <1), Keywords F1 0.82.
   * Weak on variable fields: Affiliations 0.43, Publishers 14 edits.
* **Small LMs**
   * Qwen-3B best overall: Authors 0.96, Affiliations 0.91.
   * Llama-3B & Phi-4-mini moderate; Qwen-4B unstable.

![bar-lev](./assets/levenshtein_bar.png)
![bar-f1-cos](./assets/f1_bar.png)

---

### Evaluation of PDF Parsers

**Metrics**

* **CER / WER** (character & word error rates, via [`jiwer`](https://github.com/jitsi/jiwer)) → fidelity
* **BLEU / ROUGE-L** (via [`sacrebleu`](https://github.com/mjpost/sacrebleu) and [`rouge_score`](https://github.com/google-research/google-research/tree/master/rouge)) → order & semantic preservation

**Observation**
* **pymupdf** and **pypdfium2** → best overall balance, with low error rates and stable BLEU/ROUGE scores.
* **pdfalto** → weakest performance, widest error distribution.

![error-rates](./assets/error_rates.png)
![order-measures](./assets/order_measures.png)

---

## Error Analysis

The evaluation surfaced recurring error patterns across layout-aware systems (GROBID), small-scale language models (SLMs), and the large generative model (`gpt_oss_20b`). These errors highlight structural limitations, context restrictions, and model-specific behaviors.

**Context Length Limitations**
* For the small-scale language models—Qwen3B, Phi-4-mini, and Llama3B—only the first page of each article was passed as input. This restriction was necessary to remain within token limits, since providing additional pages frequently led to truncated or malformed outputs that broke the structured JSON schema. 
* In contrast, the larger gpt-oss model was able to process up to two pages without exceeding its capacity, still producing valid and complete structured metadata.

**Publisher Extraction**
* **GROBID** often produced expanded or collapsed forms of publisher names instead of exact strings, e.g., *“Public Library of Science PLOS”* vs *“PLOS ONE”*.
* These semantic mismatches were heavily penalized by Levenshtein distance despite being partially correct.
* Language models performed better, though still showed inconsistencies when publishers were abbreviated or embedded in headers/footers.

**Affiliations**
* **GROBID** struggled with affiliations in small fonts or visually embedded in layout elements, frequently omitting them.
* **SLMs** missed affiliations when they appeared beyond the first page, due to input length constraints.
* **Large LM** (20B) benefited from handling longer context and achieved higher recall but at extreme computational cost.

**Abstracts**
* **SLMs** suffered from truncated or incomplete abstracts because only the first page was used as input. Abstracts spanning multiple pages were partially or entirely missed.
* **Large LLM** managed multi-page input better, retaining more complete abstracts.
* **GROBID** excelled here, consistently capturing complete abstracts with high semantic similarity to ground truth.

**Keywords**
* **SLMs** were prone to hallucination: in several (19) cases, they generated keywords even when none existed in the source paper. These fabricated keywords often appeared plausible but undermined metadata reliability.
* **Large LLM** hallucinated far less frequently (only 3 cases), showing better adherence to source text.
* **GROBID**, while not hallucinating, occasionally failed to detect non-standard keyword placements or unusual formatting.

**Emails**
* **GROBID** does not support email extraction, consistently returning empty outputs.
* **SLMs** recovered many but not all email addresses, with variability across models.
* **Large LLM** achieved near-perfect recovery, highlighting the value of contextual inference in free text.

---

## Conclusions

**Summary**
* **Practical Advantage of LLMs:** Transformer-based LLMs can be quickly adapted to extract new metadata fields by simply adjusting the prompt—no retraining or annotated data required. This lowers resource costs and enables capabilities (e.g., extracting email addresses) that layout-aware tools like GROBID cannot handle.
* **Layout-aware systems**: precise and reliable on well-structured fields (abstracts, titles, keywords), but brittle when metadata is encoded with layout nuances (publishers, affiliations).
* **Small-scale LLMs**: competitive in recall but constrained by context length and prone to hallucination.
* **Large-scale LLMs**: deliver the most accurate and complete metadata but are computationally impractical for large-scale deployment.
Here’s a concise **README-style rewrite** of the *Limitations* and *Future Scope* sections:

**Limitations**

* **Dataset scope:** Only 61 PDFs and 101 DocBank pages; limited language diversity.
* **Format coverage:** Focus on born-digital PDFs; scanned documents not evaluated.
* **Context window constraints:** Small models restricted to first page; larger models only handle up to two pages → incomplete metadata capture.
* **Resource profiling gaps:** Incomplete measurement for large models (gpt-oss 20B).

**Future Scope**

* **Hybrid pipelines:** Use layout-aware tools (fast, structured fields) + instruction-tuned LMs (flexible, variable fields).
* **Dataset expansion:** Broaden coverage to more publishers, domains, languages, and OCR/scanned PDFs.
* **Handling long documents:** Apply sliding-window inference or retrieval-augmented generation (RAG) to capture full multi-page metadata.
* **Model adaptation:** Explore LoRA, prompt tuning, and domain-adaptive pretraining for cost-efficient accuracy gains.
* **Better metrics:** Move beyond exact matches to fuzzy/semantic similarity and measure downstream impact.

---
---