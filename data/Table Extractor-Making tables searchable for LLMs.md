# Table Extractor - Making tables searchable for LLMs

## Table of Contents
1. [Introduction](#introduction)
2. [Documents](#documents)
3. [Archetypes and String Formats](#archetypes-and-string-formats)
4. [Native Libraries](#native-libraries)
5. [File Conversion](#file-conversion)
6. [Table Extraction Pipelines](#table-extraction-pipelines)
7. [Results](#results)
8. [Conclusions](#conclusions)

---

## Introduction

- The table extractor is an advanced software application that can automatically identify and retrieve tables from a variety of documents and extract the text data while maintaining the interconnections between the table’s cells.
- The goal of this project is to correctly extract text data that will allow LLMs to generate answers based on these documents.
- This project's main problem goes beyond simple text extraction; rather, it centres on maintaining and understanding the intricate relationships found in table data.
- This case study signifies a great collaboration with ONTEC AG, an Austrian custom solution developer that develops software and ML/AI based solutions individually for their respective customers. 

---

## Documents
- DOCX, 
- PDF, 
- scanned PDF,
- PPTX,
- XLSX

---

## Archetypes and String Formats

### Archetype 1: The simple table without row names

![archetype1](./assets/archetype1.png)

f"{Headline_1} {Row_1_Column_1}, {Headline_2} {Row_1_Column_2}, {Headline_3} {Row_1_Column_3}"

### Archetype 2: The simple table with row names

![archetype2](./assets/archetype2.png)

f"{Row_name_1} {Headline_1} {Row_1_Column_1}, {Row_name_1} {Headline_2} {Row_1_Column_2}"

### Archetype 3: The compound-headlines

![archetype3](./assets/archetype3.png)

f"{Super_Headline_1} {Headline_1}, {Super_Headline_1} {Headline_2}, {Super_Headline_2} {Headline_3}"

### Archetype 4: No headlines

![archetype4](./assets/archetype4.png)

f"{Row_1_Column_1} , {Row_1_Column_2}"

---

## Native Libraries

| Library | File Type | Documentation |
|---|---|---|
| python-docx | DOCX | [python-docx](https://python-docx.readthedocs.io/en/latest/) |
| openpyxl | XLSX | [openpyxl](https://openpyxl.readthedocs.io/en/stable/) |
| pptx | PPT | [python-pptx](https://python-pptx.readthedocs.io/en/latest/) |

### Advantages:
- Easy and fast to use (no machine learning involved).
- Text data can be efficiently extracted.
- Python-docx can also detect tables without borders.
- MIT License.

### Disadvantages:
- In DOCX, table (without borders) can be used to make the page look good, without containing table information. It is difficult to filter out this information.
- In XLSX, multiple tables in a sheet will be read as a single table.
- In PPT, if the table is an image, it is not detected as a table.
- In all cases, it is difficult to **filter out non table information**.

---

## File Conversion

Due to the inefficiencies of extracting table data from various documents (`.pptx`, `.xlsx`, `.docx`, and `.pdf`) using native libraries, we convert these documents into images. 

To facilitate this process, we have a **File Conversion API** that enables users to convert various file types into images.

[File Conversion Api](code/file-conversion/README.md)

These images can then be processed through the table extraction pipelines given below.

---

## Table Extraction Pipelines

1. [Pipeline 1](code/pipeline1/README.md)
2. [Pipeline 2](code/pipeline2/README.md)
3. [Pipeline 3](code/pipeline3/README.md)
4. [Pipeline 4](code/pipeline4/README.md)

---

## Results

Averages of `cosine similarity` for each pipeline are:

### **Archetype 1**: The simple table without row names

- Pipeline 1: 0.5433
- Pipeline 2: 0.2739
- Pipeline 3 - Tesseract: 0.6442
- Pipeline 3 - EasyOCR: 0.5745
- **Pipeline 4A: 0.8991**
- Pipeline 4B: 0.8911

>Archetype 2 is not being detected correctly. It is represented as Archetype 1.

### **Archetype 3**: The compound-headlines

- Pipeline 1: 0.6789
- Pipeline 2: 0.4068
- Pipeline 3 - Tesseract: 0.6772
- Pipeline 3 - EasyOCR: 0.6851
- Pipeline 4A: 0.7843
- **Pipeline 4B: 0.8431**

>Pipeline 4B is generating a table with combined headings.

### **Archetype 4**: No headlines

- Pipeline 1: 0.6148
- Pipeline 2: 0.0800
- Pipeline 3 - Tesseract: 0.8265
- **Pipeline 3 - EasyOCR: 0.8349**
- Pipeline 4A: 0.6800
- Pipeline 4B: 0.6463

## Conclusions

| Archetype | Pipeline |
|---|---|
| 1 | 4A |
| 2 | 4A |
| 3 | 4B |
| 4 | 3 - EasyOCR |

- **Edge Case 1**: Tables that have no borders

    The table transformer detection model is able to detect tables without borders as well.

- **Edge Case 2**: Tables that start on one page and proceed on other pages without repeating the headlines

    Not covered.

---