    # Mining Project Intelligence System

This repository contains a complete pipeline for identifying mining project names from geological PDFs and inferring their approximate geographic coordinates.

---

## 📌 Overview

This system was developed as part of the **RadiXplore Candidate Coding Challenge**, and fulfills the following objectives:

- Extract project mentions from unstructured PDF reports.
- Predict geographic coordinates using semantic context + LLM.
- Output structured JSONL containing projects, location info, and context.

---

## 🔧 Setup Instructions

### 1. Clone Repository & Install Requirements

```bash
git clone <your-repo>
cd MiningProjectPipeline
pip install -r requirements.txt
```

### 2. Download spaCy Model

```bash
python -m spacy download en_core_web_sm
```

---

## 🚀 Running the Pipeline

### Step 1: Train the NER model

```bash
python train_ner.py \
  --ann data/sample_annotations.json \
  --model_out models/spacy_ner
```

### Step 2: Run Full Inference Pipeline

```bash
python main.py \
  --pdf_dir pdfs/ \
  --ner_model models/spacy_ner \
  --gazetteer data/gazetteer.csv \
  --output output/results.jsonl \
  --gemini_key <YOUR_GEMINI_API_KEY>
```

---

## 🛠 Tools & Technologies

- Python 3.10
- `spaCy` for NER model training and inference
- `SentenceTransformers` + `FAISS` for location matching
- Google Gemini API for fallback coordinate reasoning
- `PyMuPDF` and `pdfplumber` for PDF parsing

---

## 🧠 Model & Strategy

### NER
- Trained on Label Studio JSON annotations
- Uses `bert-base-cased` via `spaCy` pipeline
- High precision on real-world noisy documents

### Geolocation
- Uses semantic similarity via FAISS
- Falls back to Gemini API when context is ambiguous
- Coordinates are only added if confidence is high

---

## 📦 Output Format (`results.jsonl`)

Each line in the output is a JSON object:

```json
{
  "pdf_file": "example.pdf",
  "page_number": 3,
  "project_name": "Minyari Dome Project",
  "context_sentence": "Minyari Dome Project is located in the Paterson region of WA...",
  "coordinates": [-22.5, 117.5],
  "confidence": 0.89
}
```

---

## 📈 Evaluation Summary

| Metric                | Value   |
|-----------------------|---------|
| Total Records           | 168 |
| Unique Project Names    | 30 |
| With Coordinates        | 142 |
| Missing Coordinates     | 26 |
| Success Rate (%)        | 84.52 |

---

## 📄 Assumptions

- Gazetteer CSV must have columns: `place_name`, `latitude`, `longitude`
- If no location can be inferred, `coordinates` will be `null`
- PDFs may contain duplicate project mentions on multiple pages

---

## ✅ Notes

- All dependencies listed in `requirements.txt`
- Gemini API used in compliance with free-tier usage
- LLM fallbacks use robust prompt formatting and error handling

