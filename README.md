# Challenge 1b Project README

## Project Overview

This project processes PDF documents from three collections (`Collection 1`, `Collection 2`, `Collection 3`) to extract, structure, embed, and rank text and headings based on semantic similarity to a dynamically defined persona and job/task. The workflow is modular, with each step producing intermediate outputs for the next.

---

## Folder Structure

```
Challenge_1b/
│
├─ Collection 1/
│   ├─ challenge1b_input.json
│   ├─ challenge1b_output.json
│   └─ PDFs/
│       └─ *.pdf
├─ Collection 2/
│   ├─ challenge1b_input.json
│   ├─ challenge1b_output.json
│   └─ PDFs/
│       └─ *.pdf
├─ Collection 3/
│   ├─ challenge1b_input.json
│   ├─ challenge1b_output.json
│   └─ PDFs/
│       └─ *.pdf
│
├─ outputs/                # Intermediate: Layout detection results per PDF
│   └─ Collection X/
│       └─ PDF_NAME/
│           └─ parallel_layout_results1.json
│
├─ extracted_headings/     # Intermediate: Extracted headings per collection
│   └─ Collection X.json
│
├─ extracted_texts/        # Intermediate: Structured text blocks per collection/PDF
│   └─ Collection X/
│       └─ PDF_NAME/
│           └─ parallel_layout_results1.json
│
├─ section_embeddings/     # Intermediate: Section-level embeddings per collection
│   └─ Collection X.jsonl
│
├─ text_embeddings/        # Intermediate: Text block embeddings per collection
│   └─ Collection X.jsonl
│
├─ output_rankings/        # Final: Top ranked sections per collection
│   └─ ranked_Collection X.json
│
├─ output_rankings_text/   # Final: Top ranked text blocks per collection
│   └─ ranked_Collection X.json
│
├─ step1_paddle.py
├─ step2_extract_only_headings.py
├─ step2_extract_only_texts.py
├─ step3_embeddings.py
├─ step3_embeddings_text.py
├─ step4_ranking.py
├─ step4_ranking_text.py
└─ README.md
```

---

## Processing Pipeline

### 1. **PDF Layout Detection**
- **File:** `step1_paddle.py`
- **Input:** PDFs in each collection's `PDFs/` folder.
- **Output:** `outputs/Collection X/PDF_NAME/parallel_layout_results1.json`
- **Description:** Uses PaddleOCR to detect layout elements (titles, headings, text blocks) in each PDF page. Results are saved per PDF.

---

### 2. **Extract Headings**
- **File:** `step2_extract_only_headings.py`
- **Input:** `outputs/`
- **Output:** `extracted_headings/Collection X.json`
- **Description:** Traverses all layout results and extracts only the headings (`paragraph_title`) from each PDF, grouped by collection.

---

### 3. **Extract Structured Text Blocks**
- **File:** `step2_extract_only_texts.py`
- **Input:** `outputs/`
- **Output:** `extracted_texts/Collection X/PDF_NAME/parallel_layout_results1.json`
- **Description:** Extracts and structures text blocks under each heading from layout results, saving them per collection and PDF.

---

### 4. **Create Section Embeddings**
- **File:** `step3_embeddings.py`
- **Input:** `outputs/`
- **Output:** `section_embeddings/Collection X.jsonl`
- **Description:** For each collection, creates embeddings for each section (heading + content) using SentenceTransformer. Each line is a section with its embedding.

---

### 5. **Create Text Block Embeddings**
- **File:** `step3_embeddings_text.py`
- **Input:** `extracted_texts/`
- **Output:** `text_embeddings/Collection X.jsonl`
- **Description:** For each collection, creates embeddings for each text block using SentenceTransformer. Each line is a text block with its embedding.

---

### 6. **Rank Sections by Persona/Job Similarity**
- **File:** `step4_ranking.py`
- **Input:** `section_embeddings/`
- **Output:** `output_rankings/ranked_Collection X.json`
- **Description:** For each collection, reads persona and job/task from `challenge1b_input.json`, encodes them as a query, and ranks all sections by cosine similarity to the query. Removes redundant entries and saves the top 10 per collection.

---

### 7. **Rank Text Blocks by Persona/Job Similarity**
- **File:** `step4_ranking_text.py`
- **Input:** `text_embeddings/`
- **Output:** `output_rankings_text/ranked_Collection X.json`
- **Description:** Similar to above, but operates on text blocks. Deduplicates by text content, ranks by similarity to persona/job, and saves top unique entries per collection.

---

## Dynamic Persona & Job Extraction

- Both `step4_ranking.py` and `step4_ranking_text.py` dynamically read the persona and job/task from each collection's `challenge1b_input.json`:
    - For example, in `Collection 1/challenge1b_input.json`:
        - Persona: `"Travel Planner"` (from `persona.role`)
        - Job: `"Plan a trip of 4 days for a group of 10 college friends."` (from `job_to_be_done.task`)
    - The code builds the query as `"{persona}. {job}"` for semantic ranking.

---

## Output Summary

- **Intermediate outputs:**  
  - `outputs/` (layout detection results)
  - `extracted_headings/` (headings per collection)
  - `extracted_texts/` (structured text blocks per collection/PDF)
  - `section_embeddings/` (section-level embeddings)
  - `text_embeddings/` (text block embeddings)

- **Final outputs:**  
  - `output_rankings/` (top ranked sections per collection)
  - `output_rankings_text/` (top ranked text blocks per collection)

---

## Notes

- All intermediate and final outputs are grouped by collection (`Collection 1`, `Collection 2`, `Collection 3`).
- The code is modular; you can run each step independently.
- Ensure all required Python packages are installed (`paddleocr`, `sentence_transformers`, `scikit-learn`, `numpy`, `Pillow`, `PyMuPDF`).
- The persona/job extraction expects the structure in `challenge1b_input.json` as shown above.

---

## How to Run

1. Place all PDFs in the appropriate `Collection X/PDFs/` folders.
2. Run each step in order:
    - `python step1_paddle.py`
    - `python step2_extract_only_headings.py`
    - `python step2_extract_only_texts.py`
    - `python step3_embeddings.py`
    - `python step3_embeddings_text.py`
    - `python step4_ranking.py`
    - `python step4_ranking_text.py`
3. Final results will be in `output_rankings/` and `output_rankings_text/`.

---

## Contact

For issues or questions, please contact
