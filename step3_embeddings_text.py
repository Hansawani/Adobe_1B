import os
import json
from sentence_transformers import SentenceTransformer

# === CONFIG ===
INPUT_DIR = "extracted_texts"
OUTPUT_DIR = "text_embeddings"
MODEL_NAME = "paraphrase-MiniLM-L6-v2"

os.makedirs(OUTPUT_DIR, exist_ok=True)
model = SentenceTransformer(MODEL_NAME)

for collection in sorted(os.listdir(INPUT_DIR)):
    collection_path = os.path.join(INPUT_DIR, collection)
    if not os.path.isdir(collection_path):
        continue

    embedded_texts = []

    # Go into each doc subfolder
    for doc_folder in sorted(os.listdir(collection_path)):
        doc_path = os.path.join(collection_path, doc_folder)
        if not os.path.isdir(doc_path):
            continue

        # Read each JSON file inside
        for file in sorted(os.listdir(doc_path)):
            if not file.endswith(".json"):
                continue

            file_path = os.path.join(doc_path, file)

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    sections = json.load(f)
            except json.JSONDecodeError:
                print(f"⚠️ Skipped corrupt JSON: {file_path}")
                continue

            if not isinstance(sections, list):
                print(f"⚠️ Unexpected structure in file: {file_path}")
                continue

            for section in sections:
                text = section.get("text", "").strip()
                title = section.get("title", "").strip()
                doc_title = section.get("doc_title", "").strip()
                page_number = section.get("page_number", None)

                if text:
                    embedded_texts.append({
                        "doc": doc_title,
                        "title": title,
                        "page": page_number,
                        "text": text,
                        "embedding": model.encode(text).tolist()
                    })

    print(f"📄 {collection}: Found {len(embedded_texts)} text elements")

    output_path = os.path.join(OUTPUT_DIR, f"{collection}.jsonl")
    with open(output_path, "w", encoding="utf-8") as f_out:
        for entry in embedded_texts:
            f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✅ Saved embeddings to {output_path}")
