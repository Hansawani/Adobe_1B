import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Paths
INPUT_DIR = "outputs"
OUTPUT_DIR = "section_embeddings"
MODEL_NAME = "paraphrase-MiniLM-L6-v2"

os.makedirs(OUTPUT_DIR, exist_ok=True)
model = SentenceTransformer(MODEL_NAME)

def normalize_text(text):
    return text.lower().strip()

for collection in sorted(os.listdir(INPUT_DIR)):
    collection_path = os.path.join(INPUT_DIR, collection)
    if not os.path.isdir(collection_path):
        continue

    embeddings = []

    for doc in sorted(os.listdir(collection_path)):
        doc_path = os.path.join(collection_path, doc)
        if not os.path.isdir(doc_path):
            continue

        section_text = ""
        current_heading = None

        for file in sorted(os.listdir(doc_path)):
            if not file.endswith(".json"):
                continue

            with open(os.path.join(doc_path, file), "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    continue

            elements = data.get("elements", [])
            for element in elements:
                el_type = element.get("type", "")
                el_text = element.get("text", "").strip()

                if el_type in ["paragraph_title", "doc_title"]:
                    if current_heading and section_text:
                        embeddings.append({
                            "doc": doc,
                            "section_title": current_heading,
                            "content": section_text.strip()
                        })
                    current_heading = el_text
                    section_text = ""

                elif el_type == "text" and current_heading:
                    section_text += el_text.strip() + "\n"

        # Save last section of the file
        if current_heading and section_text:
            embeddings.append({
                "doc": doc,
                "section_title": current_heading,
                "content": section_text.strip()
            })

    print(f"📄 {collection}: Found {len(embeddings)} sections")

    output_path = os.path.join(OUTPUT_DIR, f"{collection}.jsonl")
    with open(output_path, "w", encoding="utf-8") as f_out:
        for entry in embeddings:
            entry["embedding"] = model.encode(entry["content"]).tolist()
            f_out.write(json.dumps(entry) + "\n")

    print(f"✅ Saved embeddings to {output_path}")