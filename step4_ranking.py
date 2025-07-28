import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Paths
INPUT_DIR = "section_embeddings"
OUTPUT_DIR = "output_rankings"
MODEL_NAME = "paraphrase-MiniLM-L6-v2"

os.makedirs(OUTPUT_DIR, exist_ok=True)
model = SentenceTransformer(MODEL_NAME)

def get_persona_job(collection_name):
    input_json_path = os.path.join(collection_name, "challenge1b_input.json")
    if not os.path.exists(input_json_path):
        print(f"⚠️ {input_json_path} not found.")
        return "", ""
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data.get("persona", ""), data.get("job", "")

def is_redundant(new_emb, existing_embs, threshold=0.9):
    return any(cosine_similarity([new_emb], [e])[0][0] > threshold for e in existing_embs)

# Process all .jsonl files in section_embeddings/
for file in sorted(os.listdir(INPUT_DIR)):
    if not file.endswith(".jsonl"):
        continue

    collection_name = file.replace(".jsonl", "")
    persona, job = get_persona_job(collection_name)
    if not persona and not job:
        continue

    query = f"{persona}. {job}"
    query_emb = model.encode([query])[0]

    # Read all embedded entries
    embedded_sections = []
    with open(os.path.join(INPUT_DIR, file), "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                data["similarity"] = float(cosine_similarity([query_emb], [data["embedding"]])[0][0])
                embedded_sections.append(data)
            except Exception as e:
                print(f"⚠️ Error reading line: {e}")

    print(f"📄 {collection_name}: Found {len(embedded_sections)} sections")

    # Sort and filter top 10
    embedded_sections = sorted(embedded_sections, key=lambda x: x["similarity"], reverse=True)
    top_sections, seen = [], []

    for s in embedded_sections:
        emb = np.array(s["embedding"])
        if not is_redundant(emb, seen):
            seen.append(emb)
            s.pop("embedding", None)  # remove large field
            top_sections.append(s)
        if len(top_sections) == 10:
            break

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"ranked_{collection_name}.json")
    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump(top_sections, f_out, indent=2)

    print(f"✅ Saved top 10 sections to {output_path}")
