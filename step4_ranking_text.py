import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
import re

# Paths
INPUT_DIR = "text_embeddings"
OUTPUT_DIR = "output_rankings_text"
MODEL_NAME = "paraphrase-MiniLM-L6-v2"

os.makedirs(OUTPUT_DIR, exist_ok=True)
model = SentenceTransformer(MODEL_NAME)

def get_persona_job(collection_name):
    """Fetch persona and job from challenge1b_input.json in the collection folder."""
    input_json_path = os.path.join(collection_name, "challenge1b_input.json")
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        persona = data.get("persona", "")
        job = data.get("job", "")
    return persona, job


def clean_text_for_deduplication(text):
    """Clean text to identify true duplicates"""
    cleaned = re.sub(r'\s+', ' ', text.strip())
    cleaned = re.sub(r'^[•\-\*]\s*', '', cleaned)
    cleaned = re.sub(r'^(tip|note|additional|tips?):?\s*', '', cleaned, flags=re.IGNORECASE)
    return cleaned.lower()

def create_text_hash(text):
    """Create hash for exact duplicate detection"""
    cleaned = clean_text_for_deduplication(text)
    return hashlib.md5(cleaned.encode('utf-8')).hexdigest()

def calculate_cosine_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings"""
    # Reshape embeddings to 2D arrays for sklearn
    emb1 = np.array(embedding1).reshape(1, -1)
    emb2 = np.array(embedding2).reshape(1, -1)
    
    # Calculate cosine similarity
    similarity = cosine_similarity(emb1, emb2)[0][0]
    return float(similarity)

def is_embedding_redundant(new_embedding, existing_embeddings, threshold=0.9):
    """Check if new embedding is too similar to existing ones"""
    if len(existing_embeddings) == 0:
        return False
    
    for existing_emb in existing_embeddings:
        similarity = calculate_cosine_similarity(new_embedding, existing_emb)
        if similarity > threshold:
            return True
    return False

# Process each JSONL file
for filename in sorted(os.listdir(INPUT_DIR)):
    if not filename.endswith(".jsonl"):
        continue

    print(f"\n" + "="*60)
    print(f"🔄 Processing: {filename}")
    print("="*60)
    
    collection_name = filename.split("_")[0].replace(".jsonl", "").strip()
    persona, job = get_persona_job(collection_name)
    query = f"{persona}. {job}"

    print(f"🎯 Query: {query}")
    print("🔄 Encoding query...")
    query_embedding = model.encode([query])[0]
    print(f"✅ Query embedding shape: {query_embedding.shape}")

    input_path = os.path.join(INPUT_DIR, filename)
    
    # Step 1: Load all entries and deduplicate by text content
    print("📖 Loading and deduplicating text entries...")
    unique_texts = {}  # hash -> entry with best similarity
    total_entries = 0
    
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 200 == 0:
                print(f"  Processed {line_num} lines...")
                
            try:
                entry = json.loads(line.strip())
                total_entries += 1
            except json.JSONDecodeError:
                continue

            text = entry.get("text", "").strip()
            embedding = entry.get("embedding", [])
            
            if not text or not embedding:
                continue

            # Convert embedding to numpy array
            embedding = np.array(embedding)
            if embedding.size == 0:
                continue

            # Create hash for deduplication
            text_hash = create_text_hash(text)
            
            # Calculate cosine similarity with query
            text_query_similarity = calculate_cosine_similarity(embedding, query_embedding)
            
            # Store entry with similarity score
            entry_with_similarity = {
                "doc": entry.get("doc", ""),
                "file": entry.get("file", ""),
                "page": entry.get("page", 0),
                "text": text,
                "embedding": embedding,
                "similarity": round(text_query_similarity, 4)
            }
            
            # Keep the version with highest similarity if duplicate text found
            if text_hash in unique_texts:
                if text_query_similarity > unique_texts[text_hash]["similarity"]:
                    unique_texts[text_hash] = entry_with_similarity
            else:
                unique_texts[text_hash] = entry_with_similarity

    unique_entries = list(unique_texts.values())
    print(f"📊 Total entries processed: {total_entries}")
    print(f"📊 Unique text blocks: {len(unique_entries)}")
    print(f"📊 Duplicates removed: {total_entries - len(unique_entries)}")

    # Step 2: Sort by cosine similarity (highest first)
    print("\n🔢 Sorting by cosine similarity scores...")
    unique_entries.sort(key=lambda x: x["similarity"], reverse=True)
    
    print(f"📈 Similarity range: {unique_entries[0]['similarity']:.4f} to {unique_entries[-1]['similarity']:.4f}")

    # Step 3: Select entries>0.2 while avoiding embedding redundancy
    print("\n🎯 Selecting entries with similarity > 0.2 ...")
    final_results = []

    for entry in unique_entries:
        if entry["similarity"] > 0.2:
            final_entry = {
                "doc": entry["doc"],
                "file": entry["file"], 
                "page": entry["page"],
                "text": entry["text"],
                "similarity": entry["similarity"]
            }
            final_results.append(final_entry)

    print(f"✅ Final selection: {len(final_results)} entries with similarity > 0.2")

    # Step 4: Save results
    collection_name = os.path.splitext(filename)[0]
    output_path = os.path.join(OUTPUT_DIR, f"ranked_{collection_name}.json")
    
    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump(final_results, f_out, indent=2, ensure_ascii=False)

    print(f"💾 Saved to: {output_path}")

print("\n" + "="*60)
print("🎉 ALL FILES PROCESSED SUCCESSFULLY!")
print("="*60)
print(f"📁 Results saved in: {OUTPUT_DIR}/")
print("🔍 Each file contains top unique text entries ranked by cosine similarity")