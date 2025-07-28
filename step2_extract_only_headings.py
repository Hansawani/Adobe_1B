import os
import json

INPUT_ROOT = "outputs"
OUTPUT_DIR = "extracted_headings"
os.makedirs(OUTPUT_DIR, exist_ok=True)

collection_sections = {}

# Traverse each collection (e.g., Collection 1, Collection 2)
for collection_name in os.listdir(INPUT_ROOT):
    collection_path = os.path.join(INPUT_ROOT, collection_name)
    if not os.path.isdir(collection_path):
        continue

    all_sections = []

    for root, dirs, files in os.walk(collection_path):
        for file_name in files:
            if not file_name.startswith("parallel_layout_results") or not file_name.endswith(".json"):
                continue

            input_path = os.path.join(root, file_name)
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            doc_path = data.get("document", "")
            doc_title = os.path.splitext(os.path.basename(doc_path))[0]

            for page in data.get("pages", []):
                for elem in page.get("elements", []):
                    if elem.get("type") == "paragraph_title":
                        heading = elem.get("text", "").strip()
                        if heading:
                            all_sections.append({
                                "heading": heading,
                                "level": "paragraph_title",
                                "doc_title": doc_title
                            })

    # Save all sections for this collection
    if all_sections:
        output_path = os.path.join(OUTPUT_DIR, f"{collection_name}.json")
        with open(output_path, "w", encoding="utf-8") as out_f:
            json.dump(all_sections, out_f, indent=2)

print("✅ Extracted one JSON file per collection in extracted_sections/")
