import os
import json

# === CONFIG ===
INPUT_ROOT = "outputs"
OUTPUT_ROOT = "extracted_texts"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

def process_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"❌ Skipping invalid JSON: {input_path}")
            return

    doc_title = data.get("document", os.path.basename(input_path))
    pages = data.get("pages", [])

    structured_sections = []
    current_title = None
    current_text_blocks = []
    current_page_number = None
    tracking_text_started = False

    for page in pages:
        page_number = page.get("page_number", None)
        elements = page.get("elements", [])
        for elem in elements:
            elem_type = elem.get("type")
            if elem_type == "paragraph_title":
                # Save the previous section
                if current_title and current_text_blocks:
                    structured_sections.append({
                        "title": current_title.strip(),
                        "text": " ".join(t.strip() for t in current_text_blocks),
                        "doc_title": doc_title,
                        "page_number": current_page_number
                    })
                # Reset for next
                current_title = elem.get("text", "")
                current_text_blocks = []
                current_page_number = None
                tracking_text_started = False

            elif elem_type == "text":
                if not tracking_text_started:
                    current_page_number = page_number
                    tracking_text_started = True
                current_text_blocks.append(elem.get("text", ""))

    # Final section
    if current_title and current_text_blocks:
        structured_sections.append({
            "title": current_title.strip(),
            "text": " ".join(t.strip() for t in current_text_blocks),
            "doc_title": doc_title,
            "page_number": current_page_number
        })

    # Write output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(structured_sections, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved: {output_path}")

# === Recursively process all JSONs ===
for dirpath, _, filenames in os.walk(INPUT_ROOT):
    for filename in filenames:
        if filename.endswith(".json"):
            input_path = os.path.join(dirpath, filename)
            relative_path = os.path.relpath(input_path, INPUT_ROOT)
            output_path = os.path.join(OUTPUT_ROOT, relative_path)
            process_file(input_path, output_path)
