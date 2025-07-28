from paddleocr import LayoutDetection
from PIL import Image
import fitz  # PyMuPDF
import os
import json
from datetime import datetime
from io import BytesIO
import multiprocessing as mp
import gc
import time
import traceback
import numpy as np

os.environ['OMP_NUM_THREADS'] = str(min(8, os.cpu_count()))
os.environ['MKL_NUM_THREADS'] = str(min(8, os.cpu_count()))

# Global variable for model (loaded once per process)
layout_model = None

def init_worker():
    """Initialize the model once per worker process"""
    global layout_model
    try:
        print(f"🔄 Loading model in process {os.getpid()}...")
        layout_model = LayoutDetection(model_name="PP-DocLayout-L")
        print(f"✅ Model loaded successfully in process {os.getpid()}")
    except Exception as e:
        print(f"❌ Failed to load model in process {os.getpid()}: {e}")
        traceback.print_exc()
        raise

def process_page_worker(args):
    """Worker function that uses the pre-loaded model"""
    global layout_model
    
    page_num = args["page_num"]
    img = args["img"]
    pdf_path = args["pdf_path"]
    dpi = args["dpi"]
    
    try:
        print(f"🔍 Processing page {page_num + 1}... (PID: {os.getpid()})")
        
        if layout_model is None:
            raise RuntimeError("Layout model not initialized in worker process")
        
        # Convert PIL Image to numpy array (RGB format)
        img_array = np.array(img)
        
        start = time.time()
        layout_output = layout_model.predict(img_array, batch_size=1)
        layout_time = time.time() - start
        
        print(f"📊 Layout detection took {layout_time:.2f}s for page {page_num + 1}")
        print(f"🔍 Found {len(layout_output)} layout results for page {page_num + 1}")
        
        # Open PDF document for text extraction
        doc = fitz.open(pdf_path)
        
        results = []
        for det_result in layout_output:
            result = process_layout_result(det_result, doc, page_num, dpi)
            results.append(result)
            print(f"📄 Page {page_num + 1}: Found {len(result.get('elements', []))} elements")
        
        doc.close()
        
        # Clean up image from memory
        del img, img_array
        gc.collect()
        
        print(f"✅ Page {page_num + 1} completed")
        return results, layout_time
        
    except Exception as e:
        print(f"❌ Error processing page {page_num + 1}: {e}")
        traceback.print_exc()
        return [], 0.0

def extract_text_from_coordinates(pdf_doc, page_num, bbox, dpi=72):
    """Extract text from specific coordinates in PDF"""
    try:
        page = pdf_doc.load_page(page_num)
        page_rect = page.rect
        pix = page.get_pixmap(dpi=dpi)
        scale_x = page_rect.width / pix.width
        scale_y = page_rect.height / pix.height
        x1, y1, x2, y2 = bbox
        pdf_rect = fitz.Rect(
            x1 * scale_x, y1 * scale_y,
            x2 * scale_x, y2 * scale_y
        )
        text = page.get_text("text", clip=pdf_rect)
        return text.strip() if text else ""
    except Exception as e:
        print(f"⚠️ Error extracting text from coordinates: {e}")
        return ""

def process_layout_result(det_result, pdf_doc, page_num, dpi=72):
    """Process layout detection results for a single page"""
    result = {
        "page_number": page_num + 1,
        "elements": [],
        "element_counts": {}
    }
    
    try:
        print(f"🔄 Processing layout result for page {page_num + 1}: {type(det_result)}")
        
        boxes = det_result.get('boxes', [])
        if not boxes:
            print(f"⚠️ No boxes found in layout result for page {page_num + 1}")
            print(f"📋 Layout result keys: {list(det_result.keys()) if isinstance(det_result, dict) else 'Not a dict'}")
            return result
            
        print(f"📦 Found {len(boxes)} boxes for page {page_num + 1}")
        sorted_boxes = sorted(boxes, key=lambda b: b.get('coordinate', [0, 0])[1])
        
        for i, box in enumerate(sorted_boxes):
            label = box.get('label', 'unknown').lower()
            score = box.get('score', 0)
            coordinate = box.get('coordinate', [0, 0, 0, 0])
            
            result["element_counts"][label] = result["element_counts"].get(label, 0) + 1
            
            text_content = ""
            if label in ['doc_title', 'paragraph_title', 'text']:
                text_content = extract_text_from_coordinates(pdf_doc, page_num, coordinate, dpi)
                if text_content:
                    print(f"📝 Extracted text for {label}: {text_content[:50]}...")
            
            element = {
                "id": i + 1,
                "type": label,
                "confidence": float(round(score, 3)),  # Convert numpy float to Python float
                "text": text_content,
                "coordinates": [float(x) for x in coordinate]  # Convert numpy floats to Python floats
            }
            result["elements"].append(element)
            
    except Exception as e:
        print(f"❌ Error processing layout result for page {page_num + 1}: {e}")
        traceback.print_exc()
        result["error"] = str(e)
    
    return result

class FastPDFProcessor:
    def __init__(self, max_workers=None):
        if max_workers is None:
            self.max_workers = min(mp.cpu_count(), 4)  # Limit to 4 to avoid memory issues
        else:
            self.max_workers = max_workers
        
        print(f"🚀 Initialized FastPDFProcessor with {self.max_workers} workers")

    def convert_pdf_to_images_in_memory(self, pdf_path, dpi=72):
        """Convert PDF pages to in-memory images"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        doc = fitz.open(pdf_path)
        images = []

        print(f"☘️ Converting {len(doc)} pages to in-memory images (DPI: {dpi})...")

        for page_num in range(len(doc)):
            try:
                page = doc.load_page(page_num)
                pix = page.get_pixmap(dpi=dpi)
                img_bytes = BytesIO(pix.tobytes("png"))
                img = Image.open(img_bytes).convert("RGB")
                images.append((page_num, img))
                print(f"📄 Converted page {page_num + 1} ({img.size[0]}x{img.size[1]})")
                
                # Clean up pixmap
                pix = None
            except Exception as e:
                print(f"❌ Error converting page {page_num + 1}: {e}")

        doc.close()
        print(f"✅ Successfully converted {len(images)} pages")
        return images

    def process_pdf_parallel(self, pdf_path, output_dir="output", dpi=72):
        """Process PDF with parallel workers"""
        print("🚀 Starting parallel PDF layout processing...")
        t0 = time.time()

        try:
            # Convert PDF to images
            images = self.convert_pdf_to_images_in_memory(pdf_path, dpi)
            
            if not images:
                print("❌ No images to process!")
                return []
            
            # Prepare arguments for workers
            args = [
                {
                    "page_num": page_num, 
                    "img": img, 
                    "pdf_path": pdf_path, 
                    "dpi": dpi
                } 
                for page_num, img in images
            ]

            print(f"🔄 Starting {self.max_workers} worker processes...")
            
            # Process pages in parallel with model initialization
            with mp.Pool(processes=self.max_workers, initializer=init_worker) as pool:
                results = pool.map(process_page_worker, args)

            print("✅ All workers completed")

            # Collect results
            final_results = []
            layout_total = 0
            for page_results, layout_time in results:
                final_results.extend(page_results)
                layout_total += layout_time

            total_time = time.time() - t0

            print(f"📊 Processing summary:")
            print(f"   - Total pages: {len(images)}")
            print(f"   - Results collected: {len(final_results)}")
            print(f"   - Total layout time: {layout_total:.2f}s")
            print(f"   - Total processing time: {total_time:.2f}s")

            # Save results
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, "parallel_layout_results1.json")
            
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump({
                    "document": pdf_path,
                    "total_pages": len(images),
                    "processing_time": f"{total_time:.2f}s",
                    "layout_time_total": f"{layout_total:.2f}s",
                    "optimization": "multiprocessing+in-memory+shared-model",
                    "pages": final_results
                }, f, indent=2, ensure_ascii=False)

            print(f"\n🕰️ Done in {total_time:.2f} seconds. Layout time: {layout_total:.2f} seconds")
            print(f"📄 Results saved to: {output_file}")
            
            # Print extracted titles like your original code
            self.print_extracted_titles(final_results)
            
            # Clean up
            del images
            gc.collect()
            
            return final_results
            
        except Exception as e:
            print(f"❌ Error in process_pdf_parallel: {e}")
            traceback.print_exc()
            return []

    def print_extracted_titles(self, results):
        """Print extracted titles like the original code"""
        print("\n📋 EXTRACTED TITLES:")
        print("=" * 50)
        
        title_count = 0
        for page_result in results:
            for element in page_result.get('elements', []):
                if element['type'] in ['doc_title', 'paragraph_title'] and element['text'].strip():
                    print(f"📄 Page {page_result['page_number']}: {element['text'].strip()}")
                    title_count += 1
        
        print(f"\n🎯 Found {title_count} titles total")


if __name__ == "__main__":
    base_dir = "CHALLENGE_1B"
    processor = FastPDFProcessor(max_workers=4)  # Adjust as needed

    for collection_name in os.listdir(base_dir):
        collection_path = os.path.join(base_dir, collection_name)
        pdf_dir = os.path.join(collection_path, "PDFs")

        if os.path.isdir(pdf_dir):
            output_dir = os.path.join("outputs", collection_name)
            os.makedirs(output_dir, exist_ok=True)

            for filename in os.listdir(pdf_dir):
                if filename.endswith(".pdf"):
                    pdf_path = os.path.join(pdf_dir, filename)
                    pdf_name = os.path.splitext(filename)[0]
                    
                    print(f"\n📘 Processing PDF: {pdf_path}")
                    
                    # Save result to a separate file for each PDF
                    pdf_output_dir = os.path.join(output_dir, pdf_name)
                    os.makedirs(pdf_output_dir, exist_ok=True)
                    
                    processor.process_pdf_parallel(pdf_path, output_dir=pdf_output_dir)
