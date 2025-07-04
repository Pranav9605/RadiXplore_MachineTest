import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import os
import json
import argparse
import pandas as pd
from tqdm import tqdm

from src.pdf_parser   import parse_pdf
from src.ner          import load_spacy_model
from src.geolocation  import GeoLocator

def run_pipeline(pdf_dir, ner_model, gaz_file, output_file, gemini_key):

    nlp = load_spacy_model(ner_model)


    print(f"Loading gazetteer from {gaz_file}")
    gaz = pd.read_csv(gaz_file)
    print(f"Gazetteer loaded: {len(gaz)} entries")
    print(f"Gazetteer columns: {list(gaz.columns)}")
    

    print("\n Sample gazetteer entries:")
    print(gaz.head())

    geo = GeoLocator(gaz, gemini_key, threshold=0.4)  # Lowered threshold

    results = []
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    
    for fname in tqdm(pdf_files, desc="Processing PDFs"):
        path = os.path.join(pdf_dir, fname)
        pages = parse_pdf(path)
        
        for page_num, text in pages:
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ == "PROJECT":
                    context = ""
                    if ent.sent:
                        context = ent.sent.text
                    else:

                        start_idx = max(0, ent.start_char - 200)
                        end_idx = min(len(text), ent.end_char + 200)
                        context = text[start_idx:end_idx]
                    
                    print(f"\n📄 Processing: {fname} (Page {page_num})")
                    print(f"🏷️ Project: {ent.text}")
                    
                    coords = geo.infer(context, project_name=ent.text)
                    
                    results.append({
                        "pdf_file": fname,
                        "page_number": page_num,
                        "project_name": ent.text,
                        "context_sentence": context.strip(),
                        "coordinates": coords if coords else None,
                        "latitude": coords[0] if coords else None,
                        "longitude": coords[1] if coords else None
                    })


    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec) + "\n")

    total_records = len(results)
    with_coords = sum(1 for r in results if r["coordinates"] is not None)
    success_rate = (with_coords / total_records * 100) if total_records > 0 else 0
    
    print(f"\n✨ Pipeline complete!")
    print(f"📊 Total records: {total_records}")
    print(f"📍 Records with coordinates: {with_coords}")
    print(f"📈 Success rate: {success_rate:.1f}%")
    print(f"💾 Output saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_dir",   required=True)
    parser.add_argument("--ner_model", required=True)
    parser.add_argument("--gazetteer", required=True)
    parser.add_argument("--output",    required=True)
    parser.add_argument("--gemini_key",required=True)
    args = parser.parse_args()

    run_pipeline(
        pdf_dir=args.pdf_dir,
        ner_model=args.ner_model,
        gaz_file=args.gazetteer,
        output_file=args.output,
        gemini_key=args.gemini_key
    )