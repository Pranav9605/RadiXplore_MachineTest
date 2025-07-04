import json
import spacy
from spacy.tokens import DocBin
from pathlib import Path
from typing import List, Tuple, Dict

LABEL = "PROJECT"

def convert_annotations_to_spacy(data_path: str, output_path: str):
    """
    Reads Label‑Studio JSON, writes a .spacy DocBin for training.
    """
    with open(data_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    nlp = spacy.blank("en")
    db = DocBin()

    for item in items:
        text = item["data"]["text"]
        ents = []
        for ann in item.get("annotations", []):
            for res in ann.get("result", []):
                if res["type"] == "labels" and LABEL in res["value"]["labels"]:
                    start = res["value"]["start"]
                    end   = res["value"]["end"]
                    ents.append((start, end, LABEL))

        doc = nlp.make_doc(text)
        spans = []
        for start, end, lbl in ents:
            span = doc.char_span(start, end, label=lbl)
            if span is not None:
                spans.append(span)
        doc.ents = spans
        db.add(doc)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    db.to_disk(output_path)


def train_spacy_ner(train_spacy_path: str, model_output: str, n_iter: int = 10):
    """
    Train a spaCy NER model from a .spacy DocBin.
    """
    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner")
    ner.add_label(LABEL)

    optimizer = nlp.begin_training()
    docbin = DocBin().from_disk(train_spacy_path)

    for i in range(n_iter):
        losses = {}
        for doc in docbin.get_docs(nlp.vocab):
            example = spacy.training.Example.from_dict(
                doc,
                {"entities": [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]}
            )
            nlp.update([example], sgd=optimizer, losses=losses)
        print(f"Iteration {i+1}/{n_iter} — Losses: {losses}")

    output_dir = Path(model_output)
    output_dir.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(output_dir)
    print(f" spaCy model saved to {output_dir}")


def load_spacy_model(model_path: str):
    """
    Load and return a trained spaCy model.

    """
    nlp = spacy.load(model_path)
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp    

