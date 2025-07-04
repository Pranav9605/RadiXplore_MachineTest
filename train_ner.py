import argparse
from src.ner import convert_annotations_to_spacy, train_spacy_ner

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann", required=True, help="annotations.json")
    parser.add_argument("--model_out", required=True, help="where to save spaCy model")
    args = parser.parse_args()

    convert_annotations_to_spacy(args.ann, "data/train.spacy")

    train_spacy_ner("data/train.spacy", args.model_out)
    print(f"Model trained and saved to {args.model_out}")