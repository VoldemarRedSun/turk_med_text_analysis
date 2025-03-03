from pathlib import Path
import spacy

TURK_TEXT_PATH = Path(__file__).parent.parent / "data_input" / "text-1-turk.txt"
TATAR_TEXT_PATH = Path(__file__).parent.parent / "data_input" / "text-1-tatar.txt"
EN_TEXT_PATH = Path(__file__).parent.parent / "data_input" / "text-1-en.txt"
OUTPUT_PATH = Path(__file__).parent.parent / "data_output" 

class SpacyNerModels:
    MEDICAL_ENGLISH_MODEL = "en_ner_bc5cdr_md"

def make_and_save_ner(model: str, input_text_path: str) -> None:
    nlp = spacy.load(model)
    text = Path(input_text_path).read_text()
    doc = nlp(text)
    output = ""
    for ent in doc.ents:
       output += f"Сущность: {ent.text}, Метка: {ent.label_}\n"
    output_path = OUTPUT_PATH / f"model_{model}_{Path(input_text_path).name}"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(output)


if __name__ == "__main__":
    make_and_save_ner(model=SpacyNerModels.MEDICAL_ENGLISH_MODEL, input_text_path=EN_TEXT_PATH)
