from pathlib import Path
from transformers import pipeline
from transformers import AutoModelForTokenClassification, AutoConfig, AutoTokenizer

TURK_TEXT_PATH = Path(__file__).parent.parent / "data_input" / "text-1-turk.txt"
EN_TEXT_PATH = Path(__file__).parent.parent / "data_input" / "text-1-en.txt"
OUTPUT_PATH = Path(__file__).parent.parent / "data_output" 

class HfNerModels:
    GENERAL_TURKISH_MODEL = "akdeniz27/bert-base-turkish-cased-ner"
    MEDICAL_TURKISH_MODEL = "akdeniz27/mDeBERTa-v3-base-turkish-ner"
    MEDICAL_ENGLISH_MODEL_1 = "d4data/biomedical-ner-all"
    MEDICAL_ENGLISH_MODEL_2 = "Helios9/BioMed_NER"


def make_and_save_ner(model_name: str, input_text_path: str | Path):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)

    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)
    text = Path(input_text_path).read_text()
    # text = "The patient was diagnosed with diabetes and prescribed metformin."
    # text = "Türk Dışişleri Bakanı, ABD'nin İsrail'e yönelik politikasında değişiklik yapılmasına izin verdi."
    results = ner_pipeline(text)
    output = ""
    for entity in results:
       output += f"Entity: {entity['word']}, Label: {entity['entity']}\n"
    output_path = OUTPUT_PATH / f"model_{model_name}_{Path(input_text_path).name}"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(output)

if __name__ == "__main__":
    make_and_save_ner(model_name = HfNerModels.MEDICAL_ENGLISH_MODEL_2, input_text_path=EN_TEXT_PATH)


