# NER entities in medical texts
available texts lie in data_input/

## Installation
```bash
pip install poetry==1.5.1
git clone https://github.com/VoldemarRedSun/turk_med_text_analysis
cd turk_med_text_analysis
```
put the file here downloaded from [link](https://drive.google.com/drive/folders/19Ffnh59SIJQPK8XM42ehi_owWP63OGMp?usp=sharing)
```bash
poetry install
poetry shell
python <path_to_script_you_want_to_run>
```
## RUN NER WITH SPACY
```bash
python turkic_medicine_nlp/spacy_ner.py
```
## RUN NER WITH HuggingFace
### available ner models:
* GENERAL_TURKISH_MODEL = "akdeniz27/bert-base-turkish-cased-ner"
* MEDICAL_TURKISH_MODEL = "akdeniz27/mDeBERTa-v3-base-turkish-ner"
* MEDICAL_ENGLISH_MODEL_1 = "d4data/biomedical-ner-all"
* MEDICAL_ENGLISH_MODEL_2 = "Helios9/BioMed_NER"
### available texts:
* TURK_TEXT_PATH
* EN_TEXT_PATH
```bash
python turkic_medicine_nlp/hf_ner.py (run with default args: model_name= MEDICAL_ENGLISH_MODEL_2, text=EN_TEXT_PATH)
```
if you want change model and text modify command 
```bash
if __name__ == "__main__":
    make_and_save_ner(model_name = <model_name>, input_text_path=<text_path>)
```
and run
```bash
python turkic_medicine_nlp/hf_ner.py
```
Results of NER will lie in data_output/<model_name>_<text_name>.txt