# Named Entity Recognition (NER) System — CoNLL-2003

End-to-end sequence labeling pipeline comparing **LSTM**, **BiLSTM**, **BiLSTM + CRF** (with GloVe 100d + Char-CNN for OOV handling), and **Fine-tuned BERT (`bert-base-cased`)** evaluated with **`seqeval`** metrics.

## Quick Start (Google Colab — T4 GPU)
1. Upload `ner_sequence_labeling.ipynb` to Google Colab.
2. Enable **T4 GPU** (`Runtime -> Change runtime type -> T4 GPU`).
3. Click `Runtime -> Run all` (~10–12 minutes total).
