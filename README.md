# 🪔 Tamil Named Entity Recognition (NER) with Gradio UI

NERpaarvai is a transformer-based NER model trained on the Naamapadam dataset using XLM-R + BiLSTM + FFNN. It identifies person, organization, and location entities in Tamil text.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-yellow)
![Gradio](https://img.shields.io/badge/Gradio-UI-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

---

##  Demo

**Live Hugging Face Space:**  
[👉 Click here to try the model](https://huggingface.co/spaces/Aranganathan-S/NERpaarvai)

---

## Features

- Entity recognition for Tamil text
- Powered by custom fine-tuned Transformer embeddings
- Clean and simple Gradio UI for easy interaction
- Token-wise highlighting of detected entities

---

## Model Info

The model was trained on the [Naamapadam](https://huggingface.co/datasets/ai4bharat/naamapadam) dataset using a BiLSTM + FFNN decoder on top of Transformer embeddings (IndicBERT or XLM-R). Evaluation was done on token-level classification metrics (F1, precision, recall).

---

## Tech Stack

- Python
- Hugging Face Transformers
- PyTorch
- Gradio
- Tokenizers

---

## Installation

```bash
git clone https://github.com/yourusername/tamil-ner-gradio.git
cd tamil-ner-gradio
python app.py
```

---

## Sample Output

![Gradio Screenshot]
<img width="1885" height="559" alt="image" src="https://github.com/user-attachments/assets/81c86a12-d85b-4cc8-9024-8426f8235f3f" />


---

## Repo Structure

```
.
├── app.py              # Gradio UI script
├── README.md           # This file
├── .gitattributes
└── LICENSE  
```

---

## License

MIT License. Feel free to use, modify, or contribute 🙌

---

## Credits

- Naamapadam Dataset - AI4Bharat  
- Hugging Face Transformers  
- Gradio
