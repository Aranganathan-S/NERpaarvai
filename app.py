
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

for pkg in ["torch", "transformers", "gradio", "requests", "numpy", "gdown"]:
    install(pkg)


import os
import torch
import torch.nn as nn
import gradio as gr
import gdown
from transformers import AutoTokenizer, AutoModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MODEL_PATH = "NERpaarvai.pt"
GDRIVE_FILE_ID = "10K-Wq8omarYFxDQqOwkFMUu0eePqebji"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

if not os.path.exists(MODEL_PATH):
    print("⬇️ Downloading model using gdown...")
    gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
    print("✅ Model downloaded successfully!")
else:
    print("✅ Model already exists.")


model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

label_list = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}


class NERModel(nn.Module):
    def __init__(self, num_labels):
        super(NERModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.bilstm = nn.LSTM(input_size=768, hidden_size=256, num_layers=1, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(512, num_labels)

        weights = torch.tensor([0.0338, 0.0437, 0.0273, 0.1162, 0.0271, 0.0325, 0.7194])
        self.class_weights = weights ** 0.5

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        lstm_out, _ = self.bilstm(outputs.last_hidden_state)
        lstm_out = self.dropout(lstm_out)
        logits = self.classifier(lstm_out)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(input_ids.device), ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
        return {"loss": loss, "logits": logits}


model = NERModel(num_labels=len(label_list)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("✅ Model loaded and ready!")


def predict_custom(text):
    words = text.strip().split()
    encoding = tokenizer(words, is_split_into_words=True, return_offsets_mapping=True,
                         return_tensors="pt", truncation=True, padding='max_length', max_length=128)
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    word_ids = encoding.word_ids()
    offset_mapping = encoding["offset_mapping"][0]

    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = output["logits"]
        predictions = torch.argmax(logits, dim=-1).squeeze().cpu().numpy()

    final_output = []
    previous_word_idx = None
    for idx, word_idx in enumerate(word_ids):
        if word_idx is None or word_idx == previous_word_idx:
            continue
        label = id2label[predictions[idx]]
        final_output.append(f"{words[word_idx]} → {label}")
        previous_word_idx = word_idx

    return "\n".join(final_output)


gr.Interface(
    fn=predict_custom,
    inputs=gr.Textbox(lines=2, label="தமிழ் உரையை உள்ளிடுக (Enter Tamil Text)", placeholder="உதாரணம்: அமித்ஷா இன்று சென்னை வந்தார்"),
    outputs=gr.Textbox(label="NER பெறுபாடு ( Output )"),
    title="🪔 NERpaarvai- நேர்பார்வை    - Tamil Named Entity Recognition",
    description="🔍 XLM-RoBERTa + BiLSTM NER model fine-tuned on Naamapadam dataset"
).launch()
