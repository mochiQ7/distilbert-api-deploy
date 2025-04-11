import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import os
import requests
from dotenv import load_dotenv

# 背景色＋ボタン色のCSSカスタマイズ
st.markdown(
    """
    <style>
    body {
        background-color: #f0f8ff;
    }
    div.stButton > button:first-child {
        background-color: #ffc0cb;
        color: black;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

load_dotenv()
MODEL_URL = os.getenv("MODEL_URL")
MODEL_PATH = "model/distilbert_model_v3.pth"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("モデルをダウンロード中...")
        os.makedirs("model", exist_ok=True)
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        print("モデルダウンロード完了！")

download_model()

# モデルとトークナイザの読み込み
@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    model.load_state_dict(torch.load("model/distilbert_model_v3.pth", map_location=torch.device("cpu")))
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# UI表示
st.markdown("<h1 style='text-align: center;'>🌪️ 災害ツイート判定アプリ ✨</h1>", unsafe_allow_html=True)
st.markdown("このツールは、ツイートが災害に関係あるかどうかをAIが予測します 💡<br>たった1行で、緊急性を見抜く！😼", unsafe_allow_html=True)

tweet = st.text_area("😼ツイートを入力してね（English only):")

if st.button("判定する！"):
    if tweet.strip() == "":
        st.warning("ツイートを入力してね！")
    else:
        inputs = tokenizer(tweet, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()

        if pred == 1:
            st.markdown("🔥 <b>📢 このツイートは <span style='color:red;'>災害に関する内容</span> です！</b> 🙀", unsafe_allow_html=True)
        else:
            st.markdown("🌈 <b>🕊️ このツイートは <span style='color:green;'>災害とは無関係</span> っぽいです。</b> 😸", unsafe_allow_html=True)
