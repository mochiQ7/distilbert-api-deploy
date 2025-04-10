from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import torch.serialization
import logging
from google.cloud import storage
import os
import requests

# Cloud Storageからモデルをダウンロード
def download_model_from_gcs():
    model_url = "https://storage.googleapis.com/twt-model-bucket/distilbert_model_v2.pth"
    destination_file_name = "model/distilbert_model_v2.pth"

    if not os.path.exists(destination_file_name):
        os.makedirs("model", exist_ok=True)
        print("GCSからモデルをダウンロード中")
        response = requests.get(model_url)
        with open(destination_file_name, "wb") as f:
            f.write(response.content)
        print("モデルを保存しました")

download_model_from_gcs()  # 起動時にダウンロード

# モデルとトークナイザの読み込み
torch.serialization.add_safe_globals([DistilBertForSequenceClassification])
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = torch.load("model/distilbert_model_v2.pth", map_location=torch.device("cpu"), weights_only=False)
model.eval()


# ログの設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# FastAPIアプリの生成
app = FastAPI()

# 入力データの型定義
class TWTRequest(BaseModel):
    text: str

# 予測エンドポイント
@app.post("/predict")
def predict(tweet: TWTRequest):
    try:
        logging.info(f"予測リクエスト受信: {tweet.text}")

        inputs = tokenizer(tweet.text, return_tensors="pt", truncation=True, padding=True, max_length=128)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()

        logging.info(f"予測完了: {pred} (確率: {probs[0][pred].item():.4f})")

        return {
            "prediction": pred,
            "confidence": round(probs[0][pred].item(), 4)
        }

    except Exception as e:
        logging.error(f"予測エラー: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed.")

# ローカルでの起動用
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
