from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import logging
from google.cloud import logging as gcloud_logging


# model, tokenizer読み込み
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
model = torch.load("model/distilbert_model_full.pth", map_location=torch.device("cpu"), weights_only=False)
model.eval()

# GoogleCloudLogging
#client = gcloud_logging.Client()
#client.setup_logging()

# ログの設定
logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s - %(levelname)s - %(message)s",
)

# FastAPIのインスタンスを生成
app = FastAPI()

# 入力用のデータモデル定義
class TWTRequest(BaseModel):
    text: str

# 予測用のAPIエンドポイント
@app.post("/predict")
def predict(tweet: TWTRequest):
    try:
        # ログにリクエストを記録
        logging.info(f"予測リクエスト受信: {tweet.text}")

        # テキストをトークナイズ
        inputs = tokenizer(tweet.text, return_tensors="pt", truncation=True, padding=True, max_length=128)

        # 推論実行
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
