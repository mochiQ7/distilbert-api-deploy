# 1.モデル/APIコードの準備
model = torch.load("model/xxx.pth", map_location=torch.device("cpu"))

# 2. Dockerfileの作成
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
COPY TWT_main.py .
COPY model ./model
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8080
CMD ["uvicorn", "TWT_main:app", "--host", "0.0.0.0", "--port", "8080"]

# 3. Artifact Registry作成（初回のみ）
gcloud config set project [プロジェクトID]
gcloud services enable artifactregistry.googleapis.com
gcloud artifacts repositories create docker-repo \
  --repository-format=docker \
  --location=asia-northeast1 \
  --description="DistilBERT API for TWT"

# 4. Dockerビルド & プッシュ
docker build -t asia-northeast1-docker.pkg.dev/[プロジェクトID]/docker-repo/distilbert-api .
gcloud auth configure-docker asia-northeast1-docker.pkg.dev
docker push asia-northeast1-docker.pkg.dev/[プロジェクトID]/docker-repo/distilbert-api

# 5. Cloud Runへデプロイ
gcloud services enable run.googleapis.com
gcloud run deploy distilbert-api \
  --image asia-northeast1-docker.pkg.dev/[プロジェクトID]/docker-repo/distilbert-api \
  --platform managed \
  --region asia-northeast1 \
  --allow-unauthenticated \
  --memory 2Gi

# 6. テスト（Swagger UI）
アクセスURL:
https://[サービス名]-[ランダムID].asia-northeast1.run.app/docs