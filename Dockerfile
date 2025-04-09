# ベースイメージ
FROM python:3.10-slim

# 作業ディレクトリ作成
WORKDIR /app

# 必要ファイルをコピー（venv除く）
COPY requirements.txt .
COPY TWT_main.py .
COPY model ./model

# 依存パッケージをインストール
RUN pip install --no-cache-dir -r requirements.txt

# ポート解放（Cloud Runでも使えるように）
EXPOSE 8080

# FastAPIアプリ起動
CMD ["uvicorn", "TWT_main:app", "--host", "0.0.0.0", "--port", "8080"]