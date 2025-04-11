## 🚀😼
FastAPIで作成した災害ツイートを判定するAPIを Cloud Run にデプロイしました。  
[🌐 APIを試す（Swagger UI）](https://distilbert-api-26127316042.asia-northeast1.run.app/docs)


災害関連ツイートを判定するAIアプリ  
[🌐 アプリを試す](https://distilbert-api-deploy-cztkh7p8qtj8tgbvm4szuv.streamlit.app/)

## 🔧 構成
- モデル：DistilBERT (Hugging Face Transformers)
- 推論：PyTorch
- UI：Streamlit
- デプロイ：Streamlit Cloud
- モデル配布：GCS + dotenv
