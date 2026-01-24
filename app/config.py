from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseModel):
    bot_token: str
    openrouter_api_key: str  # новый ключ OpenRouter
    
    # для embeddings (пока оставляем yandex или можем переключить на openrouter)
    yc_api_key: str
    yc_folder_id: str
    yc_embedding_model_uri: str
    yc_embedding_dim: int = 256
    
    sqlite_path: str = "./audit.db"

def get_settings() -> Settings:
    return Settings(
        bot_token=os.environ["BOT_TOKEN"],
        openrouter_api_key=os.environ["OPENROUTER_API_KEY"],  # sk-or-v1-...
        yc_api_key=os.environ["YC_API_KEY"],
        yc_folder_id=os.environ["YC_FOLDER_ID"],
        yc_embedding_model_uri=os.environ["YC_EMBEDDING_MODEL_URI"],
        yc_embedding_dim=int(os.getenv("YC_EMBEDDING_DIM", "256")),
        sqlite_path=os.getenv("SQLITE_PATH", "./audit.db"),
    )
