from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseModel):
    bot_token: str
    claude_api_key: str
    sqlite_path: str = "./audit.db"

def get_settings() -> Settings:
    return Settings(
        bot_token=os.environ["BOT_TOKEN"],
        claude_api_key=os.environ["CLAUDE_API_KEY"],
        sqlite_path=os.getenv("SQLITE_PATH", "./audit.db"),
    )
