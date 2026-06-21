import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")

BASE_DIR = Path(r"C:\проекты\aitam")
TEST_IMAGES_DIR = BASE_DIR / "plan_prediction" / "test_images"
MODEL_PATH = BASE_DIR / "plan_prediction" / "diffusion_checkpoints" / "best_model.pth"

