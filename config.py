# config.py
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "fireguard.db"
SAVED_MODELS_DIR = BASE_DIR / "saved_models"

# آدرس سنسور (اگر بعداً وصل شد)
ESP32_URL = "http://10.220.1.37/sensor"

# محدوده‌های معتبر سنسور
TEMP_MIN = -5.0
TEMP_MAX = 65.0
TEMP_IMPOSSIBLE_LOW = -40.0
TEMP_IMPOSSIBLE_HIGH = 85.0

HUMIDITY_MIN = 0.0
HUMIDITY_MAX = 100.0

SMOKE_MIN = 0
SMOKE_MAX = 4095
SMOKE_IMPOSSIBLE_HIGH = 5000

# آستانه‌های تشخیص لحظه‌ای (قابل تنظیم)
# این اعداد بر اساس تجربه اولیه با داده شما تنظیم شده‌اند
# و باید بعداً با برچسب انسانی کالیبره شوند
SMOKE_ELEVATED = 2200
SMOKE_HIGH = 2500
SMOKE_VERY_HIGH = 2800

TEMP_ELEVATED = 34.0
TEMP_HIGH = 37.0
TEMP_CRITICAL = 39.0

HUMIDITY_LOW = 25.0
HUMIDITY_CRITICAL = 18.0

ROLLING_WINDOW = 5