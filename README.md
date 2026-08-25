# 🔥 FireGuard

سیستم تشخیص و هشدار آتش‌سوزی

## ساختار پروژه

```
fireguard/
├── app.py                  # داشبورد اصلی
├── config.py               # تنظیمات و آستانه‌ها
├── database.py             # کار با دیتابیس
├── requirements.txt
├── sensors/
│   ├── esp32_reader.py    # دریافت داده از ESP32
│   └── data_validator.py  # اعتبارسنجی داده
├── features/
│   └── feature_builder.py # ساخت ویژگی‌ها
├── detection/
│   └── rule_engine.py     # موتور تشخیص
├── models/
│   └── predictor.py       # پیش‌بینی با مدل
├── forecast/
│   └── simple_trend.py    # پیش‌بینی روند
├── nasa/
│   └── firms_client.py    # NASA FIRMS API
├── alerts/
│   └── alert_manager.py   # مدیریت هشدارها
├── training/
│   └── train_model.py     # آموزش مدل
├── data/                  # داده‌ها
│   └── fireguard.db
└── saved_models/          # مدل‌های آموزش‌دیده
```

## نصب و اجرا

```bash
# نصب وابستگی‌ها
pip install -r requirements.txt

# اجرا
python app.py
```
