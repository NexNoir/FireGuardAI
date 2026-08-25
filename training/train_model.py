# training/train_model.py
"""
مدل ML تشخیص آتش (Hybrid Rule + Machine Learning)
داده‌های برچسب‌زده شده (از update_label) استفاده می‌کند
"""

import sys
from pathlib import Path

# اضافه کردن مسیر پروژه برای جلوگیری از خطای ModuleNotFound
sys.path.append(str(Path(__file__).parent.parent))

from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd
import numpy as np
from database import get_connection


class FireGuardModel:
    def __init__(self):
        self.model = None
        self.feature_columns = ['temperature', 'humidity', 'smoke', 'flame']
        self.is_trained = False

    def train(self):
        """آموزش مدل با داده‌های برچسب‌زده شده در دیتابیس"""
        print("=== شروع آموزش مدل ML ===")
        print("جستجوی داده‌های برچسب‌زده...")

        conn = get_connection()
        df = pd.read_sql_query('''
            SELECT temperature, humidity, smoke, flame, label 
            FROM sensor_readings 
            WHERE label IN ('confirmed_fire', 'confirmed_no_fire')
            ORDER BY timestamp DESC
            LIMIT 100
        ''', conn)
        conn.close()

        if len(df) < 10:
            print(f"⚠️ فقط {len(df)} نمونه کافی برای آموزش داریم. حداقل ۱۰ نمونه لازم است.")
            print("   مدل فقط با قوانین (Rule-based) کار خواهد کرد.")
            self.model = None
            self.is_trained = False
            return self

        # تبدیل برچسب به عدد
        df['target'] = df['label'].map({'confirmed_fire': 1, 'confirmed_no_fire': 0})

        X = df[self.feature_columns]
        y = df['target']

        print(f"تعداد نمونه‌های آموزش: {len(X)}")
        print("آموزش مدل RandomForest...")

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y)

        # ذخیره مدل
        joblib.dump(self.model, 'saved_models/fireguard_model.pkl')
        self.is_trained = True

        print("✅ مدل با موفقیت آموزش دید!")
        print("   مدل ذخیره شد در: saved_models/fireguard_model.pkl")
        return self

    def predict(self, reading):
        """پیش‌بینی یک رکورد"""
        if self.model is None:
            # fallback به قوانین موجود
            return self._rule_based_fallback(reading)

        features = np.array([[
            reading['temperature'],
            reading['humidity'],
            reading['smoke'],
            reading['flame']
        ]])

        prob = self.model.predict_proba(features)[0][1]  # احتمال آتش
        pred_class = 1 if prob > 0.5 else 0

        return {
            'class': 'fire' if pred_class == 1 else 'safe',
            'probability': float(prob),
            'reason': f"مدل ML ({prob*100:.1f}%)"
        }

    def _rule_based_fallback(self, reading):
        """قوانین پیش‌فرض اگر مدل آموزش دیده نبود"""
        if reading['flame'] == 1:
            return {'class': 'fire', 'probability': 0.95, 'reason': 'سنسور شعله فعال (fallback)'}
        if reading['smoke'] > 2200 or reading['temperature'] > 35:
            return {'class': 'fire', 'probability': 0.7, 'reason': 'دود یا دما بالا (fallback)'}
        return {'class': 'safe', 'probability': 0.1, 'reason': 'همه چیز نرمال'}


# ====================== تابع اصلی برای استفاده آسان ======================
def train():
    """تابع اصلی که همه دستورهای قبلی ما نیاز دارن"""
    model = FireGuardModel()
    model.train()
    return model


if __name__ == "__main__":
    model = train()
    print(f"\nوضعیت نهایی مدل: {'آموزش دیده' if model.is_trained else 'فقط قوانین فعال'}")