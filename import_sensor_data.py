# import_sensor_data.py
import pandas as pd
from database import init_database, insert_reading

def main():
    init_database()
    df = pd.read_csv("data/sensor_history.csv")

    print(f"در حال وارد کردن {len(df)} رکورد...")

    for _, row in df.iterrows():
        insert_reading(
            timestamp=str(row["time"]),
            temperature=float(row["temp"]),
            humidity=float(row["humidity"]),
            smoke=int(row["smoke"]),
            flame=int(row["flame"]),
            label="unverified"
        )

    print("تمام رکوردها با برچسب unverified وارد دیتابیس شدند.")
    print("حالا می‌توانید برچسب‌های دستی بزنید.")

if __name__ == "__main__":
    main()