import os
import shutil
from pathlib import Path

def move_files_to_folder(file_list, destination_folder):
    """
    انتقال فایل‌های مشخص شده به پوشه مقصد
    
    Args:
        file_list: لیست نام فایل‌ها
        destination_folder: مسیر پوشه مقصد
    """
    # ایجاد پوشه مقصد در صورت عدم وجود
    Path(destination_folder).mkdir(parents=True, exist_ok=True)
    
    moved_count = 0
    not_found_count = 0
    
    for filename in file_list:
        # بررسی وجود فایل در دایرکتوری جاری
        if os.path.isfile(filename):
            try:
                # ساخت مسیر کامل مقصد
                destination_path = os.path.join(destination_folder, filename)
                
                # انتقال فایل
                shutil.move(filename, destination_path)
                print(f"✓ انتقال: {filename} -> {destination_folder}")
                moved_count += 1
                
            except Exception as e:
                print(f"✗ خطا در انتقال {filename}: {e}")
        else:
            print(f"✗ فایل یافت نشد: {filename}")
            not_found_count += 1
    
    print("\n" + "="*50)
    print(f"تعداد فایل‌های انتقال یافته: {moved_count}")
    print(f"تعداد فایل‌های یافت نشده: {not_found_count}")

# لیست فایل‌های مورد نظر
files_to_move = [
    "real_firms_application_integration.py",
    "real_firms_production_service.py",
    "real_firms_production_smoke_test.py",
    "real_firms_production_inference_audit.py",
    "real_firms_production_readiness_audit.py",
    "real_firms_production_readiness_audit_v3.py",
    "real_firms_service.py",
    "real_firms_local_predict.py",
    "real_firms_final_threshold_test.py",
    "real_firms_inference_audit.py",
    "real_firms_model_training.py",
    "real_firms_test_evaluation.py",
    "real_firms_threshold_audit.py",
    "real_firms_threshold_config.py",
    "real_firms_threshold_inference.py",
    "train_real_firms_models.py",
    "build_real_firms_forecast_dataset.py"
]

# تنظیم پوشه مقصد - می‌توانید مسیر دلخواه را تغییر دهید
destination = "real_firms_backup"  # نام پوشه مقصد

# اجرای تابع انتقال
if __name__ == "__main__":
    print("شروع انتقال فایل‌ها...")
    print(f"پوشه مقصد: {destination}")
    print("="*50)
    
    move_files_to_folder(files_to_move, destination)