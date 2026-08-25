from pathlib import Path
from datetime import datetime
import shutil
import zipfile

from production_config.settings import PROJECT_ROOT, DATABASE_FILE


BACKUP_ROOT = PROJECT_ROOT / "backups"

# فقط فایل‌ها و پوشه‌های مهم پروژه بکاپ گرفته می‌شوند.
BACKUP_TARGETS = [
    PROJECT_ROOT / ".env",
    PROJECT_ROOT / "data",
    PROJECT_ROOT / "saved_models",
    PROJECT_ROOT / "models",
    PROJECT_ROOT / "database",
    PROJECT_ROOT / "self_learning",
]


def copy_file_if_exists(source: Path, destination_dir: Path):
    """یک فایل را فقط در صورت وجود کپی می‌کند."""
    if source.exists() and source.is_file():
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / source.name
        shutil.copy2(source, destination)
        print(f"PASS | File backed up: {source.name}")
        return True

    print(f"WARN | File not found: {source}")
    return False


def copy_directory_if_exists(source: Path, destination_dir: Path):
    """یک پوشه را فقط در صورت وجود کپی می‌کند."""
    if source.exists() and source.is_dir():
        destination = destination_dir / source.name

        shutil.copytree(
            source,
            destination,
            dirs_exist_ok=True
        )

        print(f"PASS | Directory backed up: {source.name}")
        return True

    print(f"WARN | Directory not found: {source}")
    return False


def create_backup():
    print("=" * 70)
    print("FireGuard — Stage 14 Production Backup")
    print("=" * 70)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    backup_dir = BACKUP_ROOT / f"fireguard_backup_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    print(f"Backup directory: {backup_dir}")
    print("-" * 70)

    copied = 0
    warnings = 0

    # اول دیتابیس اصلی را جداگانه بررسی و بکاپ می‌گیریم.
    if DATABASE_FILE.exists():
        database_backup_dir = backup_dir / "database_backup"
        database_backup_dir.mkdir(parents=True, exist_ok=True)

        destination = database_backup_dir / DATABASE_FILE.name
        shutil.copy2(DATABASE_FILE, destination)

        print(f"PASS | Database backed up: {DATABASE_FILE.name}")
        copied += 1
    else:
        print(f"WARN | Database not found: {DATABASE_FILE}")
        warnings += 1

    # بکاپ سایر بخش‌های مهم.
    for target in BACKUP_TARGETS:
        if target == DATABASE_FILE:
            continue

        if target.exists():
            if target.is_file():
                files_dir = backup_dir / "files"
                if copy_file_if_exists(target, files_dir):
                    copied += 1
            elif target.is_dir():
                dirs_dir = backup_dir / "project_data"
                if copy_directory_if_exists(target, dirs_dir):
                    copied += 1
        else:
            print(f"WARN | Backup target not found: {target}")
            warnings += 1

    print("-" * 70)

    # ساخت ZIP برای نگهداری آسان‌تر.
    zip_path = BACKUP_ROOT / f"fireguard_backup_{timestamp}.zip"

    try:
        with zipfile.ZipFile(
            zip_path,
            "w",
            compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for file_path in backup_dir.rglob("*"):
                if file_path.is_file():
                    archive_name = file_path.relative_to(backup_dir)
                    archive.write(file_path, archive_name)

        print(f"PASS | ZIP created: {zip_path.name}")

    except Exception as e:
        print(f"FAIL | ZIP creation failed: {type(e).__name__}: {e}")
        return 1

    print("-" * 70)
    print(f"Backup items copied : {copied}")
    print(f"Warnings            : {warnings}")
    print(f"Backup folder       : {backup_dir}")
    print(f"Backup archive      : {zip_path}")
    print("=" * 70)

    print("STATUS: BACKUP COMPLETED")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(create_backup())

    except KeyboardInterrupt:
        print("\nSTATUS: BACKUP CANCELLED")
        raise SystemExit(1)

    except Exception as e:
        print("=" * 70)
        print("STATUS: BACKUP FAILED SAFELY")
        print(f"{type(e).__name__}: {e}")
        print("=" * 70)
        raise SystemExit(1)