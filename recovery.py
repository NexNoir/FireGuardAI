from pathlib import Path
import shutil
import zipfile

from production_config.settings import PROJECT_ROOT


BACKUP_ROOT = PROJECT_ROOT / "backups"


def list_backups():
    """نمایش Backupهای موجود بدون تغییر هیچ فایلی."""
    if not BACKUP_ROOT.exists():
        return []

    backups = sorted(
        [
            path for path in BACKUP_ROOT.iterdir()
            if path.is_dir() and path.name.startswith("fireguard_backup_")
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    return backups


def show_backups(backups):
    print("=" * 70)
    print("FireGuard — Stage 14 Recovery")
    print("=" * 70)

    if not backups:
        print("NO BACKUPS FOUND")
        print(f"Expected directory: {BACKUP_ROOT}")
        return False

    print("Available backups:")
    print("-" * 70)

    for index, backup in enumerate(backups, start=1):
        print(f"{index}. {backup.name}")

    print("-" * 70)
    return True


def restore_file(source, destination):
    """
    Restore one file safely.
    Existing file is first copied to .before_restore.
    """
    if not source.exists():
        print(f"WARN | Backup source missing: {source}")
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        safety_copy = destination.with_suffix(
            destination.suffix + ".before_restore"
        )
        shutil.copy2(destination, safety_copy)
        print(f"PASS | Safety copy created: {safety_copy.name}")

    shutil.copy2(source, destination)
    print(f"PASS | Restored: {destination}")

    return True


def restore_database(backup_dir):
    """
    Restore فقط دیتابیس از Backup انتخاب‌شده.
    """
    backup_database = (
        backup_dir
        / "database_backup"
        / "fireguard_history.db"
    )

    target_database = (
        PROJECT_ROOT
        / "data"
        / "fireguard_history.db"
    )

    if not backup_database.exists():
        print("FAIL | Database file not found in selected backup")
        print(f"Expected: {backup_database}")
        return False

    print("-" * 70)
    print("RESTORE DATABASE ONLY")
    print(f"From: {backup_database}")
    print(f"To  : {target_database}")
    print("-" * 70)

    confirmation = input(
        "Type RESTORE_DATABASE to continue: "
    ).strip()

    if confirmation != "RESTORE_DATABASE":
        print("CANCELLED | No files were changed")
        return False

    return restore_file(backup_database, target_database)


def main():
    backups = list_backups()

    if not show_backups(backups):
        return 1

    print("Options:")
    print("1. Verify latest backup")
    print("2. Restore database from a backup")
    print("3. Exit")
    print("=" * 70)

    choice = input("Select: ").strip()

    if choice == "1":
        latest = backups[0]

        print(f"Checking: {latest}")

        database_file = (
            latest
            / "database_backup"
            / "fireguard_history.db"
        )

        if database_file.exists():
            print("PASS | Database backup exists")
            print(f"Size | {database_file.stat().st_size} bytes")
        else:
            print("FAIL | Database backup missing")
            return 1

        print("STATUS: BACKUP VERIFICATION COMPLETE")
        return 0

    if choice == "2":
        print("Available backups:")

        for index, backup in enumerate(backups, start=1):
            print(f"{index}. {backup.name}")

        try:
            selected = int(
                input("Backup number: ").strip()
            )

            if selected < 1 or selected > len(backups):
                print("FAIL | Invalid backup number")
                return 1

        except ValueError:
            print("FAIL | Please enter a valid number")
            return 1

        selected_backup = backups[selected - 1]

        success = restore_database(selected_backup)

        if success:
            print("=" * 70)
            print("STATUS: DATABASE RECOVERY COMPLETED")
            print("IMPORTANT: Run health_check.py before starting the system.")
            print("=" * 70)
            return 0

        return 1

    if choice == "3":
        print("Recovery tool closed. No files changed.")
        return 0

    print("FAIL | Invalid option")
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())

    except KeyboardInterrupt:
        print("\nRecovery cancelled safely.")
        raise SystemExit(1)

    except Exception as e:
        print("=" * 70)
        print("RECOVERY FAILED SAFELY")
        print(f"{type(e).__name__}: {e}")
        print("=" * 70)
        raise SystemExit(1)