from pathlib import Path
from datetime import datetime
import json
import shutil

BASE_DIR = Path(__file__).resolve().parents[1]

ACTIVE_DIR = BASE_DIR / "saved_models" / "active"
ARCHIVE_DIR = BASE_DIR / "saved_models" / "archive"

MANIFEST_FILE = (
    BASE_DIR / "saved_models" / "model_manifest.json"
)


def latest_file(directory):
    files = sorted(
        directory.glob("*.joblib"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    return files[0] if files else None


def main():
    print("=" * 70)
    print("🔥 FireGuard — Safe Model Rollback")
    print("=" * 70)

    archived_model = latest_file(ARCHIVE_DIR)

    if archived_model is None:
        raise FileNotFoundError(
            "No archived model available for rollback."
        )

    ACTIVE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    current_active = latest_file(ACTIVE_DIR)

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    if current_active is not None:
        backup_current = (
            ARCHIVE_DIR
            / f"{current_active.stem}_before_rollback_{timestamp}"
            f"{current_active.suffix}"
        )

        shutil.move(
            str(current_active),
            str(backup_current),
        )

    restored_path = (
        ACTIVE_DIR / archived_model.name
    )

    shutil.copy2(
        archived_model,
        restored_path,
    )

    manifest = {
        "active_model": str(restored_path),
        "rollback_source": str(archived_model),
        "rolled_back_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "status": "ROLLED_BACK",
    }

    with open(
        MANIFEST_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n🟢 ROLLBACK COMPLETE")
    print(f"Restored model: {restored_path}")


if __name__ == "__main__":
    main()