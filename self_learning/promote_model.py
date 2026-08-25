from pathlib import Path
from datetime import datetime
import json
import shutil

BASE_DIR = Path(__file__).resolve().parents[1]

ACTIVE_DIR = BASE_DIR / "saved_models" / "active"
CANDIDATE_DIR = BASE_DIR / "saved_models" / "candidates"
ARCHIVE_DIR = BASE_DIR / "saved_models" / "archive"

COMPARISON_FILE = (
    BASE_DIR / "data" / "self_learning" / "model_comparison.json"
)

MANIFEST_FILE = (
    BASE_DIR / "saved_models" / "model_manifest.json"
)


def latest_file(directory, pattern):
    files = sorted(
        directory.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    return files[0] if files else None


def main():
    print("=" * 70)
    print("🔥 FireGuard — Safe Model Promotion")
    print("=" * 70)

    if not COMPARISON_FILE.exists():
        raise FileNotFoundError(
            "Comparison result not found. "
            "Run compare_models.py first."
        )

    with open(
        COMPARISON_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        comparison = json.load(f)

    if (
        comparison.get("promotion_status")
        != "APPROVED_FOR_MANUAL_PROMOTION"
    ):
        raise RuntimeError(
            "Promotion blocked. Candidate was not approved."
        )

    candidate_path = Path(
        comparison["candidate_model"]
    )

    if not candidate_path.exists():
        raise FileNotFoundError(
            f"Candidate not found: {candidate_path}"
        )

    ACTIVE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ARCHIVE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    active_models = list(
        ACTIVE_DIR.glob("*.joblib")
    )

    archived_models = []

    for active_model in active_models:
        archive_path = (
            ARCHIVE_DIR
            / f"{active_model.stem}_archived_{timestamp}"
            f"{active_model.suffix}"
        )

        shutil.move(
            str(active_model),
            str(archive_path),
        )

        archived_models.append(str(archive_path))

    new_active_path = (
        ACTIVE_DIR
        / candidate_path.name.replace(
            "candidate_",
            "active_",
        )
    )

    shutil.copy2(
        candidate_path,
        new_active_path,
    )

    manifest = {
        "active_model": str(new_active_path),
        "promoted_from_candidate": str(candidate_path),
        "archived_models": archived_models,
        "promoted_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "promotion_status": "PROMOTED",
        "rollback_available": len(archived_models) > 0,
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

    print("\n🟢 MODEL PROMOTED")
    print(f"New active model : {new_active_path}")

    print("\nArchived models:")
    for path in archived_models:
        print(f"  - {path}")

    print("\nRollback is available with:")
    print("python self_learning/rollback_model.py")


if __name__ == "__main__":
    main()