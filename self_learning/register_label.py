from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


# ============================================================
# FireGuard — Safe Label Registration
# READ/WRITE ONLY TO LABEL STORE
# NO MODEL TRAINING
# NO MODEL MODIFICATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "self_learning"
DATA_DIR.mkdir(parents=True, exist_ok=True)

INCOMING_FILE = DATA_DIR / "incoming_events.csv"
LABEL_FILE = DATA_DIR / "verified_labels.csv"

VALID_STATUSES = {
    "unverified",
    "confirmed_fire",
    "confirmed_no_fire",
}

REQUIRED_COLUMNS = [
    "event_id",
    "timestamp",
    "temperature",
    "humidity",
    "smoke",
    "flame",
    "label_status",
    "confirmed_by",
    "confirmation_source",
    "notes",
    "created_at",
]


# ============================================================
# UTILITIES
# ============================================================

def pause():
    input("\nPress Enter to continue...")


def normalize_timestamp(value: str) -> str:
    value = str(value).strip()

    try:
        dt = pd.to_datetime(value, errors="raise")
    except Exception as exc:
        raise ValueError(
            "Invalid timestamp. Use: YYYY-MM-DD HH:MM:SS"
        ) from exc

    return dt.strftime("%Y-%m-%d %H:%M:%S")


def make_event_id(timestamp: str) -> str:
    raw = f"fireguard|{timestamp}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def detect_file_format(path: Path) -> str:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return "CSV"

    if suffix == ".xls":
        return "XLS"

    if suffix == ".xlsx":
        return "XLSX"

    raise RuntimeError(
        f"Unsupported input format: {suffix}\n"
        "Supported formats: CSV, XLS, XLSX"
    )


# ============================================================
# SAFE FILE READER
# ============================================================

def read_csv_safely(path: Path) -> pd.DataFrame:
    """
    Robust CSV reader.

    Important:
    - delimiter is always a real string
    - never passes a format name as delimiter
    - supports common Windows encodings
    - automatically tries comma, semicolon and tab
    """

    encodings = [
        "utf-8-sig",
        "utf-8",
        "cp1256",
        "cp1252",
        "latin1",
    ]

    delimiters = [
        ",",
        ";",
        "\t",
        "|",
    ]

    errors = []

    for encoding in encodings:
        for delimiter in delimiters:

            try:
                df = pd.read_csv(
                    path,
                    encoding=encoding,
                    sep=delimiter,
                    engine="python",
                    dtype=str,
                )

                # Reject files that clearly did not parse correctly.
                if df.shape[1] < 2:
                    raise ValueError(
                        f"Only {df.shape[1]} column detected"
                    )

                df.columns = [
                    str(col).strip()
                    for col in df.columns
                ]

                return df

            except Exception as exc:
                errors.append(
                    f"- encoding={encoding}, "
                    f"delimiter={repr(delimiter)}: "
                    f"{type(exc).__name__}: {exc}"
                )

    raise RuntimeError(
        "Could not safely read the file as CSV.\n"
        f"File: {path}\n"
        "Attempts:\n"
        + "\n".join(errors)
        + "\n"
        "No model retraining was triggered by this error."
    )


def read_xls_safely(path: Path) -> pd.DataFrame:
    try:
        return pd.read_excel(
            path,
            engine="xlrd",
        )
    except ImportError as exc:
        raise RuntimeError(
            "Old XLS support requires xlrd.\n"
            "Install it with:\n"
            "pip install xlrd\n"
            "No model retraining was triggered by this error."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Could not safely read XLS file:\n{path}\n"
            f"{type(exc).__name__}: {exc}\n"
            "No model retraining was triggered by this error."
        ) from exc


def read_xlsx_safely(path: Path) -> pd.DataFrame:
    try:
        return pd.read_excel(
            path,
            engine="openpyxl",
        )
    except ImportError as exc:
        raise RuntimeError(
            "XLSX support requires openpyxl.\n"
            "Install it with:\n"
            "pip install openpyxl\n"
            "No model retraining was triggered by this error."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Could not safely read XLSX file:\n{path}\n"
            f"{type(exc).__name__}: {exc}\n"
            "No model retraining was triggered by this error."
        ) from exc


def read_input_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{path}"
        )

    detected = detect_file_format(path)

    print(f"Input file      : {path}")
    print(f"Detected format : {detected}")

    if detected == "CSV":
        return read_csv_safely(path)

    if detected == "XLS":
        return read_xls_safely(path)

    if detected == "XLSX":
        return read_xlsx_safely(path)

    raise RuntimeError(
        f"Unsupported detected format: {detected}"
    )


# ============================================================
# LABEL STORE
# ============================================================

def ensure_label_file():
    if not LABEL_FILE.exists():
        pd.DataFrame(
            columns=REQUIRED_COLUMNS
        ).to_csv(
            LABEL_FILE,
            index=False,
            encoding="utf-8-sig",
        )


def load_labels() -> pd.DataFrame:
    ensure_label_file()

    try:
        df = pd.read_csv(
            LABEL_FILE,
            encoding="utf-8-sig",
            dtype=str,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not read label database:\n"
            f"{LABEL_FILE}\n"
            f"{type(exc).__name__}: {exc}"
        ) from exc

    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    return df[REQUIRED_COLUMNS]


def save_labels(df: pd.DataFrame):
    # Atomic write.
    temp_file = LABEL_FILE.with_suffix(".tmp")

    df.to_csv(
        temp_file,
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_MINIMAL,
    )

    temp_file.replace(LABEL_FILE)


# ============================================================
# INCOMING EVENT SEARCH
# ============================================================

def find_event_in_source(timestamp: str):
    if not INCOMING_FILE.exists():
        return None

    df = read_input_file(INCOMING_FILE)

    if "timestamp" not in df.columns:
        return None

    timestamps = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    target = pd.to_datetime(timestamp)

    matches = df[timestamps == target]

    if matches.empty:
        return None

    return matches.iloc[0].to_dict()


def get_numeric_value(event, name, default=0.0):
    value = event.get(name, default)

    if value is None:
        return default

    try:
        return float(value)
    except Exception:
        return default


# ============================================================
# REGISTER EVENT
# ============================================================

def register_event(
    timestamp,
    temperature,
    humidity,
    smoke,
    flame,
    label_status,
    confirmed_by,
    confirmation_source,
    notes,
):
    timestamp = normalize_timestamp(timestamp)

    if label_status not in VALID_STATUSES:
        raise ValueError(
            "Invalid label status. "
            f"Allowed: {', '.join(sorted(VALID_STATUSES))}"
        )

    # Safety rule:
    # unverified does not count as training truth.
    if label_status == "unverified":
        confirmed_by = ""
        confirmation_source = ""

    labels = load_labels()

    event_id = make_event_id(timestamp)

    # Duplicate protection.
    if event_id in labels["event_id"].astype(str).values:
        raise RuntimeError(
            f"Duplicate event already registered:\n{event_id}"
        )

    source_event = find_event_in_source(timestamp)

    if source_event is not None:
        temperature = get_numeric_value(
            source_event,
            "temperature",
            temperature,
        )
        humidity = get_numeric_value(
            source_event,
            "humidity",
            humidity,
        )
        smoke = get_numeric_value(
            source_event,
            "smoke",
            smoke,
        )
        flame = get_numeric_value(
            source_event,
            "flame",
            flame,
        )

    created_at = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    row = {
        "event_id": event_id,
        "timestamp": timestamp,
        "temperature": float(temperature),
        "humidity": float(humidity),
        "smoke": float(smoke),
        "flame": float(flame),
        "label_status": label_status,
        "confirmed_by": confirmed_by,
        "confirmation_source": confirmation_source,
        "notes": notes,
        "created_at": created_at,
    }

    labels = pd.concat(
        [
            labels,
            pd.DataFrame([row]),
        ],
        ignore_index=True,
    )

    save_labels(labels)

    print("\nEVENT REGISTERED")
    print("--------------------------------------")
    print(f"Event ID       : {event_id}")
    print(f"Timestamp      : {timestamp}")
    print(f"Temperature    : {row['temperature']}")
    print(f"Humidity       : {row['humidity']}")
    print(f"Smoke          : {row['smoke']}")
    print(f"Flame          : {row['flame']}")
    print(f"Label          : {label_status}")
    print(f"Confirmed by   : {confirmed_by or 'N/A'}")
    print(f"Source         : {confirmation_source or 'N/A'}")
    print("--------------------------------------")
    print("Model retraining: NOT triggered")


# ============================================================
# CONFIRM EXISTING EVENT
# ============================================================

def confirm_existing_event():
    labels = load_labels()

    if labels.empty:
        print("\nNo registered events.")
        return

    unverified = labels[
        labels["label_status"] == "unverified"
    ]

    if unverified.empty:
        print("\nNo unverified events.")
        return

    print("\nUNVERIFIED EVENTS")
    print("=" * 70)

    for _, row in unverified.iterrows():
        print(
            f"{row['event_id']} | "
            f"{row['timestamp']} | "
            f"{row['notes']}"
        )

    event_id = input(
        "\nEvent ID to confirm: "
    ).strip()

    indexes = labels.index[
        labels["event_id"] == event_id
    ]

    if len(indexes) == 0:
        print("Event not found.")
        return

    idx = indexes[0]

    print("\n1. confirmed_fire")
    print("2. confirmed_no_fire")

    choice = input("Select: ").strip()

    if choice == "1":
        status = "confirmed_fire"
    elif choice == "2":
        status = "confirmed_no_fire"
    else:
        print("Invalid selection.")
        return

    confirmed_by = input(
        "Confirmed by: "
    ).strip()

    source = input(
        "Confirmation source: "
    ).strip()

    notes = input(
        "Notes: "
    ).strip()

    if not confirmed_by:
        print(
            "ERROR: Human/source confirmation is required."
        )
        return

    if not source:
        print(
            "ERROR: Confirmation source is required."
        )
        return

    labels.at[idx, "label_status"] = status
    labels.at[idx, "confirmed_by"] = confirmed_by
    labels.at[idx, "confirmation_source"] = source

    if notes:
        labels.at[idx, "notes"] = notes

    save_labels(labels)

    print("\nEVENT CONFIRMED")
    print("--------------------------------------")
    print(f"Event ID : {event_id}")
    print(f"Label    : {status}")
    print(f"By       : {confirmed_by}")
    print(f"Source   : {source}")
    print("--------------------------------------")
    print("Model retraining: NOT triggered")


# ============================================================
# SHOW EVENTS
# ============================================================

def show_events():
    labels = load_labels()

    if labels.empty:
        print("\nNo events registered.")
        return

    print("\nREGISTERED EVENTS")
    print("=" * 100)

    print(
        labels[
            [
                "event_id",
                "timestamp",
                "label_status",
                "confirmed_by",
                "confirmation_source",
            ]
        ].to_string(index=False)
    )


# ============================================================
# REGISTER NEW EVENT
# ============================================================

def register_new_event():
    print("\nREGISTER NEW EVENT")
    print("=" * 60)

    timestamp = input(
        "Timestamp (YYYY-MM-DD HH:MM:SS): "
    ).strip()

    print("\nLabel status:")
    print("1. unverified")
    print("2. confirmed_fire")
    print("3. confirmed_no_fire")

    choice = input("Select: ").strip()

    if choice == "1":
        status = "unverified"
    elif choice == "2":
        status = "confirmed_fire"
    elif choice == "3":
        status = "confirmed_no_fire"
    else:
        print("Invalid selection.")
        return

    # Sensor values are entered only when the event is not
    # available in the incoming source.
    print(
        "\nIf the timestamp exists in incoming_events.csv/XLS/XLSX, "
        "sensor values will be read automatically."
    )

    temperature_text = input(
        "Temperature (press Enter for 0): "
    ).strip()

    humidity_text = input(
        "Humidity (press Enter for 0): "
    ).strip()

    smoke_text = input(
        "Smoke (press Enter for 0): "
    ).strip()

    flame_text = input(
        "Flame (press Enter for 0): "
    ).strip()

    try:
        temperature = (
            float(temperature_text)
            if temperature_text
            else 0.0
        )

        humidity = (
            float(humidity_text)
            if humidity_text
            else 0.0
        )

        smoke = (
            float(smoke_text)
            if smoke_text
            else 0.0
        )

        flame = (
            float(flame_text)
            if flame_text
            else 0.0
        )

    except ValueError:
        print(
            "\nERROR: Sensor values must be numeric."
        )
        return

    confirmed_by = ""
    confirmation_source = ""

    if status != "unverified":
        confirmed_by = input(
            "Confirmed by: "
        ).strip()

        confirmation_source = input(
            "Confirmation source: "
        ).strip()

        if not confirmed_by or not confirmation_source:
            print(
                "\nERROR: confirmed_fire and "
                "confirmed_no_fire require "
                "a confirmer and source."
            )
            return

    notes = input(
        "Notes: "
    ).strip()

    try:
        register_event(
            timestamp=timestamp,
            temperature=temperature,
            humidity=humidity,
            smoke=smoke,
            flame=flame,
            label_status=status,
            confirmed_by=confirmed_by,
            confirmation_source=confirmation_source,
            notes=notes,
        )

    except Exception as exc:
        print("\nERROR")
        print(type(exc).__name__ + ":")
        print(exc)
        print(
            "\nNo model retraining was triggered by this error."
        )


# ============================================================
# MAIN MENU
# ============================================================

def main():
    while True:
        print("\n" + "=" * 70)
        print("🔥 FireGuard — Safe Label Registration")
        print("=" * 70)
        print("1. Register new event")
        print("2. Confirm existing event")
        print("3. Show events")
        print("4. Exit")
        print("=" * 70)

        choice = input("Select: ").strip()

        if choice == "1":
            register_new_event()
            pause()

        elif choice == "2":
            try:
                confirm_existing_event()
            except Exception as exc:
                print("\nERROR")
                print(type(exc).__name__ + ":")
                print(exc)
            pause()

        elif choice == "3":
            try:
                show_events()
            except Exception as exc:
                print("\nERROR")
                print(type(exc).__name__ + ":")
                print(exc)
            pause()

        elif choice == "4":
            print("\nExit.")
            break

        else:
            print("Invalid selection.")


if __name__ == "__main__":
    main()