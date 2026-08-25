from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
from shapely.geometry import Point, shape
from shapely.prepared import prep


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

FIRMS_PATH = (
    BASE_DIR
    / "data"
    / "historical"
    / "firms"
    / "fire_archive_M-C61_790637.csv"
)

ECOREGION_PATH = (
    BASE_DIR
    / "data"
    / "historical"
    / "ecoregions"
    / "PA0407.geojson"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "historical"
    / "hyrcanian"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "hyrcanian_firms_fire_history.csv"
)

REPORT_JSON = (
    OUTPUT_DIR
    / "hyrcanian_firms_report.json"
)


# ============================================================
# LOAD OFFICIAL HYRCANIAN ECOREGION
# ============================================================

def load_hyrcanian_geometry():
    print("Loading official Hyrcanian ecoregion...")

    if not ECOREGION_PATH.exists():
        raise FileNotFoundError(
            f"Ecoregion file not found: {ECOREGION_PATH}"
        )

    with open(
        ECOREGION_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        geojson = json.load(f)

    features = geojson.get(
        "features",
        [],
    )

    if len(features) != 1:
        raise ValueError(
            "Expected exactly one Hyrcanian ecoregion feature, "
            f"got {len(features)}"
        )

    feature = features[0]

    properties = feature.get(
        "properties",
        {},
    )

    eco_name = properties.get(
        "eco_name"
    )

    eco_id = properties.get(
        "eco_id"
    )

    print(
        "Ecoregion:",
        eco_name,
    )

    print(
        "Ecoregion ID:",
        eco_id,
    )

    if eco_name != "Caspian Hyrcanian mixed forests":
        raise ValueError(
            "Wrong ecoregion loaded: "
            f"{eco_name}"
        )

    if int(eco_id) != 649:
        raise ValueError(
            f"Unexpected eco_id: {eco_id}"
        )

    geometry = shape(
        feature["geometry"]
    )

    print(
        "Geometry type:",
        geometry.geom_type,
    )

    print(
        "Geometry valid:",
        geometry.is_valid,
    )

    if not geometry.is_valid:
        geometry = geometry.buffer(0)

        print(
            "Geometry repaired:",
            geometry.is_valid,
        )

    return geometry, properties


# ============================================================
# LOAD FIRMS DATA
# ============================================================

def load_firms_data():
    print("\nLoading FIRMS historical fire data...")

    if not FIRMS_PATH.exists():
        raise FileNotFoundError(
            f"FIRMS file not found: {FIRMS_PATH}"
        )

    df = pd.read_csv(
        FIRMS_PATH,
        low_memory=False,
    )

    print(
        "Total FIRMS records:",
        len(df),
    )

    required_columns = [
        "latitude",
        "longitude",
        "acq_date",
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # --------------------------------------------------------
    # Clean coordinates
    # --------------------------------------------------------

    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce",
    )

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce",
    )

    before = len(df)

    df = df.dropna(
        subset=[
            "latitude",
            "longitude",
        ]
    ).copy()

    print(
        "Removed invalid coordinates:",
        before - len(df),
    )

    # --------------------------------------------------------
    # Parse date
    # --------------------------------------------------------

    df["acq_date"] = pd.to_datetime(
        df["acq_date"],
        errors="coerce",
    )

    before = len(df)

    df = df.dropna(
        subset=["acq_date"]
    ).copy()

    print(
        "Removed invalid dates:",
        before - len(df),
    )

    return df


# ============================================================
# SPATIAL FILTER
# ============================================================

def filter_points_inside_hyrcanian(
    df,
    geometry,
):
    print(
        "\nFiltering FIRMS points inside official "
        "Caspian Hyrcanian mixed forests boundary..."
    )

    prepared_geometry = prep(
        geometry
    )

    minx, miny, maxx, maxy = geometry.bounds

    print(
        "Hyrcanian bounds:",
        {
            "min_longitude": minx,
            "min_latitude": miny,
            "max_longitude": maxx,
            "max_latitude": maxy,
        }
    )

    # --------------------------------------------------------
    # Fast bounding-box prefilter
    # --------------------------------------------------------

    bbox_mask = (
        (df["longitude"] >= minx)
        & (df["longitude"] <= maxx)
        & (df["latitude"] >= miny)
        & (df["latitude"] <= maxy)
    )

    candidates = df.loc[
        bbox_mask
    ].copy()

    print(
        "Records inside bounding box:",
        len(candidates),
    )

    # --------------------------------------------------------
    # Exact point-in-polygon filter
    # covers() includes points exactly on the boundary.
    # --------------------------------------------------------

    inside_mask = []

    total = len(candidates)

    for index, row in enumerate(
        candidates.itertuples(index=False),
        start=1,
    ):
        point = Point(
            row.longitude,
            row.latitude,
        )

        inside_mask.append(
            prepared_geometry.covers(point)
        )

        if (
            index % 10000 == 0
            or index == total
        ):
            print(
                f"Processed {index:,} / {total:,} candidates"
            )

    hyrcanian_df = candidates.loc[
        inside_mask
    ].copy()

    print(
        "Records inside official Hyrcanian polygon:",
        len(hyrcanian_df),
    )

    return hyrcanian_df


# ============================================================
# ADD ANALYSIS COLUMNS
# ============================================================

def add_analysis_columns(df):
    print(
        "\nAdding temporal analysis columns..."
    )

    df = df.copy()

    df["year"] = (
        df["acq_date"].dt.year
    )

    df["month"] = (
        df["acq_date"].dt.month
    )

    df["day"] = (
        df["acq_date"].dt.day
    )

    df["season"] = df["month"].map(
        {
            12: "winter",
            1: "winter",
            2: "winter",

            3: "spring",
            4: "spring",
            5: "spring",

            6: "summer",
            7: "summer",
            8: "summer",

            9: "autumn",
            10: "autumn",
            11: "autumn",
        }
    )

    return df


# ============================================================
# BUILD REPORT
# ============================================================

def build_report(
    source_df,
    hyrcanian_df,
    properties,
):
    print(
        "\nBuilding statistical report..."
    )

    yearly_counts = (
        hyrcanian_df
        .groupby("year")
        .size()
        .to_dict()
    )

    monthly_counts = (
        hyrcanian_df
        .groupby("month")
        .size()
        .to_dict()
    )

    seasonal_counts = (
        hyrcanian_df
        .groupby("season")
        .size()
        .to_dict()
    )

    report = {
        "dataset": "NASA FIRMS MODIS C6.1",
        "spatial_filter": {
            "method": (
                "Exact point-in-polygon filtering"
            ),
            "ecoregion_name": properties.get(
                "eco_name"
            ),
            "ecoregion_id": int(
                properties.get(
                    "eco_id"
                )
            ),
            "boundary_file": str(
                ECOREGION_PATH.relative_to(
                    BASE_DIR
                )
            ),
        },
        "source": {
            "file": str(
                FIRMS_PATH.relative_to(
                    BASE_DIR
                )
            ),
            "total_iran_records": int(
                len(source_df)
            ),
        },
        "result": {
            "hyrcanian_records": int(
                len(hyrcanian_df)
            ),
            "percentage_of_source": round(
                (
                    len(hyrcanian_df)
                    / len(source_df)
                    * 100
                ),
                4
                if len(source_df) > 0
                else 0,
            ),
            "start_date": str(
                hyrcanian_df["acq_date"].min().date()
            )
            if not hyrcanian_df.empty
            else None,
            "end_date": str(
                hyrcanian_df["acq_date"].max().date()
            )
            if not hyrcanian_df.empty
            else None,
        },
        "statistics": {
            "yearly_counts": {
                str(k): int(v)
                for k, v in yearly_counts.items()
            },
            "monthly_counts": {
                str(k): int(v)
                for k, v in monthly_counts.items()
            },
            "seasonal_counts": {
                str(k): int(v)
                for k, v in seasonal_counts.items()
            },
        },
    }

    # Optional FIRMS statistics
    if "confidence" in hyrcanian_df.columns:
        confidence = pd.to_numeric(
            hyrcanian_df["confidence"],
            errors="coerce",
        )

        report["statistics"][
            "confidence"
        ] = {
            "mean": round(
                float(confidence.mean()),
                4,
            )
            if confidence.notna().any()
            else None,
            "median": float(
                confidence.median()
            )
            if confidence.notna().any()
            else None,
        }

    if "frp" in hyrcanian_df.columns:
        frp = pd.to_numeric(
            hyrcanian_df["frp"],
            errors="coerce",
        )

        report["statistics"][
            "frp"
        ] = {
            "mean": round(
                float(frp.mean()),
                4,
            )
            if frp.notna().any()
            else None,
            "median": round(
                float(frp.median()),
                4,
            )
            if frp.notna().any()
            else None,
            "max": round(
                float(frp.max()),
                4,
            )
            if frp.notna().any()
            else None,
        }

    return report


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_outputs(
    hyrcanian_df,
    report,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\nSaving filtered Hyrcanian dataset..."
    )

    output_df = hyrcanian_df.copy()

    output_df["acq_date"] = (
        output_df["acq_date"]
        .dt.strftime("%Y-%m-%d")
    )

    output_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print(
        "Dataset saved:",
        OUTPUT_CSV.resolve(),
    )

    print(
        "Dataset rows:",
        len(output_df),
    )

    print(
        "\nSaving report..."
    )

    REPORT_JSON.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "Report saved:",
        REPORT_JSON.resolve(),
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print(
        "FireGuard - Hyrcanian Historical FIRMS Dataset Builder"
    )
    print("=" * 60)

    geometry, properties = (
        load_hyrcanian_geometry()
    )

    source_df = (
        load_firms_data()
    )

    hyrcanian_df = (
        filter_points_inside_hyrcanian(
            source_df,
            geometry,
        )
    )

    hyrcanian_df = (
        add_analysis_columns(
            hyrcanian_df
        )
    )

    report = build_report(
        source_df=source_df,
        hyrcanian_df=hyrcanian_df,
        properties=properties,
    )

    save_outputs(
        hyrcanian_df=hyrcanian_df,
        report=report,
    )

    print("\n" + "=" * 60)
    print("SUCCESS")
    print("=" * 60)

    print(
        "\nOfficial ecoregion:",
        properties.get("eco_name"),
    )

    print(
        "Ecoregion ID:",
        properties.get("eco_id"),
    )

    print(
        "Iran FIRMS records:",
        len(source_df),
    )

    print(
        "Hyrcanian FIRMS records:",
        len(hyrcanian_df),
    )

    print(
        "\nOutput dataset:",
        OUTPUT_CSV.resolve(),
    )

    print(
        "Output report:",
        REPORT_JSON.resolve(),
    )


if __name__ == "__main__":
    main()