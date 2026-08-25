from __future__ import annotations

from pathlib import Path
import argparse
import sys

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_FIRMS = (
    BASE_DIR
    / "data"
    / "historical"
    / "firms"
    / "fire_archive_M-C61_790637.csv"
)

DEFAULT_ECOREGION_DIR = (
    BASE_DIR
    / "data"
    / "historical"
    / "ecoregions"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "historical"
    / "firms"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "hyrcanian_firms_modis_2001_2026.csv"
)

OUTPUT_SUMMARY = (
    OUTPUT_DIR
    / "hyrcanian_firms_seasonal_summary.csv"
)

OUTPUT_YEARLY = (
    OUTPUT_DIR
    / "hyrcanian_firms_yearly_summary.csv"
)

OUTPUT_GEOJSON = (
    OUTPUT_DIR
    / "hyrcanian_firms_modis_2001_2026.geojson"
)

OUTPUT_ECOREGION = (
    OUTPUT_DIR
    / "hyrcanian_pa0407_boundary.geojson"
)


# ============================================================
# CONSTANTS
# ============================================================

PA0407_CODES = {
    "PA0407",
    "0407",
}

PA0407_NAME = (
    "caspian hyrcanian mixed forests"
)


REQUIRED_FIRMS_COLUMNS = {
    "latitude",
    "longitude",
    "acq_date",
    "confidence",
    "frp",
}


# ============================================================
# HELPERS
# ============================================================

def fail(message: str) -> None:
    print()
    print("=" * 70)
    print("ERROR")
    print("=" * 70)
    print(message)
    print()
    sys.exit(1)


def find_ecoregion_file(directory: Path) -> Path:
    """
    Find a likely terrestrial ecoregion GIS file.
    """

    if not directory.exists():
        fail(
            f"Ecoregion directory does not exist:\n"
            f"{directory}"
        )

    candidates = []

    for pattern in (
        "*.shp",
        "*.gpkg",
        "*.geojson",
        "*.json",
    ):
        candidates.extend(directory.glob(pattern))

    if not candidates:
        fail(
            "No GIS boundary file found in:\n"
            f"{directory}\n\n"
            "Place the RESOLVE/WWF terrestrial "
            "ecoregion shapefile or GeoJSON there."
        )

    # Prefer files whose names suggest ecoregions.
    preferred = [
        p for p in candidates
        if any(
            word in p.name.lower()
            for word in (
                "ecoregion",
                "wwf",
                "resolve",
                "terrestrial",
            )
        )
    ]

    if preferred:
        return preferred[0]

    return candidates[0]


def detect_columns(gdf: gpd.GeoDataFrame):
    """
    Detect common WWF/RESOLVE ecoregion field names.
    """

    normalized = {
        str(c).strip().lower(): c
        for c in gdf.columns
    }

    code_candidates = [
        "eco_code",
        "ecoregion",
        "ecoregion_code",
        "eco_name",
        "code",
        "wwf_mhtnum",
        "wwf_ecoregion",
        "biome",
    ]

    name_candidates = [
        "eco_name",
        "ecoregion_name",
        "name",
        "ecoregion",
        "eco_name",
    ]

    code_col = None
    name_col = None

    for candidate in code_candidates:
        if candidate in normalized:
            code_col = normalized[candidate]
            break

    for candidate in name_candidates:
        if candidate in normalized:
            name_col = normalized[candidate]
            break

    return code_col, name_col


def locate_pa0407(
    gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:

    code_col, name_col = detect_columns(gdf)

    print()
    print("ECOREGION COLUMNS")
    print("-" * 70)
    print("Code column:", code_col)
    print("Name column:", name_col)

    if code_col is not None:

        values = (
            gdf[code_col]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        mask = values.isin(PA0407_CODES)

        result = gdf.loc[mask].copy()

        if not result.empty:
            print(
                f"PA0407 found using column: {code_col}"
            )
            return result

    if name_col is not None:

        values = (
            gdf[name_col]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        mask = values.str.contains(
            "caspian hyrcanian mixed forests",
            regex=False,
            na=False,
        )

        result = gdf.loc[mask].copy()

        if not result.empty:
            print(
                f"PA0407 found using name column: {name_col}"
            )
            return result

    # Last controlled fallback:
    # inspect rows for PA0407 text.
    for column in gdf.columns:

        if gdf[column].dtype == object:

            values = (
                gdf[column]
                .astype(str)
                .str.strip()
                .str.lower()
            )

            mask = (
                values.str.contains(
                    "pa0407",
                    regex=False,
                    na=False,
                )
                |
                values.str.contains(
                    "caspian hyrcanian",
                    regex=False,
                    na=False,
                )
            )

            result = gdf.loc[mask].copy()

            if not result.empty:

                print(
                    f"PA0407 found using column: {column}"
                )

                return result

    fail(
        "PA0407 was not found in the supplied "
        "ecoregion dataset.\n\n"
        "The script will NOT use an approximate "
        "latitude/longitude rectangle."
    )


def prepare_firms(path: Path) -> pd.DataFrame:

    if not path.exists():
        fail(
            f"FIRMS file not found:\n{path}"
        )

    print()
    print("Loading NASA FIRMS...")
    print(path)

    df = pd.read_csv(path)

    missing = (
        REQUIRED_FIRMS_COLUMNS
        - set(df.columns)
    )

    if missing:
        fail(
            "Required FIRMS columns are missing:\n"
            f"{sorted(missing)}"
        )

    df["acq_date"] = pd.to_datetime(
        df["acq_date"],
        errors="coerce",
    )

    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce",
    )

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce",
    )

    df["confidence"] = pd.to_numeric(
        df["confidence"],
        errors="coerce",
    )

    df["frp"] = pd.to_numeric(
        df["frp"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "acq_date",
            "latitude",
            "longitude",
        ]
    ).copy()

    return df


def spatial_filter(
    firms: pd.DataFrame,
    ecoregion: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:

    print()
    print("Creating FIRMS point geometry...")

    points = gpd.GeoDataFrame(
        firms.copy(),
        geometry=[
            Point(lon, lat)
            for lon, lat
            in zip(
                firms["longitude"],
                firms["latitude"],
            )
        ],
        crs="EPSG:4326",
    )

    if ecoregion.crs is None:
        fail(
            "Ecoregion boundary has no CRS."
        )

    ecoregion = ecoregion.to_crs(
        "EPSG:4326"
    )

    # Dissolve multiple PA0407 polygons
    # into one geometry.
    boundary = ecoregion.dissolve()

    print(
        "Applying PA0407 spatial filter..."
    )

    mask = points.geometry.within(
        boundary.geometry.iloc[0]
    )

    result = points.loc[mask].copy()

    return result, boundary


def add_temporal_features(
    gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:

    df = pd.DataFrame(
        gdf.drop(
            columns="geometry"
        )
    )

    dt = pd.to_datetime(
        df["acq_date"],
        errors="coerce",
    )

    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day_of_year"] = dt.dt.dayofyear

    def season(month):

        if month in (3, 4, 5):
            return "spring"

        if month in (6, 7, 8):
            return "summer"

        if month in (9, 10, 11):
            return "autumn"

        return "winter"

    df["season"] = df["month"].apply(
        season
    )

    df["confidence_numeric"] = pd.to_numeric(
        df["confidence"],
        errors="coerce",
    )

    df["frp_numeric"] = pd.to_numeric(
        df["frp"],
        errors="coerce",
    )

    return df


def create_summaries(
    df: pd.DataFrame,
):

    seasonal = (
        df.groupby(
            [
                "year",
                "season",
            ],
            dropna=False,
        )
        .agg(
            fire_detections=(
                "latitude",
                "size",
            ),
            mean_confidence=(
                "confidence_numeric",
                "mean",
            ),
            max_confidence=(
                "confidence_numeric",
                "max",
            ),
            mean_frp=(
                "frp_numeric",
                "mean",
            ),
            max_frp=(
                "frp_numeric",
                "max",
            ),
            total_frp=(
                "frp_numeric",
                "sum",
            ),
        )
        .reset_index()
        .sort_values(
            ["year", "season"]
        )
    )

    yearly = (
        df.groupby(
            "year",
            dropna=False,
        )
        .agg(
            fire_detections=(
                "latitude",
                "size",
            ),
            mean_confidence=(
                "confidence_numeric",
                "mean",
            ),
            max_confidence=(
                "confidence_numeric",
                "max",
            ),
            mean_frp=(
                "frp_numeric",
                "mean",
            ),
            max_frp=(
                "frp_numeric",
                "max",
            ),
            total_frp=(
                "frp_numeric",
                "sum",
            ),
        )
        .reset_index()
        .sort_values("year")
    )

    return seasonal, yearly


def save_outputs(
    df: pd.DataFrame,
    seasonal: pd.DataFrame,
    yearly: pd.DataFrame,
    boundary: gpd.GeoDataFrame,
    geo_df: gpd.GeoDataFrame,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("Saving outputs...")

    df.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    seasonal.to_csv(
        OUTPUT_SUMMARY,
        index=False,
        encoding="utf-8-sig",
    )

    yearly.to_csv(
        OUTPUT_YEARLY,
        index=False,
        encoding="utf-8-sig",
    )

    boundary.to_file(
        OUTPUT_ECOREGION,
        driver="GeoJSON",
    )

    geo_df.to_file(
        OUTPUT_GEOJSON,
        driver="GeoJSON",
    )

    print()
    print("=" * 70)
    print("OUTPUT")
    print("=" * 70)

    print(
        "Hyrcanian FIRMS CSV:"
    )
    print(OUTPUT_CSV)

    print(
        "\nSeasonal summary:"
    )
    print(OUTPUT_SUMMARY)

    print(
        "\nYearly summary:"
    )
    print(OUTPUT_YEARLY)

    print(
        "\nHyrcanian boundary:"
    )
    print(OUTPUT_ECOREGION)

    print(
        "\nHyrcanian FIRMS GeoJSON:"
    )
    print(OUTPUT_GEOJSON)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Build Hyrcanian PA0407 "
            "dataset from NASA FIRMS MODIS."
        )
    )

    parser.add_argument(
        "--firms",
        type=Path,
        default=DEFAULT_FIRMS,
    )

    parser.add_argument(
        "--ecoregion-dir",
        type=Path,
        default=DEFAULT_ECOREGION_DIR,
    )

    args = parser.parse_args()

    print("=" * 70)
    print("FireGuard — Hyrcanian FIRMS Dataset Builder")
    print("=" * 70)

    print(
        "\nFIRMS:",
        args.firms,
    )

    print(
        "Ecoregion directory:",
        args.ecoregion_dir,
    )

    # --------------------------------------------------------
    # FIRMS
    # --------------------------------------------------------

    firms = prepare_firms(
        args.firms
    )

    print(
        f"Loaded FIRMS records: {len(firms):,}"
    )

    # --------------------------------------------------------
    # ECOREGION
    # --------------------------------------------------------

    ecoregion_path = find_ecoregion_file(
        args.ecoregion_dir
    )

    print(
        "\nEcoregion file:",
        ecoregion_path,
    )

    ecoregions = gpd.read_file(
        ecoregion_path
    )

    print(
        "Ecoregion records:",
        len(ecoregions),
    )

    # --------------------------------------------------------
    # PA0407
    # --------------------------------------------------------

    pa0407 = locate_pa0407(
        ecoregions
    )

    print(
        "PA0407 geometry count:",
        len(pa0407),
    )

    # --------------------------------------------------------
    # SPATIAL FILTER
    # --------------------------------------------------------

    filtered, boundary = spatial_filter(
        firms,
        pa0407,
    )

    print()
    print(
        f"Hyrcanian FIRMS records: "
        f"{len(filtered):,}"
    )

    if filtered.empty:
        fail(
            "No FIRMS detections were found "
            "inside PA0407."
        )

    # --------------------------------------------------------
    # TEMPORAL FEATURES
    # --------------------------------------------------------

    result = add_temporal_features(
        filtered
    )

    # --------------------------------------------------------
    # SUMMARIES
    # --------------------------------------------------------

    seasonal, yearly = create_summaries(
        result
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_outputs(
        df=result,
        seasonal=seasonal,
        yearly=yearly,
        boundary=boundary,
        geo_df=filtered,
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("VALIDATION REPORT")
    print("=" * 70)

    print(
        f"Input FIRMS records : {len(firms):,}"
    )

    print(
        f"Hyrcanian records   : {len(result):,}"
    )

    print(
        "Date range          : "
        f"{result['acq_date'].min()} "
        f"→ "
        f"{result['acq_date'].max()}"
    )

    print(
        "Years               : "
        f"{result['year'].min()} "
        f"→ "
        f"{result['year'].max()}"
    )

    print(
        "Unique years        : "
        f"{result['year'].nunique()}"
    )

    print()
    print("Season counts:")
    print(
        result["season"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("Yearly counts:")
    print(
        yearly[
            [
                "year",
                "fire_detections",
                "mean_confidence",
                "mean_frp",
                "max_frp",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "DONE — no approximate geographic "
        "rectangle was used."
    )


if __name__ == "__main__":
    main()