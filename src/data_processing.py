"""
Data processing pipeline for the Buenos Aires Real Estate project.

Handles: merging the two raw Properati exports, parsing the nested location
string, extracting coordinates, cleaning outliers, and training a simple
price-estimation model used by the "Price Estimator" tab in the app.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Friendly, human-readable labels for the raw Properati region codes.
REGION_LABELS = {
    "Capital Federal": "Capital Federal",
    "Bs.As. G.B.A. Zona Norte": "Greater BA — North",
    "Bs.As. G.B.A. Zona Oeste": "Greater BA — West",
    "Bs.As. G.B.A. Zona Sur": "Greater BA — South",
}


def load_and_merge_data(file1_path: str, file2_path: str) -> pd.DataFrame:
    """Read both CSV exports and stack them into one DataFrame.

    The two files are simply two batches of the same export (no overlapping
    listings), so this is a straight concat, de-duplicated on the listing
    URL as a safety net in case the same property was scraped twice.
    """
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)
    df = pd.concat([df1, df2], ignore_index=True)
    df = df.drop_duplicates(subset=["properati_url"]).reset_index(drop=True)
    return df


def clean_and_transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """Parse locations/coordinates, coerce numerics, and drop unusable rows."""
    cleaned = df.copy()

    # 1. Split "|Argentina|State|Neighborhood|Sub-neighborhood|" into columns.
    places = cleaned["place_with_parent_names"].str.strip("|").str.split("|", expand=True)
    cleaned["country"] = places[0] if 0 in places.columns else "Argentina"
    cleaned["region"] = places[1] if 1 in places.columns else "Unknown"
    cleaned["neighborhood"] = places[2] if 2 in places.columns else "Unknown"

    cleaned["region"] = cleaned["region"].fillna("Unknown")
    cleaned["neighborhood"] = cleaned["neighborhood"].fillna("Unknown")
    cleaned["region_label"] = cleaned["region"].map(REGION_LABELS).fillna(cleaned["region"])

    # 2. Split "lat,lon" into numeric columns.
    lat_lon = cleaned["lat-lon"].str.split(",", expand=True)
    cleaned["lat"] = pd.to_numeric(lat_lon[0], errors="coerce")
    cleaned["lon"] = pd.to_numeric(lat_lon[1], errors="coerce")

    # 3. Coerce the numeric columns we actually use.
    for col in ["price_aprox_usd", "surface_covered_in_m2", "surface_total_in_m2", "rooms"]:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    # 4. Fill in price-per-m2 where it's missing but we can derive it.
    mask = cleaned["price_usd_per_m2"].isnull() & (cleaned["surface_covered_in_m2"] > 0)
    cleaned.loc[mask, "price_usd_per_m2"] = (
        cleaned.loc[mask, "price_aprox_usd"] / cleaned.loc[mask, "surface_covered_in_m2"]
    )

    # 5. Keep only listings with a sane, usable price (drops placeholder $0
    #    listings and a handful of multi-million dollar data-entry errors).
    valid = (
        cleaned["price_aprox_usd"].notnull()
        & cleaned["price_aprox_usd"].between(5_000, 2_000_000)
    )
    cleaned = cleaned[valid].reset_index(drop=True)

    # 6. Cap surface area at a sane ceiling too (a few listings have
    #    surface areas in the tens of thousands of m2, clearly bad data).
    cleaned = cleaned[
        cleaned["surface_covered_in_m2"].isnull()
        | cleaned["surface_covered_in_m2"].between(10, 1000)
    ].reset_index(drop=True)

    return cleaned


def train_price_model(df: pd.DataFrame):
    """Train a Random Forest to estimate price_aprox_usd from a handful of
    easy-to-fill-in property attributes. Returns (model, feature_names, metrics).
    """
    model_df = df.dropna(
        subset=["price_aprox_usd", "surface_covered_in_m2", "property_type", "region_label"]
    ).copy()
    model_df["rooms"] = model_df["rooms"].fillna(model_df["rooms"].median())

    X = model_df[["property_type", "region_label", "surface_covered_in_m2", "rooms"]].copy()
    y = model_df["price_aprox_usd"]
    X = pd.get_dummies(X, columns=["property_type", "region_label"], drop_first=False)
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "r2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
    }
    return model, feature_names, metrics
