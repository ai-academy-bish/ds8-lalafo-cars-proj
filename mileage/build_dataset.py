"""Modelling table for the mileage pilot: unique used cars of model years 2010-2024.

Run:  venv/bin/python mileage/build_dataset.py [--listings-only]
Writes mileage/data/mileage_2010_2024.parquet (features, target, fixed train/test split)
and mileage/data/build_summary.json (row funnel, unlabelled population).
With --listings-only (only dataset/data/listings.parquet present) it writes
mileage_2010_2024_listings.parquet / build_summary_listings.json instead - see build().

The split is stored with the data on purpose: every later experiment must be
measured on the same held-out test set, or the numbers stop being comparable.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "EDA"))
import prep  # noqa: E402

RANDOM_STATE = 42
SNAPSHOT_YEAR = 2026          # all ads were scraped on 2026-09-09
YEAR_MIN, YEAR_MAX = 2010, 2024
KM_MIN = 1_000                # below this a used 2010-2024 car is a data-entry error
KM_PER_YEAR = (1_000, 60_000) # outside this the declared mileage is implausible for the age
TEST_SIZE = 0.2

OUT = HERE / "data"
FUEL = {"Бензин": "fuel_petrol", "Дизель": "fuel_diesel", "Газ": "fuel_lpg",
        "Гибрид": "fuel_hybrid", "Электромобиль": "fuel_ev"}
TECH_WORST_FIRST = ["На запчасти", "Битый", "Аварийное", "Хорошее", "Идеальное"]


def single_or_multi(s: pd.Series) -> pd.Series:
    """Multi-select set -> the option if exactly one was ticked, 'multi' if several, NaN if none."""
    return s.map(lambda x: next(iter(x)) if len(x) == 1 else ("multi" if x else np.nan))


def worst_option(s: pd.Series, order: list[str]) -> pd.Series:
    """Multi-select set -> the most severe ticked option (contradictory picks resolve to the worst)."""
    return s.map(lambda x: next((o for o in order if o in x), np.nan))


def as_category(s: pd.Series) -> pd.Series:
    """Plain object column with np.nan for missing (what SimpleImputer expects)."""
    return s.astype(object).where(s.notna(), np.nan)


def out_paths(listings_only: bool) -> tuple[Path, Path]:
    """Separate files per mode, so a listings-only build never overwrites the pilot table."""
    sfx = "_listings" if listings_only else ""
    return OUT / f"mileage_2010_2024{sfx}.parquet", OUT / f"build_summary{sfx}.json"


def build(listings_only: bool = False) -> tuple[pd.DataFrame, dict]:
    """listings_only=True needs only dataset/data/listings.parquet:
    cities come from the ads themselves, duplicates are found by rule C only (no photo hashes),
    and 'dealer' means >= 3 ads in listings (no `pro` flag). Rows and split differ from the pilot."""
    raw = prep.load_listings()
    if listings_only:
        cities, users, im = prep.cities_from_listings(raw), None, None
    else:
        cities, users, im = prep.load_cities(), prep.load_users(), prep.load_image_meta()
    df = prep.clean(raw, cities)
    df = prep.mark_duplicates(df, prep.duplicate_pairs(df, im))
    df["seller"] = prep.seller_type(df, users)

    km_per_year = df.mileage_n / df.age.clip(lower=1)
    steps = [
        ("все объявления", pd.Series(True, index=df.index)),
        ("без «Других автомобилей»", df.is_car),
        ("без повторов одной машины", ~df.is_dup),
        (f"год выпуска {YEAR_MIN}-{YEAR_MAX}", df.year_n.between(YEAR_MIN, YEAR_MAX)),
        ("пробег указан и прошёл очистку prep.py", df.mileage_n.notna()),
        (f"пробег >= {KM_MIN:,} км".replace(",", " "), df.mileage_n >= KM_MIN),
        ("годовой пробег 1-60 тыс. км", km_per_year.between(*KM_PER_YEAR)),
    ]
    keep = pd.Series(True, index=df.index)
    funnel = []
    for name, cond in steps:
        keep &= cond
        funnel.append({"step": name, "rows": int(keep.sum())})
    d = df[keep]

    # the population the model would be applied to: same cars, mileage not stated
    scope = df.is_car & ~df.is_dup & df.year_n.between(YEAR_MIN, YEAR_MAX)
    unl = df[scope & df.mileage_rule.eq("missing")]
    lab = d

    def profile(x):
        return {"rows": int(len(x)), "median_year": float(x.year_n.median()),
                "dealer_share": round(float(x.seller.eq("dealer").mean()), 3),
                "median_price_usd": round(float(x.price_kgs.median() / df.attrs["usd_rate"])),
                "kgs_currency_share": round(float(x.currency.eq("KGS").mean()), 3),
                "vip_share": round(float(x.is_vip.mean()), 3),
                "chui_share": round(float(x.region_fixed.eq("chui").mean()), 3)}

    out = pd.DataFrame({
        "ad_id": d.ad_id, "user_id": d.user_id,                     # ids: never features
        "mileage_km": d.mileage_n,                                  # target
        "year": d.year_n.astype(int),
        "age": (SNAPSHOT_YEAR - d.year_n).astype(float),
        "engine_l": d.engine_l,
        "price_kgs": d.price_kgs,
        "image_count": d.image_count.astype(float),
        "brand": as_category(d.brand),
        "brand_model": as_category(d.brand + " | " + d.model.fillna("NA")),
        "body_type": as_category(d.body_type),
        "transmission": as_category(single_or_multi(d.transmission_set)),
        "drive": as_category(d.drive),
        "steering_wheel": as_category(d.steering_wheel),
        "condition": as_category(d.condition),
        "tech_condition": as_category(worst_option(d.technical_condition_set, TECH_WORST_FIRST)),
        "customs_cleared": as_category(d.customs_cleared),
        "registration_country": as_category(d.registration_country),
        "region": as_category(d.region_fixed),
        "seller": as_category(d.seller),
        "currency": as_category(d.currency),
        "is_vip": as_category(d.is_vip.map({True: "yes", False: "no"})),
        "mileage_rule": d.mileage_rule,                             # audit only: derived from y
    })
    for ru, col in FUEL.items():
        out[col] = d.fuel_set.map(lambda s, ru=ru: float(ru in s))
    out["fuel_isna"] = d.fuel_set.map(lambda s: float(len(s) == 0))
    # deterministic transforms (no fitting -> no leakage when done before the split)
    out["log_age"] = np.log(out.age)
    out["log_price"] = np.log(out.price_kgs)

    age_band = pd.cut(out.age, [1, 4, 7, 11, 16], labels=["2022-24", "2019-21", "2015-18", "2010-14"])
    tr, te = train_test_split(out.index, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=age_band)
    out["split"] = "train"
    out.loc[te, "split"] = "test"
    out = out.reset_index(drop=True)

    summary = {
        "listings_only": listings_only,
        "snapshot_year": SNAPSHOT_YEAR, "years": [YEAR_MIN, YEAR_MAX], "km_min": KM_MIN,
        "km_per_year": list(KM_PER_YEAR), "test_size": TEST_SIZE, "random_state": RANDOM_STATE,
        "funnel": funnel, "split_counts": out.split.value_counts().to_dict(),
        "labelled_profile": profile(lab), "unlabelled_profile": profile(unl),
        "excluded_by_plausibility": int((scope & df.mileage_n.notna()).sum() - len(d)),
    }
    return out, summary


if __name__ == "__main__":
    listings_only = "--listings-only" in sys.argv[1:]
    OUT.mkdir(exist_ok=True)
    data, summary = build(listings_only)
    data_path, summary_path = out_paths(listings_only)
    data.to_parquet(data_path, index=False)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(data.drop(columns=["ad_id", "user_id"]).describe(include="all").T.to_string())
