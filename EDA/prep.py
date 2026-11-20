"""Loading and cleaning helpers for the lalafo.kg cars dataset.

Every rule here is backed by evidence in eda_report.md (figures from eda.py), which shows the
before/after for each one. Raw columns are never overwritten: cleaned values go
to new columns (``*_n`` / ``*_fixed``) next to a ``*_rule`` column that says
which rule fired, so every correction stays auditable.
"""
from __future__ import annotations

import collections
import glob
from pathlib import Path

import numpy as np
import pandas as pd

def _find_data() -> Path:
    """dataset/data in this file's folder or the nearest parent (the file may be moved)."""
    for d in Path(__file__).resolve().parents:
        if (d / "dataset" / "data" / "listings.parquet").exists():
            return d / "dataset" / "data"
    raise FileNotFoundError("dataset/data/listings.parquet not found above " + __file__)


DATA = _find_data()

# Personal data - never shown in reports, dropped by load_listings(pii=False).
PII_COLS = ["mobile", "email", "username", "device_id", "user_hash", "avatar"]

# Attributes flattened from lalafo `params` (DATASET_GUIDE §5.2).
CAR_ATTRS = [
    "model", "year", "body_type", "color", "fuel", "transmission",
    "registration_country", "engine_volume_l", "drive", "steering_wheel",
    "condition", "technical_condition", "availability", "mileage_km",
    "payment_terms", "customs_cleared", "vin_status", "battery_capacity",
]
# Params where the seller may tick several options; the API joins them by ", ".
MULTI_SELECT = ["fuel", "transmission", "technical_condition", "payment_terms"]

KG_BBOX = dict(lat=(39.1, 43.3), lng=(69.2, 80.3))

# Manual oblast corrections (checked against coordinates and a map):
#  * homonyms the source's city->oblast lookup matched by name, while their
#    coordinates (median over all their ads) lie in another oblast;
#  * unlabelled towns where the nearest-labelled-town rule picks the wrong
#    oblast across a mountain border, or the name states the oblast outright.
REGION_FIXES = {
    "Кара-Суу": "osh", "Кара-Джыгач": "osh", "Кара-Ой": "jalal_abad",            # homonyms
    "Кочкор": "naryn", "Чаек": "naryn", "Дароот-Коргон": "osh",                  # border towns
    "Кайынды (Чуйская обл.)": "chui",
}
# Every unlabelled town farther than 30 km from a labelled one was checked by
# hand (all 23 are right after the fixes above); the farthest sits at 88 km.
REGION_MAX_KM = 90

PRICE_BAND_KGS = (50_000, 50_000_000)  # ~ $570 .. $570k at the crawl's rate
YEAR_BAND = (1950, 2026)
ENGINE_BAND = (0.6, 8.5)
MILEAGE_MAX = 1_000_000


# --------------------------------------------------------------------------- load
def load_listings(pii: bool = False) -> pd.DataFrame:
    df = pd.read_parquet(DATA / "listings.parquet")
    return df if pii else df.drop(columns=[c for c in PII_COLS if c in df])


def load_users(pii: bool = False) -> pd.DataFrame:
    u = pd.read_parquet(DATA / "users.parquet")
    return u if pii else u.drop(columns=[c for c in PII_COLS if c in u])


def load_cities() -> pd.DataFrame:
    return pd.read_parquet(DATA / "cities.parquet")


def load_image_meta() -> pd.DataFrame:
    """Image rows without the image bytes (column projection: ~1 s for 73 GB)."""
    import pyarrow.dataset as ds

    files = sorted(glob.glob(str(DATA / "images-*.parquet")))
    cols = ["foto_id", "ad_id", "image_id", "p_hash", "is_main"]
    return ds.dataset(files, format="parquet").to_table(columns=cols).to_pandas()


# ------------------------------------------------------------------------ helpers
def first_number(s: pd.Series) -> pd.Series:
    """'412000, 412000' -> 412000; '2,5' is not expected (decimals use '.')."""
    return pd.to_numeric(s.astype("string").str.split(",").str[0].str.strip(), errors="coerce")


def split_multi(s: pd.Series) -> pd.Series:
    """Multi-select string -> frozenset of options (empty set for null)."""
    return s.map(lambda v: frozenset(v.split(", ")) if isinstance(v, str) else frozenset())


def usd_rate(df: pd.DataFrame) -> float:
    usd = df[(df.currency == "USD") & df.national_price.notna() & (df.price > 0)]
    return float((usd.national_price / usd.price).median())


def haversine_km(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    la1, lo1 = np.radians(a[:, 0])[:, None], np.radians(a[:, 1])[:, None]
    la2, lo2 = np.radians(b[:, 0])[None], np.radians(b[:, 1])[None]
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(h))


# ------------------------------------------------------------------------- region
def city_table(df: pd.DataFrame, cities: pd.DataFrame) -> pd.DataFrame:
    """Per city: ad count, median coordinates (inside KG), given and restored oblast."""
    inkg = df.lat.between(*KG_BBOX["lat"]) & df.lng.between(*KG_BBOX["lng"])
    cc = (df[inkg].groupby("city_id").agg(lat=("lat", "median"), lng=("lng", "median"))
          .join(cities.set_index("city_id")[["name", "region", "listing_count"]], how="right"))
    cc["region_fixed"] = cc.name.map(REGION_FIXES).fillna(cc.region)
    cc["region_rule"] = np.where(cc.name.isin(REGION_FIXES), "manual",
                                 np.where(cc.region.notna(), "given", None))
    lab = cc[cc.region_fixed.notna() & cc.lat.notna()]
    todo = cc.region_fixed.isna() & cc.lat.notna()
    if todo.any():
        d = haversine_km(cc.loc[todo, ["lat", "lng"]].values, lab[["lat", "lng"]].values)
        near, dist = lab.region_fixed.values[d.argmin(1)], d.min(1)
        cc.loc[todo, "nearest_km"] = dist.round(1)
        ok = dist <= REGION_MAX_KM
        idx = cc.index[todo]
        cc.loc[idx[ok], "region_fixed"] = near[ok]
        cc.loc[idx[ok], "region_rule"] = "nearest_town"
    cc["region_fixed"] = cc.region_fixed.fillna("unknown")
    cc["region_rule"] = cc.region_rule.fillna("unknown")
    return cc


# -------------------------------------------------------------------------- clean
def clean(df: pd.DataFrame, cities: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add parsed / repaired columns. Does not drop rows."""
    df = df.copy()
    df["is_car"] = df.brand.ne("Другие автомобили")  # leaf 1612 is a junk catch-all

    # year: 1920 is the first option of lalafo's picker - a placeholder, not a year
    y = pd.to_numeric(df.year, errors="coerce")
    df["year_n"] = y.where(y.between(*YEAR_BAND))
    df["age"] = 2026 - df.year_n

    # multi-select -> sets
    for c in MULTI_SELECT:
        df[c + "_set"] = split_multi(df[c])
    is_ev = df.fuel.eq("Электромобиль")

    # engine volume: 0.1-0.5 L are placeholders, EVs have no displacement, >8.5 L are trucks
    e = first_number(df.engine_volume_l)
    rule = np.select([e.isna(), is_ev, e < ENGINE_BAND[0], e > ENGINE_BAND[1]],
                     ["missing", "ev", "too_small", "too_large"], "ok")
    df["engine_l"] = e.where(rule == "ok")
    df["engine_rule"] = rule

    # mileage: < 1000 on a used, pre-2024 car is thousands of km (300 -> 300 000)
    km = first_number(df.mileage_km)
    # (1..49 on such a car is a placeholder - x1000 would still give ~0 km/year)
    used_old = df.condition.ne("Новый") & (df.year_n <= 2023)
    thousands = km.between(50, 999) & used_old
    fixed = km.where(~thousands, km * 1000)
    rule = np.select([km.isna(), fixed > MILEAGE_MAX, thousands, (km < 50) & used_old],
                     ["missing", "too_large", "x1000", "too_small"], "ok")
    df["mileage_n"] = fixed.where(np.isin(rule, ["ok", "x1000"]))
    df["mileage_rule"] = rule

    # price -> comparable KGS, with unit repair (see eda_report.md, price section)
    df = repair_price(df)

    rate = df.attrs["usd_rate"]
    if cities is not None:
        cc = city_table(df, cities)
        df = df.join(cc[["region_fixed", "region_rule"]], on="city_id")
    df.attrs["usd_rate"] = rate
    return df


def repair_price(df: pd.DataFrame) -> pd.DataFrame:
    rate = usd_rate(df)
    kgs = np.where(df.currency.eq("KGS"), df.price,
                   np.where(df.currency.eq("USD"), df.price * rate, np.nan))
    df["price_kgs_raw"] = kgs

    # reference = median price of the same brand/model/year among plausible prices
    plaus = pd.Series(kgs, index=df.index).between(*PRICE_BAND_KGS)
    ref = (df[plaus].assign(p=kgs[plaus]).groupby(["brand", "model", "year_n"]).p
           .agg(["median", "size"]))
    ref = ref[ref["size"] >= 3]["median"].rename("price_ref")
    r = df.join(ref, on=["brand", "model", "year_n"]).price_ref

    p, cur = df.price, df.currency
    fixed = pd.Series(kgs, index=df.index, dtype=float)
    rule = pd.Series(np.where(df.price.isna(), "negotiable", "ok"), index=df.index, dtype=object)

    def pick(mask, options, default):
        """options: {rule_name: multiplier on kgs}; choose the one closest to the reference."""
        nonlocal fixed, rule
        names = list(options)
        cand = np.column_stack([fixed[mask].values * options[k] for k in names])
        with np.errstate(divide="ignore", invalid="ignore"):
            dist = np.abs(np.log10(cand / r[mask].values[:, None]))
        dist = np.where(np.isfinite(dist), dist, np.inf)
        best = np.array(names, dtype=object)[dist.argmin(1)]
        best = np.where(dist.min(1) < 0.5, best, default)  # no/far reference -> default rule
        best = pd.Series(best, index=fixed.index[mask])
        for k, m in options.items():
            sel = best.index[best == k]
            fixed.loc[sel] = fixed.loc[sel] * m
            rule.loc[sel] = k

    pick(cur.eq("KGS") & (p < 1_000), {"kgs_thousands": 1_000, "usd_as_kgs": rate}, "kgs_thousands")
    pick(cur.eq("KGS") & p.between(1_000, 50_000, inclusive="left"),
         {"usd_as_kgs": rate, "kgs_thousands": 1_000}, "usd_as_kgs")
    pick(cur.eq("USD") & (p < 100), {"usd_thousands": 1_000}, "usd_thousands")

    out = ~fixed.between(*PRICE_BAND_KGS) & fixed.notna()
    rule[out] = "out_of_range"
    df["price_kgs"] = fixed.where(~out)
    df["price_rule"] = rule
    df["price_ref"] = r
    df.attrs["usd_rate"] = rate
    return df


# --------------------------------------------------------------------- duplicates
class _DSU:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def duplicate_pairs(df: pd.DataFrame, im: pd.DataFrame) -> pd.DataFrame:
    """Candidate duplicate ad pairs, one row per (pair, rule).

    A - share >= 3 identical photos (p_hash), ignoring hashes seen in >= 20 ads (stock art)
    B - cover photos within Hamming distance 3 and same brand/model/year
    C - same brand, model, year, price, currency and seller
    """
    out = []
    ph = im.dropna(subset=["p_hash"])[["ad_id", "p_hash"]].drop_duplicates()
    n_ads = ph.groupby("p_hash").ad_id.nunique()
    shared = ph[ph.p_hash.map(n_ads).between(2, 19)]
    pr = shared.merge(shared, on="p_hash")
    pr = pr[pr.ad_id_x < pr.ad_id_y].groupby(["ad_id_x", "ad_id_y"]).size()
    a = pr[pr >= 3].reset_index()[["ad_id_x", "ad_id_y"]]
    out.append(a.assign(rule="A_3plus_photos"))

    main = im[im.is_main].drop_duplicates("ad_id").dropna(subset=["p_hash"])
    h = dict(zip(main.ad_id, main.p_hash.astype("uint64")))
    hv = main.p_hash.astype("uint64").values
    cand = set()
    for b in range(4):  # pigeonhole: Hamming <= 3 -> one 16-bit band is identical
        band = (hv >> np.uint64(16 * b)) & np.uint64(0xFFFF)
        for ids in pd.Series(main.ad_id.values).groupby(band).agg(list):
            if 1 < len(ids) < 200:
                ids = sorted(ids)
                cand.update((ids[i], ids[j]) for i in range(len(ids)) for j in range(i + 1, len(ids)))
    key = dict(zip(df.ad_id, df.brand.fillna("") + "|" + df.model.fillna("") + "|" + df.year.fillna("")))
    b = [(x, y) for x, y in cand
         if bin(int(h[x]) ^ int(h[y])).count("1") <= 3 and key.get(x) == key.get(y)]
    out.append(pd.DataFrame(b, columns=["ad_id_x", "ad_id_y"]).assign(rule="B_cover_photo"))

    k = ["brand", "model", "year", "price", "currency", "user_id"]
    g = df.dropna(subset=["price"]).groupby(k).ad_id.agg(list)
    c = [(min(v[0], x), max(v[0], x)) for v in g if len(v) > 1 for x in v[1:]]
    out.append(pd.DataFrame(c, columns=["ad_id_x", "ad_id_y"]).assign(rule="C_same_key"))
    return pd.concat(out, ignore_index=True)


def mark_duplicates(df: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """dup_cluster (ad_id of the kept ad) and is_dup (True for every ad but the kept one).
    The kept ad is the most recently updated one in its cluster."""
    dsu = _DSU()
    for x, y in zip(pairs.ad_id_x, pairs.ad_id_y):
        dsu.union(x, y)
    df = df.copy()
    root = df.ad_id.map(lambda a: dsu.find(a) if a in dsu.p else a)
    keep = df.assign(root=root).sort_values("updated_time").groupby("root").ad_id.last()
    df["dup_cluster"] = root.map(keep)
    df["dup_size"] = df.groupby("dup_cluster").ad_id.transform("size")
    df["is_dup"] = df.ad_id.ne(df.dup_cluster)
    return df


def seller_type(df: pd.DataFrame, users: pd.DataFrame) -> pd.Series:
    """'dealer' = pro account or >= 3 ads in the dataset, else 'private'."""
    u = users.set_index("user_id")
    pro = df.user_id.map(u.pro).fillna(False).astype(bool)
    n = df.user_id.map(u.ads_count).fillna(1)
    return pd.Series(np.where(pro | (n >= 3), "dealer", "private"), index=df.index)


# ------------------------------------------------------------------------- stats
def norm_entropy(s: pd.Series) -> float:
    p = s.value_counts(normalize=True)
    return float(-(p * np.log(p)).sum() / np.log(len(p))) if len(p) > 1 else 0.0


def eta_squared(cat: pd.Series, y: pd.Series, min_n: int = 30) -> float:
    """Share of variance of y explained by the groups of cat (correlation ratio^2).
    Groups smaller than min_n are pooled into one 'rare' group."""
    d = pd.DataFrame({"c": cat.astype("string"), "y": y}).dropna()
    vc = d.c.value_counts()
    d["c"] = d.c.where(d.c.map(vc) >= min_n, "__rare__")
    gm = d.groupby("c").y.agg(["mean", "size"])
    ss_b = (gm["size"] * (gm["mean"] - d.y.mean()) ** 2).sum()
    ss_t = ((d.y - d.y.mean()) ** 2).sum()
    return float(ss_b / ss_t) if ss_t else np.nan


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    from scipy.stats import chi2_contingency

    t = pd.crosstab(a, b)
    if t.shape[0] < 2 or t.shape[1] < 2:
        return np.nan
    chi2 = chi2_contingency(t, correction=False)[0]
    n = t.values.sum()
    return float(np.sqrt(chi2 / (n * (min(t.shape) - 1))))


# ------------------------------------------------------------------------- charts
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
NEUTRAL = "#c3c2b7"  # for "missing / unknown / other"


def set_style():
    import matplotlib as mpl
    from cycler import cycler

    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "figure.figsize": (9, 4),
        "font.family": "sans-serif", "font.size": 10,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "legend.frameon": False, "legend.labelcolor": INK2,
        "lines.linewidth": 2, "patch.linewidth": 0,
        "axes.prop_cycle": cycler(color=SERIES),
    })
