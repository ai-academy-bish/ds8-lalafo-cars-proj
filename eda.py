"""Exploratory analysis of the lalafo.kg cars dataset.

Writes every figure to ./fig and prints the numbers quoted in eda_report.md.
Run:  venv/bin/python eda.py
"""
from __future__ import annotations

import collections
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

import prep
from prep import AXIS, BLUE_RAMP, INK, INK2, MUTED, NEUTRAL, SERIES

FIG = Path(__file__).resolve().parent / "fig"
FIG.mkdir(exist_ok=True)
prep.set_style()
pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 200)

BLUE, ORANGE, AQUA = SERIES[0], SERIES[1], SERIES[2]
SEQ = LinearSegmentedColormap.from_list("seq", ["#fcfcfb"] + BLUE_RAMP)
DIV = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
REGION_RU = {"chui": "Чуйская", "osh": "Ошская", "issyk_kul": "Иссык-Кульская",
             "jalal_abad": "Джалал-Абадская", "naryn": "Нарынская", "batken": "Баткенская",
             "talas": "Таласская", "unknown": "не указана"}
STATS: dict = {}


def save(name: str):
    plt.savefig(FIG / f"{name}.png", dpi=130, bbox_inches="tight")
    plt.close("all")


def section(title: str):
    print(f"\n{'=' * 20} {title} {'=' * 20}")


def pct(x) -> str:
    return f"{100 * x:.1f}%"


def sp(v) -> str:
    """12345 -> '12 345' (thin-space thousands, as in Russian typography)."""
    return f"{v:,.0f}".replace(",", " ")


SP_FMT = mt.FuncFormatter(lambda v, _: sp(v))


def hbar(ax, s: pd.Series, color=BLUE, fmt=sp, highlight: dict | None = None, total=None):
    """Horizontal bars, largest on top, value (and share) labelled at the bar end."""
    s = s[::-1]
    colors = [(highlight or {}).get(k, color) for k in s.index]
    ax.barh(range(len(s)), s.values, color=colors, height=0.68)
    ax.set_yticks(range(len(s)), [str(i) for i in s.index])
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    xmax = s.max()
    for i, v in enumerate(s.values):
        lab = fmt(v) + (f"  ({100 * v / total:.0f}%)" if total else "")
        ax.text(v + xmax * 0.01, i, lab, va="center", fontsize=8.5, color=INK2)
    ax.set_xlim(0, xmax * 1.25)
    ax.xaxis.set_major_locator(mt.MaxNLocator(4))
    ax.xaxis.set_major_formatter(SP_FMT)


def usd_axis(axis):
    """Log axis in dollars with readable major and minor labels."""
    ticks = [500, 1000, 2000, 3000, 5000, 10000, 20000, 30000, 50000, 100000]
    axis.set_major_locator(mt.FixedLocator(ticks))
    axis.set_minor_locator(mt.NullLocator())
    axis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"${v / 1000:g}k"))


# =========================================================================== load
section("LOAD")
raw = prep.load_listings()
users = prep.load_users()
cities = prep.load_cities()
im = prep.load_image_meta()
df = prep.clean(raw, cities)
pairs = prep.duplicate_pairs(df, im)
df = prep.mark_duplicates(df, pairs)
df["seller"] = prep.seller_type(df, users)
df["log_price"] = np.log10(df.price_kgs)
RATE = df.attrs["usd_rate"]
cars = df[df.is_car]                       # without the «Другие автомобили» catch-all
uniq = cars[~cars.is_dup]                  # one row per physical car (best effort)
print(f"listings={len(df)} cars={len(cars)} unique cars={len(uniq)} users={len(users)} "
      f"cities={len(cities)} images={len(im)} usd_rate={RATE}")
created = pd.to_datetime(df.created_date).dt.tz_localize(None)
pars = pd.to_datetime(df.pars_date).dt.tz_localize(None)
print("created:", created.min(), "->", created.max(), "| scraped:", pars.min(), "->", pars.max())
age_days = (pars - created).dt.days
print("ad age (days) quantiles:", age_days.quantile([.25, .5, .75, .9]).round(0).to_dict())
print("created in last 90 days:", pct((age_days <= 90).mean()), " in Jul-Sep 2026:",
      pct((created >= "2026-07-01").mean()))

fig, ax = plt.subplots(figsize=(9, 3.2))
wk = created[created >= "2025-09-01"].dt.to_period("W").dt.start_time.value_counts().sort_index()
ax.bar(wk.index, wk.values, width=5, color=BLUE)
ax.set_title("Объявления по неделе публикации (с сентября 2025)")
ax.set_ylabel("объявлений")
older = (created < "2025-09-01").sum()
ax.text(0.01, 0.92, f"ещё {older:,} объявлений старше сентября 2025 (с 2016 г.)".replace(",", " "),
        transform=ax.transAxes, color=INK2, fontsize=9)
save("01_created_by_week")

# ====================================================================== geography
section("GEOGRAPHY")
cc = prep.city_table(df, cities)
given = df.region.fillna("unknown").value_counts()
fixed = df.region_fixed.value_counts()
geo = pd.DataFrame({"given": given, "fixed": fixed}).fillna(0).astype(int).sort_values("fixed", ascending=False)
geo["fixed_share"] = (geo.fixed / len(df)).round(3)
print(geo.to_string())
print("region rule:", df.region_rule.value_counts().to_dict())
print("cities total:", len(cc), " with given region:", cc.region.notna().sum(),
      " manual fixes:", (cc.region_rule == "manual").sum())
wrong = cc[cc.name.isin(["Кара-Суу", "Кара-Джыгач", "Кара-Ой"])][["name", "lat", "lng", "region", "region_fixed", "listing_count"]]
print("homonym fixes:\n", wrong.to_string())
top_city = df.city.value_counts()
print("top cities:\n", (top_city.head(15) / len(df)).round(3).to_string())
print("Bishkek share:", pct(top_city["Бишкек"] / len(df)), " cities with <30 ads:", (top_city < 30).sum())
inkg = df.lat.between(*prep.KG_BBOX["lat"]) & df.lng.between(*prep.KG_BBOX["lng"])
med = df.groupby("city_id")[["lat", "lng"]].transform("median")
print("coords outside KG:", (~inkg).sum(), " ads exactly at city centroid:",
      pct(((df.lat == med.lat) & (df.lng == med.lng)).mean()))

fig, ax = plt.subplots(figsize=(9, 3.6))
order = geo.index.tolist()
y = np.arange(len(order))
ax.barh(y + 0.19, geo.given.values, height=0.36, color=NEUTRAL, label="как в данных (region)")
ax.barh(y - 0.19, geo.fixed.values, height=0.36, color=BLUE, label="после исправления и восстановления")
ax.set_yticks(y, [REGION_RU[r] for r in order])
ax.invert_yaxis()
ax.set_xscale("log")
ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", " ")))
ax.grid(axis="y", visible=False)
for i, r in enumerate(order):
    ax.text(geo.fixed[r] * 1.08, i - 0.19, f"{geo.fixed[r]:,}  ({100 * geo.fixed[r] / len(df):.1f}%)".replace(",", " "),
            va="center", fontsize=8.5, color=INK2)
ax.set_xlim(right=geo.max().max() * 4)
ax.set_title("Объявления по областям (логарифмическая шкала)")
ax.legend(loc="center right", fontsize=8.5)
save("02_regions")

fig, ax = plt.subplots(figsize=(9, 5))
c2 = cc.dropna(subset=["lat"])
ax.scatter(c2.lng, c2.lat, s=np.sqrt(c2.listing_count) * 4, color=BLUE, alpha=0.55,
           edgecolor="#fcfcfb", linewidth=0.8)
LABELS = {"Бишкек": (-20, 26), "Токмак": (14, 14), "Каракол": (-10, 12), "Ош": (10, -4), "Балыкчи": (6, 8),
          "Джалал-Абад": (8, -2), "Талас": (-8, 10), "Нарын": (8, 4), "Кызыл-Кия": (-16, -14), "Баткен": (-20, 9)}
for _, r in c2[c2.name.isin(LABELS)].iterrows():
    ax.annotate(f"{r['name']} {sp(r.listing_count)}", (r.lng, r.lat), xytext=LABELS[r["name"]],
                textcoords="offset points", fontsize=8, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6) if r["name"] in ("Бишкек", "Токмак") else None)
for reg, g in c2.groupby("region_fixed"):
    w = g.listing_count
    dy = -0.42 if reg == "chui" else -0.35
    ax.text(np.average(g.lng, weights=w), np.average(g.lat, weights=w) + dy, REGION_RU[reg] + (" обл." if reg != "chui" else " обл. (80%)"),
            fontsize=8.5, color=MUTED, ha="center", style="italic")
ax.set_xlim(69.2, 80.3); ax.set_ylim(39.2, 43.4)
ax.set_aspect(1 / np.cos(np.radians(41.3)))
ax.set_title("Города: площадь круга ~ число объявлений")
ax.set_xlabel("долгота"); ax.set_ylabel("широта")
save("03_city_map")

# ================================================================ feature passport
section("FEATURES")
usable = {  # share of ads where the attribute is present AND plausible after cleaning
    "year": df.year_n.notna().mean(), "engine_volume_l": df.engine_l.notna().mean(),
    "mileage_km": df.mileage_n.notna().mean(),
}
rows = []
for c in prep.CAR_ATTRS + ["price"]:
    s = df[c]
    vc = s.value_counts()
    rows.append(dict(feature=c, fill=s.notna().mean(), usable=usable.get(c, df.price_kgs.notna().mean() if c == "price" else s.notna().mean()),
                     nunique=s.nunique(), top=str(vc.index[0])[:28] if len(vc) else "", top_share=(vc.iloc[0] / vc.sum()) if len(vc) else np.nan))
passport = pd.DataFrame(rows).sort_values("usable", ascending=False)
print(passport.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

fig, ax = plt.subplots(figsize=(9, 6))
p = passport.set_index("feature")
y = np.arange(len(p))
ax.barh(y, p.fill, color=NEUTRAL, height=0.7, label="заполнено")
ax.barh(y, p.usable, color=BLUE, height=0.7, label="заполнено и правдоподобно после очистки")
for i, (f, r) in enumerate(p.iterrows()):
    ax.text(r.fill + 0.01, i, f"{100 * r.usable:.0f}%" + (f"  (сырое {100 * r.fill:.0f}%)" if r.fill - r.usable > 0.005 else ""),
            va="center", fontsize=8.5, color=INK2)
ax.set_yticks(y, p.index); ax.invert_yaxis()
ax.axvline(0.8, color=MUTED, lw=1, ls=(0, (3, 3)))
ax.text(0.8, -0.9, "80%", color=MUTED, fontsize=8, ha="center")
ax.set_xlim(0, 1.18); ax.xaxis.set_major_formatter(mt.PercentFormatter(1))
ax.grid(axis="y", visible=False)
ax.set_title("Заполненность признаков автомобиля (все 54 518 объявлений)")
ax.legend(loc="lower right", fontsize=8.5)
save("04_fill_rates")

# -------------------------------------------------------------------- year
section("YEAR")
print(df.year_n.describe(percentiles=[.05, .25, .5, .75, .95]).round(1).to_dict())
print("year==1920 placeholder:", (pd.to_numeric(df.year, errors="coerce") == 1920).sum(),
      " <1950:", (pd.to_numeric(df.year, errors="coerce") < 1950).sum())
for lo, hi in [(1990, 2009), (2016, 2022)]:
    print(f"share {lo}-{hi}:", pct(df.year_n.between(lo, hi).mean()))
fig, ax = plt.subplots(figsize=(9, 3.4))
yc = uniq.year_n.value_counts().sort_index()
yc = yc[yc.index >= 1980]
ax.bar(yc.index, yc.values, width=0.8, color=BLUE)
ax.set_title("Год выпуска (уникальные машины, 1980+)")
ax.set_ylabel("объявлений")
ax.annotate("пик 2003: б/у «европейцы»\nи «японцы» 20+ лет", (2003, yc[2003]), xytext=(1984, yc.max() * 0.8),
            fontsize=8.5, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax.annotate("пик 2019: корейский импорт\n(Kia/Hyundai) и электромобили", (2019, yc[2019]),
            xytext=(2007.5, yc.max() * 0.95), fontsize=8.5, color=INK2,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
save("05_year")
yb = pd.cut(uniq.year_n, [1949, 2004, 2012, 2018, 2026], labels=["≤2004", "2005-12", "2013-18", "2019+"])
print("brand mix by year band (top 5 per band):")
for b, g in uniq.groupby(yb, observed=True):
    print(" ", b, (g.brand.value_counts(normalize=True).head(5) * 100).round(0).to_dict())
print("fuel EV/hybrid share by year band:",
      uniq.groupby(yb, observed=True).fuel_set.apply(lambda s: s.map(lambda x: bool(x & {"Электромобиль", "Гибрид"})).mean()).round(3).to_dict())

# -------------------------------------------------------------------- mileage
section("MILEAGE")
km = prep.first_number(df.mileage_km)
print("raw present:", pct(km.notna().mean()), " rules:", df.mileage_rule.value_counts().to_dict())
print("raw <1000:", (km < 1000).sum(), " >1M:", (km > 1e6).sum(), " top raw values <1000:",
      km[km < 1000].value_counts().head(8).to_dict())
rep = df.mileage_n / df.age.clip(lower=1)
print("km/year, raw-ok:", rep[df.mileage_rule == "ok"].quantile([.25, .5, .75]).round(0).to_dict(),
      " repaired x1000:", rep[df.mileage_rule == "x1000"].quantile([.25, .5, .75]).round(0).to_dict())
print("clean mileage:", df.mileage_n.describe(percentiles=[.05, .25, .5, .75, .95]).round(0).to_dict())
print("mileage<1000 on 'Новый':", ((km < 1000) & (df.condition == "Новый")).sum())

fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
bins = np.arange(0, 9.01, 0.1)
lk = np.log10(km.clip(lower=1)).dropna()
axs[0].hist(lk, bins=bins, color=BLUE)
axs[0].axvspan(np.log10(50), np.log10(999), color=ORANGE, alpha=0.12, lw=0)
axs[0].text(2.35, axs[0].get_ylim()[1] * 0.75, "50-999:\nпробег\nв тыс. км", ha="center", color=INK2, fontsize=8.5)
axs[0].axvspan(6, 9, color=NEUTRAL, alpha=0.25, lw=0)
axs[0].text(7.5, axs[0].get_ylim()[1] * 0.75, "> 1 млн км:\nмусор", ha="center", color=INK2, fontsize=8.5)
axs[0].set_xticks(range(0, 10), ["1", "10", "100", "1k", "10k", "100k", "1M", "10M", "100M", "1B"])
axs[0].set_title("Пробег как ввели (лог. шкала, км)")
b2 = np.arange(0, 60001, 2000)
axs[1].hist(rep[df.mileage_rule == "ok"], bins=b2, color=BLUE, density=True,
            histtype="step", lw=2, label="нормальные записи")
axs[1].hist(rep[df.mileage_rule == "x1000"], bins=b2, color=ORANGE, density=True,
            histtype="step", lw=2, label="исправленные ×1000 (сырое 50-999)")
axs[1].set_title("Пробег в год после исправления")
axs[1].set_xlabel("км в год"); axs[1].set_yticks([])
axs[1].legend(fontsize=8, loc="upper right")
axs[1].xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
save("06_mileage")

# -------------------------------------------------------------------- engine
section("ENGINE")
e = prep.first_number(df.engine_volume_l)
print("rules:", df.engine_rule.value_counts().to_dict())
print("raw <0.5 by fuel:", df[e < 0.5].fuel.fillna("NA").value_counts().head(5).to_dict())
print("clean:", df.engine_l.describe(percentiles=[.05, .25, .5, .75, .95]).round(2).to_dict())
print("top volumes:", (df.engine_l.value_counts(normalize=True).head(8) * 100).round(1).to_dict())
fig, ax = plt.subplots(figsize=(9, 3.4))
vc = e[e <= 6.5].round(1).value_counts().sort_index()
col = [ORANGE if v < 0.6 else BLUE for v in vc.index]
ax.bar(vc.index, vc.values, width=0.075, color=col)
ax.set_title("Объём двигателя как ввели, л (оранжевое — значения-заглушки < 0.6 л)")
ax.set_ylabel("объявлений")
for v in [1.5, 1.6, 2.0, 2.4]:
    ax.text(v, vc.get(v, 0) + 80, f"{v}", ha="center", fontsize=8, color=INK2)
save("07_engine")

# -------------------------------------------------------------------- price
section("PRICE")
print("currency:", df.currency.fillna("NA").value_counts().to_dict())
print("price null == is_negotiable:", (df.price.isna() == df.is_negotiable).all(), " share:", pct(df.is_negotiable.mean()))
print("national_price filled: USD", df[df.currency == "USD"].national_price.notna().mean().round(3),
      " KGS", df[df.currency == "KGS"].national_price.notna().mean().round(3))
print("price rules:", df.price_rule.value_counts().to_dict())
lr_raw = np.log10(df.price_kgs_raw / df.price_ref)
lr_fix = np.log10(df.price_kgs / df.price_ref)
print("|log10(price/ref)|>0.5 raw:", (lr_raw.abs() > 0.5).sum(), " after fix:", (lr_fix.abs() > 0.5).sum(),
      " (of", lr_raw.notna().sum(), "with reference)")
print("repair accuracy by rule (median log10 ratio):",
      df.assign(l=lr_fix).groupby("price_rule").l.median().round(3).dropna().to_dict())
print("clean price KGS:", df.price_kgs.describe(percentiles=[.05, .25, .5, .75, .95]).round(-3).to_dict())
print("clean price USD median:", round(df.price_kgs.median() / RATE), " IQR:",
      (df.price_kgs.quantile([.25, .75]) / RATE).round(-2).to_dict())
print("KGS vs USD listings median (clean, USD):",
      (df.groupby("currency").price_kgs.median() / RATE).round(-2).to_dict())

fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
bins = np.arange(0, 10.01, 0.1)
for cur, c in (("KGS", BLUE), ("USD", ORANGE)):
    v = np.log10(df.loc[df.currency == cur, "price"].clip(lower=1))
    axs[0].hist(v, bins=bins, color=c, histtype="step", lw=2, label=cur)
axs[0].set_xticks(range(0, 11, 2), ["1", "100", "10k", "1M", "100M", "10B"])
axs[0].set_title("Цена как ввели, в своей валюте (лог.)")
axs[0].legend(fontsize=8.5)
axs[0].annotate("KGS 100-999:\nцена в тыс. сом", (2.3, 330), xytext=(0.2, 1900), fontsize=8, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
axs[0].annotate("KGS 1k-50k:\nэто доллары", (4.15, 320), xytext=(6.6, 1900), fontsize=8, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
b = np.arange(-4.5, 2.51, 0.1)
axs[1].hist(lr_raw.clip(-4.5, 2.5), bins=b, color=NEUTRAL, label="до исправления")
axs[1].hist(lr_fix.clip(-4.5, 2.5), bins=b, color=BLUE, histtype="step", lw=2, label="после")
axs[1].set_yscale("log")
axs[1].set_title("Цена / медиана той же марки-модели-года")
axs[1].set_xticks([-3, np.log10(1 / RATE), 0, 2], ["1/1000", f"1/{RATE:.0f}", "1", "100"])
axs[1].legend(fontsize=8.5)
save("08_price_repair")

fig, ax = plt.subplots(figsize=(9, 3.4))
usd = df.price_kgs / RATE
ax.hist(np.log10(usd.dropna()), bins=np.arange(2.7, 5.8, 0.1), color=BLUE)
ticks = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000]
ax.set_xticks(np.log10(ticks), [f"${t // 1000}k" for t in ticks])
ax.set_ylim(top=ax.get_ylim()[1] * 1.18)
for q in (0.25, 0.5, 0.75):
    v = usd.quantile(q)
    ax.axvline(np.log10(v), color=INK2, lw=1, ls=(0, (3, 3)))
    ax.text(np.log10(v), ax.get_ylim()[1] * 0.98, f"{'Q1' if q == .25 else 'медиана' if q == .5 else 'Q3'} ${v / 1000:.1f}k",
            fontsize=8, color=INK2, va="top", ha="center", bbox=dict(fc=prep.SURFACE, ec="none", pad=1.5))
ax.set_title(f"Цена после очистки, в USD (курс {RATE} сом/$, лог. шкала)")
ax.set_ylabel("объявлений")
save("09_price_clean")

# ==================================================================== categoricals
section("CATEGORICALS")
CAT = ["body_type", "fuel", "transmission", "drive", "steering_wheel", "color", "condition",
       "technical_condition", "customs_cleared", "availability", "registration_country", "vin_status"]
balance = []
for c in CAT + ["brand", "region_fixed"]:
    if c in prep.MULTI_SELECT:
        ex = df[c + "_set"].explode().dropna()
        vc = ex.value_counts()
        combos = df[c].str.contains(", ", na=False).mean()
    else:
        vc = df[c].value_counts()
        combos = np.nan
    p_ = vc / vc.sum()
    balance.append(dict(feature=c, classes=len(vc), top=vc.index[0], top_share=p_.iloc[0],
                        classes_ge_1pct=(p_ >= 0.01).sum(), smallest=vc.iloc[-1],
                        entropy=prep.norm_entropy(df[c].dropna() if c not in prep.MULTI_SELECT else ex),
                        multi_share=combos))
bal = pd.DataFrame(balance)
print(bal.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

fig, axs = plt.subplots(4, 3, figsize=(12, 13))
titles = {"body_type": "Кузов", "fuel": "Топливо (мультивыбор, разложен)", "transmission": "Коробка (мультивыбор)",
          "drive": "Привод", "steering_wheel": "Руль", "color": "Цвет", "condition": "Состояние",
          "technical_condition": "Тех. состояние (мультивыбор)", "customs_cleared": "Растаможка",
          "availability": "Наличие", "registration_country": "Страна учёта (топ-6)", "vin_status": "VIN"}
for ax, c in zip(axs.flat, CAT):
    if c in prep.MULTI_SELECT:
        vc = df[c + "_set"].explode().dropna().value_counts()
    else:
        vc = df[c].value_counts()
    k = 6 if c == "registration_country" else 10
    if len(vc) > k:
        vc = pd.concat([vc.head(k), pd.Series({"прочие": vc.iloc[k:].sum()})])
    vc = pd.concat([vc, pd.Series({"(пропуск)": df[c].isna().sum()})])
    hbar(ax, vc, highlight={"(пропуск)": NEUTRAL, "прочие": NEUTRAL}, total=len(df))
    ax.set_title(titles[c], fontsize=10.5)
    ax.tick_params(axis="x", labelsize=7.5)
plt.tight_layout()
save("10_categoricals")

# brands & models
section("BRANDS / MODELS")
bc = cars.brand.value_counts()
print("brands:", len(bc), " >=1000:", (bc >= 1000).sum(), " >=100:", (bc >= 100).sum(), " >=30:", (bc >= 30).sum(),
      " <30:", (bc < 30).sum())
cum = bc.cumsum() / bc.sum()
print("top-5 cover:", pct(cum.iloc[4]), " top-10:", pct(cum.iloc[9]), " top-15:", pct(cum.iloc[14]), " top-20:", pct(cum.iloc[19]))
mc = cars.dropna(subset=["model"]).groupby(["brand", "model"]).size().sort_values(ascending=False)
print("brand-model pairs:", len(mc), " >=100:", (mc >= 100).sum(), " >=30:", (mc >= 30).sum(), " <10:", (mc < 10).sum())
mcum = mc.cumsum() / mc.sum()
print("ads covered by models with >=30 ads:", pct(mc[mc >= 30].sum() / mc.sum()), " >=100:", pct(mc[mc >= 100].sum() / mc.sum()))
print("top models:\n", (mc.head(20)).to_string())
print("model null:", df.model.isna().sum(), " (of which Другие автомобили:", df[df.model.isna()].brand.eq("Другие автомобили").sum(), ")",
      " 'Другая модель':", (df.model == "Другая модель").sum())

fig, axs = plt.subplots(1, 2, figsize=(12, 5.4), gridspec_kw={"width_ratios": [1.15, 1]})
hbar(axs[0], bc.head(20), total=len(cars))
axs[0].set_title("Топ-20 марок")
axs[0].tick_params(axis="x", labelsize=8)
n = np.arange(1, len(mc) + 1)
axs[1].plot(np.arange(1, len(bc) + 1), cum.values, color=BLUE, label=f"марки ({len(bc)})")
axs[1].plot(n, mcum.values, color=ORANGE, label=f"марка+модель ({len(mc)})")
axs[1].set_xscale("log")
axs[1].yaxis.set_major_formatter(mt.PercentFormatter(1))
axs[1].set_title("Накопленная доля объявлений")
axs[1].set_xlabel("топ-N категорий (лог.)")
for k, lab in [(10, "10 марок"), (100, "100 моделей")]:
    v = (cum if "мар" in lab else mcum).iloc[k - 1]
    axs[1].plot([k], [v], "o", color=INK, ms=4)
    axs[1].text(k * 1.15, v - 0.06, f"{lab}: {100 * v:.0f}%", fontsize=8.5, color=INK2)
axs[1].legend(loc="lower right", fontsize=8.5)
plt.tight_layout()
save("11_brands_models")

# ============================================================ label uniformity
section("LABEL UNIFORMITY")
for c in prep.MULTI_SELECT:
    multi = df[c].str.contains(", ", na=False)
    print(f"{c}: distinct strings={df[c].nunique()}, options={df[c + '_set'].explode().nunique()}, multi-valued ads={multi.sum()} ({pct(multi.mean())})")
contra = {
    "transmission: Автомат+Механика": df.transmission_set.map(lambda s: {"Автомат", "Механика"} <= s).sum(),
    "transmission: >=2 options": df.transmission.str.contains(", ", na=False).sum(),
    "technical: Идеальное + (Аварийное|Битый|На запчасти)": df.technical_condition_set.map(
        lambda s: "Идеальное" in s and bool(s & {"Аварийное", "Битый", "На запчасти"})).sum(),
    "payment: Возможен обмен + Обмена нет": df.payment_terms_set.map(lambda s: {"Возможен обмен", "Обмена нет"} <= s).sum(),
    "fuel: Электромобиль + Дизель": df.fuel_set.map(lambda s: {"Электромобиль", "Дизель"} <= s).sum(),
    "battery_capacity on non-EV/non-hybrid": (df.battery_capacity.notna() & ~df.fuel_set.map(lambda s: bool(s & {"Электромобиль", "Гибрид"}))).sum(),
    "battery_capacity total": df.battery_capacity.notna().sum(),
    "engine volume on pure EV": (df.fuel.eq("Электромобиль") & df.engine_volume_l.notna()).sum(),
    "condition Новый with mileage > 10 000 km": (df.condition.eq("Новый") & (df.mileage_n > 10_000)).sum(),
    "condition Новый total": df.condition.eq("Новый").sum(),
    "condition Новый and year <= 2015": (df.condition.eq("Новый") & (df.year_n <= 2015)).sum(),
}
for k, v in contra.items():
    print(f"  {k}: {v}")
mods = cars.dropna(subset=["model"]).groupby(["brand", "model"]).size().reset_index(name="n")
mods["norm"] = mods.model.str.lower().str.replace(r"[\s\-_.]", "", regex=True)
print("model case/space collisions within brand:", mods.duplicated(["brand", "norm"]).sum())
cyr = mods[mods.model.str.contains("[А-Яа-яЁё]") & (mods.model != "Другая модель")].sort_values("n", ascending=False)
print("models spelled in Cyrillic:", len(cyr), cyr.head(10)[["brand", "model", "n"]].values.tolist())
shared = mods[mods.model != "Другая модель"].groupby("model").agg(brands=("brand", lambda b: sorted(b)), n=("n", "sum"), k=("brand", "size"))
shared = shared[shared.k > 1].sort_values("n", ascending=False)
gm = mods.set_index(["brand", "model"]).n
print("machine-translated model names:", {f"{b} {m}": int(gm.get((b, m), 0)) for b, m in
      [("Voyah", "Бесплатно"), ("Honda", "Город"), ("Dodge", "Зарядник"), ("Mercedes-Benz", "Спринтер"), ("Nissan", "Мурано"), ("Toyota", "Тундра")]})
print("model names under >1 brand:", len(shared))
print(shared.head(15).to_string())
for m in ["Nexia", "Matiz", "Lacetti", "Gentra", "Damas", "Spark"]:
    sub = mods[mods.model == m]
    print(f"  {m}:", dict(zip(sub.brand, sub.n)))
junk = df[~df.is_car]
kw = {"велосипед|самокат": "велосипеды/самокаты", "запчаст|ключ|корпус|диск|шин": "запчасти",
      "выкуп|куплю|скупка": "«куплю/выкуп»", "самосвал|howo|хово|man|камаз|манипулятор|трактор|кран": "грузовики/спецтехника"}
print("«Другие автомобили»:", len(junk))
for pat, lab in kw.items():
    print(f"  {lab}: {junk.title.str.lower().str.contains(pat).sum()}")
auto_title = df.title.str.match(r"^.+: (\d{4}) г\.")
print("auto-composed titles:", pct(auto_title.mean()))

# =================================================================== duplicates
section("DUPLICATES")
print("ad_id unique:", df.ad_id.is_unique, " url unique:", df.url.is_unique, " image_id unique:", im.image_id.is_unique)
print("pairs by rule:", pairs.rule.value_counts().to_dict())
pr = pairs.drop_duplicates(["ad_id_x", "ad_id_y"])
a = df.set_index("ad_id")
px, py = a.loc[pr.ad_id_x], a.loc[pr.ad_id_y]
same = lambda c: (px[c].values == py[c].values).mean()
print("unique pairs:", len(pr), " same seller:", pct(same("user_id")), " same model:", pct(same("model")),
      " same year:", pct(same("year")), " same price:", pct(same("price")))
print("ads in clusters:", (df.dup_size > 1).sum(), " clusters:", df[df.dup_size > 1].dup_cluster.nunique(),
      " extra copies (is_dup):", df.is_dup.sum(), pct(df.is_dup.mean()))
print("cluster size distribution:", df[df.dup_size > 1].drop_duplicates("dup_cluster").dup_size.value_counts().sort_index().to_dict())
print("dup share by seller:", df.groupby("seller").is_dup.mean().round(3).to_dict())
print("dup share top brands:", df[df.brand.isin(bc.head(10).index)].groupby("brand").is_dup.mean().round(3).sort_values().to_dict())
xs = df[df.dup_size > 1].groupby("dup_cluster").user_id.nunique()
print("clusters spanning >1 seller account:", pct((xs > 1).mean()))
kd = df.duplicated(["user_id", "title", "description", "price"]).sum()
print("exact text duplicates (user,title,description,price):", kd)
print("ads without photos:", (~df.ad_id.isin(im.ad_id)).sum(), " p_hash null:", im.p_hash.isna().sum(),
      " ads with 2+ cover photos:", (im.groupby("ad_id").is_main.sum() > 1).sum())

fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
rule_ru = {"A_3plus_photos": "A: ≥3 общих фото", "B_cover_photo": "B: та же обложка\n+ марка/модель/год",
           "C_same_key": "C: марка/модель/год/\nцена/продавец"}
rc = pairs.rule.value_counts().rename(index=rule_ru)
hbar(axs[0], rc)
axs[0].set_title("Пары-дубликаты по правилам")
sh = df.groupby("seller").is_dup.mean().rename({"dealer": "дилеры", "private": "частные"})
hbar(axs[1], sh, fmt=lambda v: f"{100 * v:.1f}%")
axs[1].set_title("Доля лишних копий по типу продавца")
axs[1].xaxis.set_major_formatter(mt.PercentFormatter(1))
plt.tight_layout()
save("12_duplicates")

# ======================================================================= missing
section("MISSING")
KEY = ["model", "year_n", "body_type", "color", "fuel", "transmission", "engine_l", "drive",
       "steering_wheel", "registration_country", "condition", "technical_condition", "mileage_n",
       "customs_cleared", "availability", "payment_terms", "vin_status", "price_kgs"]
nul = df[KEY].isna()
yb_all = pd.cut(df.year_n, [1949, 1999, 2004, 2009, 2014, 2018, 2022, 2026],
                labels=["≤1999", "2000-04", "2005-09", "2010-14", "2015-18", "2019-22", "2023-26"])
heat = nul.drop(columns=["year_n"]).groupby(yb_all, observed=True).mean().T
heat["все"] = nul.drop(columns=["year_n"]).mean()
heat = heat.sort_values("все")
print((heat * 100).round(0).to_string())
print("missing by seller:\n", (nul.groupby(df.seller).mean().T * 100).round(0).to_string())
print("missing by is_vip:", (nul[["mileage_n", "customs_cleared", "condition"]].groupby(df.is_vip).mean() * 100).round(0).to_dict())
q = pd.to_datetime(df.created_date).dt.tz_localize(None).dt.to_period("Q")
print("missing by created quarter (last 6):\n", (nul[["mileage_n", "customs_cleared", "condition", "engine_l"]].groupby(q).mean().tail(6) * 100).round(0).to_string())
core = ["model", "year_n", "body_type", "fuel", "transmission", "engine_l"]
print("core-6 complete:", pct(df[core].notna().all(axis=1).mean()), " + drive:",
      pct(df[core + ["drive"]].notna().all(axis=1).mean()), " + mileage:", pct(df[core + ["mileage_n"]].notna().all(axis=1).mean()))
print("nulls per ad (16 attrs w/o price):", nul.drop(columns=["price_kgs"]).sum(axis=1).value_counts().sort_index().to_dict())

fig, ax = plt.subplots(figsize=(9, 6.2))
im_ = ax.imshow(heat.values, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(heat.shape[1]), heat.columns)
ax.set_yticks(range(heat.shape[0]), heat.index)
ax.grid(False)
for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):
        v = heat.values[i, j]
        ax.text(j, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=8, color="#ffffff" if v > 0.5 else INK)
ax.axvline(heat.shape[1] - 1.5, color="#fcfcfb", lw=3)
ax.set_title("Доля пропусков (%) по году выпуска")
ax.set_xlabel("год выпуска")
for s in ax.spines.values():
    s.set_visible(False)
save("13_missing_by_year")

corr_n = nul.drop(columns=["year_n"]).astype(float).corr()
order = corr_n.mean().sort_values().index
corr_n = corr_n.loc[order, order]
fig, ax = plt.subplots(figsize=(8.5, 7.2))
ax.imshow(corr_n.values, cmap=DIV, norm=TwoSlopeNorm(0, -1, 1))
ax.set_xticks(range(len(order)), order, rotation=60, ha="right")
ax.set_yticks(range(len(order)), order)
ax.grid(False)
for i in range(len(order)):
    for j in range(len(order)):
        if i != j and abs(corr_n.values[i, j]) >= 0.3:
            ax.text(j, i, f"{corr_n.values[i, j]:.1f}", ha="center", va="center", fontsize=7,
                    color="#ffffff" if abs(corr_n.values[i, j]) > 0.6 else INK)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title("Совместные пропуски: корреляция индикаторов «пусто» (подписи |r| ≥ 0.3)")
save("14_missing_corr")

# ================================================================ relationships
section("RELATIONSHIPS")
base = uniq.dropna(subset=["price_kgs"]).copy()
base["usd"] = base.price_kgs / RATE
print("analysis base (unique cars with clean price):", len(base))

g = base.dropna(subset=["year_n"]).groupby("year_n").usd
yr = pd.DataFrame({"q25": g.quantile(.25), "med": g.median(), "q75": g.quantile(.75), "n": g.size()})
yr = yr[(yr.n >= 30) & (yr.index >= 1985)]
print("median USD by year (every 3rd):\n", yr.med.iloc[::3].round(-2).to_string())
fig, ax = plt.subplots(figsize=(9, 3.8))
ax.fill_between(yr.index, yr.q25, yr.q75, color=BLUE, alpha=0.18, lw=0, label="Q1-Q3")
ax.plot(yr.index, yr.med, color=BLUE, label="медиана")
ax.set_yscale("log")
usd_axis(ax.yaxis)
ax.set_title("Цена по году выпуска (USD, лог. шкала; годы с ≥30 машинами)")
ax.legend(loc="upper left", fontsize=8.5)
save("15_price_vs_year")

fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
mb = base.dropna(subset=["mileage_n", "age"])
mb = mb[mb.mileage_n.between(1000, 600000)]
mb["kbin"] = (mb.mileage_n // 50000 * 50 + 25).astype(int)
for (lo, hi, lab), c in zip([(0, 7, "возраст ≤7 лет"), (8, 15, "8-15 лет"), (16, 40, "16+ лет")], (BLUE, ORANGE, AQUA)):
    s = mb[mb.age.between(lo, hi)].groupby("kbin").usd.agg(["median", "size"])
    s = s[s["size"] >= 30]
    axs[0].plot(s.index, s["median"], color=c, marker="o", ms=4, label=lab)
axs[0].set_yscale("log")
usd_axis(axs[0].yaxis)
axs[0].set_xlabel("пробег, тыс. км"); axs[0].set_title("Цена vs пробег внутри возрастных групп")
axs[0].legend(fontsize=8.5)
eb = base.dropna(subset=["engine_l"])
s = eb.groupby(eb.engine_l.round(1)).usd.agg(["median", "size"])
s = s[s["size"] >= 50]
axs[1].scatter(s.index, s["median"], s=np.sqrt(s["size"]) * 2.2, color=BLUE, alpha=0.8,
               edgecolor=prep.SURFACE, linewidth=1, zorder=3)
axs[1].set_yscale("log")
usd_axis(axs[1].yaxis)
axs[1].set_xlabel("объём, л")
axs[1].set_title("Медианная цена по объёму (площадь ~ число машин, n≥50)")
plt.tight_layout()
save("16_price_vs_mileage_engine")

fig, axs = plt.subplots(2, 2, figsize=(12, 7.5))
for ax, c, ttl in zip(axs.flat, ["body_type", "fuel", "transmission", "drive"], ["Кузов", "Топливо", "Коробка", "Привод"]):
    if c in prep.MULTI_SELECT:  # single-option ads only, to keep groups disjoint
        sub = base[base[c + "_set"].map(len) == 1]
    else:
        sub = base
    grp = sub.dropna(subset=[c]).groupby(c).usd
    keep = grp.size()[grp.size() >= 50].index
    stats = grp.median()[keep].sort_values()
    data = [np.log10(sub.loc[sub[c] == k, "usd"]) for k in stats.index]
    ax.boxplot(data, orientation="horizontal", widths=0.55, patch_artist=True, showfliers=False,
               medianprops=dict(color=INK, lw=1.5), whiskerprops=dict(color=MUTED),
               capprops=dict(color=MUTED), boxprops=dict(facecolor=BLUE_RAMP[1], edgecolor=BLUE_RAMP[4]))
    ax.set_yticks(range(1, len(stats) + 1), [f"{k}  (n={sp(grp.size()[k])})" for k in stats.index], fontsize=8.5)
    ax.set_xticks(np.log10([1000, 3000, 10000, 30000, 100000]), ["$1k", "$3k", "$10k", "$30k", "$100k"])
    ax.grid(axis="y", visible=False)
    ax.set_title(ttl)
plt.tight_layout()
save("17_price_by_category")

# informativeness: eta^2 of log price, vs fill rate
feat = {
    "brand": base.brand, "model": base.brand + "|" + base.model.fillna("NA").where(base.model.notna()),
    "year": base.year_n.round(0), "mileage": pd.qcut(base.mileage_n, 20, duplicates="drop"),
    "engine": base.engine_l.round(1), "body_type": base.body_type, "fuel": base.fuel,
    "transmission": base.transmission, "drive": base.drive, "steering_wheel": base.steering_wheel,
    "color": base.color, "condition": base.condition, "technical_condition": base.technical_condition,
    "customs_cleared": base.customs_cleared, "availability": base.availability,
    "registration_country": base.registration_country, "vin_status": base.vin_status,
    "payment_terms": base.payment_terms, "region": base.region_fixed, "city": base.city,
    "seller": base.seller, "battery_capacity": base.battery_capacity,
}
fillmap = {"year": df.year_n.notna().mean(), "mileage": df.mileage_n.notna().mean(), "engine": df.engine_l.notna().mean(),
           "region": 1.0, "city": 1.0, "seller": 1.0, "brand": 1.0, "model": df.model.notna().mean()}
info = pd.DataFrame([dict(feature=k, eta2=prep.eta_squared(v, base.log_price),
                          fill=fillmap.get(k, df[k].notna().mean() if k in df else np.nan)) for k, v in feat.items()])
info = info.sort_values("eta2", ascending=False)
print(info.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
num = base[["log_price", "year_n", "mileage_n", "engine_l", "image_count", "views", "favorite_count"]]
sp = num.corr(method="spearman")
print("spearman with log_price:", sp.log_price.round(3).to_dict())
print("spearman year~mileage:", round(sp.loc["year_n", "mileage_n"], 3))

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.scatter(info.fill, info.eta2, s=38, color=BLUE, edgecolor="#fcfcfb", linewidth=1.2, zorder=3)
off = {"engine": (6, -10), "drive": (6, 2), "body_type": (6, 4), "fuel": (-26, 5), "transmission": (6, -3),
       "brand": (8, -3), "city": (8, -4), "region": (8, -6), "seller": (8, 0), "color": (6, 3),
       "steering_wheel": (-95, -3), "registration_country": (6, -11), "condition": (6, 3),
       "customs_cleared": (-50, -13), "payment_terms": (-92, 3), "availability": (6, -7), "model": (8, -3), "year": (8, -3)}
for _, r in info.iterrows():
    ax.annotate(r.feature, (r.fill, r.eta2), xytext=off.get(r.feature, (6, 0)), textcoords="offset points",
                fontsize=8.5, color=INK)
ax.xaxis.set_major_formatter(mt.PercentFormatter(1)); ax.yaxis.set_major_formatter(mt.PercentFormatter(1))
ax.set_xlabel("заполненность (после очистки)")
ax.set_ylabel("η²: доля дисперсии log(цены),\nобъяснённая признаком в одиночку")
ax.set_title("Информативность признака для цены vs его заполненность")
ax.axvline(0.8, color=MUTED, lw=1, ls=(0, (3, 3)))
ax.set_xlim(0, 1.13)
save("18_feature_informativeness")

cv_cols = ["body_type", "fuel", "transmission", "drive", "steering_wheel", "color", "condition", "technical_condition",
           "customs_cleared", "registration_country", "region_fixed", "seller", "brand_top"]
cvb = uniq.assign(brand_top=uniq.brand.where(uniq.brand.isin(bc.head(15).index), "other"))
mat = pd.DataFrame(index=cv_cols, columns=cv_cols, dtype=float)
for i, a_ in enumerate(cv_cols):
    for b_ in cv_cols[i:]:
        v = 1.0 if a_ == b_ else prep.cramers_v(*[cvb.dropna(subset=[a_, b_])[x] for x in (a_, b_)])
        mat.loc[a_, b_] = mat.loc[b_, a_] = v
top_pairs = mat.where(np.triu(np.ones(mat.shape, bool), 1)).stack().sort_values(ascending=False).head(10)
print("top Cramér's V pairs:\n", top_pairs.round(3).to_string())
fig, ax = plt.subplots(figsize=(8, 6.8))
ax.imshow(mat.values.astype(float), cmap=SEQ, vmin=0, vmax=0.6)
ax.set_xticks(range(len(cv_cols)), cv_cols, rotation=60, ha="right")
ax.set_yticks(range(len(cv_cols)), cv_cols)
ax.grid(False)
for i in range(len(cv_cols)):
    for j in range(len(cv_cols)):
        if i != j:
            v = mat.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="#ffffff" if v > 0.35 else INK)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title("Связь категориальных признаков (V Крамера; шкала обрезана на 0.6)")
save("19_cramers_v")

bb = base[base.brand.isin(bc.head(20).index)].groupby("brand").agg(med=("usd", "median"), n=("usd", "size"),
                                                                  year=("year_n", "median")).sort_values("med", ascending=False)
print("brand median USD & median year:\n", bb.round(0).to_string())
fig, ax = plt.subplots(figsize=(9, 5.6))
s = bb.med[::-1]
ax.barh(range(len(s)), s.values, color=BLUE, height=0.68)
ax.set_yticks(range(len(s)), [f"{k}" for k in s.index])
for i, k in enumerate(s.index):
    ax.text(s[k] * 1.01 + 200, i, f"${s[k] / 1000:.1f}k · медиана года {bb.year[k]:.0f} · n={bb.n[k]:,}".replace(",", " "),
            va="center", fontsize=8, color=INK2)
ax.set_xlim(0, s.max() * 1.6)
ax.grid(axis="y", visible=False)
ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"${v / 1000:g}k"))
ax.set_title("Медианная цена по топ-20 маркам (уникальные машины)")
save("20_brand_price")

sd = base.groupby("seller").agg(n=("usd", "size"), med_usd=("usd", "median"), med_year=("year_n", "median"),
                                ev=("fuel_set", lambda s: s.map(lambda x: "Электромобиль" in x).mean()))
print("seller types:\n", sd.round(3).to_string())
print("users: pro", users.pro.sum(), " ads_count>=3:", (users.ads_count >= 3).sum(),
      " response_time<0:", (users.response_time < 0).sum(), " response_rate null:", users.response_rate.isna().sum())

print("\nfigures:", sorted(p.name for p in FIG.glob("*.png")))
