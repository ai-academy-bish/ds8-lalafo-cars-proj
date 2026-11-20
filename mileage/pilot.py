"""Mileage pilot: linear models (LinearRegression / Ridge), CV on train, one test at the end.

Run (after build_dataset.py):  venv/bin/python mileage/pilot.py
Writes figures to mileage/fig, numbers to mileage/results, the final model to
mileage/results/final_model.joblib. Every decision is made on 5-fold CV of the
train part; the test part is scored once, after the model is chosen.
"""
from __future__ import annotations

import os

# Small dense problems: dozens of BLAS threads only fight each other (8 min -> seconds).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import scipy.stats as st
import sklearn
import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error, median_absolute_error,
                             r2_score, root_mean_squared_error)
from sklearn.model_selection import KFold, cross_val_predict, cross_validate, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "EDA"))
import prep  # noqa: E402  (palette + matplotlib style shared with the EDA report)
from prep import AXIS, BLUE_RAMP, INK, INK2, MUTED, NEUTRAL, SERIES  # noqa: E402

warnings.filterwarnings("ignore", message=".*unknown categories.*")
warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------------------------------------------------------- configuration
RANDOM_STATE = 42
N_FOLDS = 5
MIN_FREQ = 30                       # one-hot: rarer categories pool into "infrequent"
ALPHAS = np.logspace(-3, 4, 71)     # Ridge search grid
AGE_CENTER = 7.0                    # fixed (not fitted) centre for the squared-age term
# Decided before training (checklist stage 1):
PRIMARY = "mae"
BAR_VS_AGE_ONLY = 0.10              # pilot succeeds if CV MAE beats "LR on age" by >= 10 %
BAR_TEST_GAP = 0.05                 # ...and test MAE is within +-5 % of CV MAE
BAR_LONG_TERM_KM = 35_000           # later goal: MAE <= 35k km (~25 % of median mileage)

DATA = HERE / "data" / "mileage_2010_2024.parquet"
FIG, RES = HERE / "fig", HERE / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)
prep.set_style()
BLUE, ORANGE, AQUA = SERIES[0], SERIES[1], SERIES[2]
SEQ = LinearSegmentedColormap.from_list("seq", ["#fcfcfb"] + BLUE_RAMP)
DIV = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])

NUM_BASE = ["age", "engine_l", "price_kgs", "image_count"]
CAT_BASE = ["brand", "body_type", "transmission", "drive", "steering_wheel", "condition", "tech_condition",
            "customs_cleared", "registration_country", "region", "seller", "currency", "is_vip"]
BIN = ["fuel_petrol", "fuel_diesel", "fuel_lpg", "fuel_hybrid", "fuel_ev", "fuel_isna"]
SCORING = {"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2",
           "medae": "neg_median_absolute_error", "mape": "neg_mean_absolute_percentage_error"}


def sp(v) -> str:
    return f"{v:,.0f}".replace(",", " ")


def km_fmt(v, _=None) -> str:
    return f"{v / 1000:,.0f}k".replace(",", " ")


def save(name: str):
    plt.savefig(FIG / f"{name}.png", dpi=130, bbox_inches="tight")
    plt.close("all")


# ------------------------------------------------------------------- estimators
class AnnualRateRegressor(RegressorMixin, BaseEstimator):
    """Fits `estimator` on mileage / age (km per year) and predicts rate * age, in km."""

    def __init__(self, estimator=None, age_col: str = "age"):
        self.estimator = estimator
        self.age_col = age_col

    def fit(self, X, y):
        age = X[self.age_col].to_numpy(float)
        self.estimator_ = clone(self.estimator).fit(X, np.asarray(y, float) / age)
        return self

    def predict(self, X):
        return self.estimator_.predict(X) * X[self.age_col].to_numpy(float)


def make_prep(num, cat, binary, drop="first") -> ColumnTransformer:
    """drop='first' for plain OLS (avoids the dummy-variable trap); None for Ridge, where the
    penalty resolves the collinearity and each category gets its own, readable deviation."""
    parts = []
    if num:
        parts.append(("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                                       ("scale", StandardScaler())]), num))
        parts.append(("miss", MissingIndicator(features="missing-only"), num))
    if cat:
        parts.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("onehot", OneHotEncoder(drop=drop, min_frequency=MIN_FREQ,
                                     handle_unknown="infrequent_if_exist", sparse_output=False)),
        ]), cat))
    if binary:
        parts.append(("bin", "passthrough", binary))
    return ColumnTransformer(parts, verbose_feature_names_out=True)


def make_estimator(exp: dict):
    if exp["model"].startswith("dummy"):
        return DummyRegressor(strategy=exp["model"].split("_")[1])
    reg = LinearRegression() if exp["model"] == "lr" else RidgeCV(alphas=ALPHAS)
    drop = "first" if exp["model"] == "lr" else None
    pipe = Pipeline([("prep", make_prep(exp["num"], exp["cat"], exp["bin"], drop=drop)), ("model", reg)])
    if exp["target"] == "log":
        return TransformedTargetRegressor(regressor=pipe, func=np.log, inverse_func=np.exp)
    if exp["target"] == "rate":
        return AnnualRateRegressor(pipe)
    return pipe


def inner_pipeline(fitted):
    """The fitted Pipeline(prep, model) inside whatever target wrapper."""
    if isinstance(fitted, TransformedTargetRegressor):
        return fitted.regressor_
    if isinstance(fitted, AnnualRateRegressor):
        return fitted.estimator_
    return fitted


def to_target_space(y, X, target):
    if target == "log":
        return np.log(y)
    if target == "rate":
        return y / X["age"].to_numpy(float)
    return y


def metrics(y, p) -> dict:
    return {"mae": mean_absolute_error(y, p), "rmse": root_mean_squared_error(y, p), "r2": r2_score(y, p),
            "medae": median_absolute_error(y, p), "mape": mean_absolute_percentage_error(y, p)}


# ========================================================================== data
print(f"sklearn {sklearn.__version__} | pandas {pd.__version__} | numpy {np.__version__} | statsmodels {sm.__version__}")
df = pd.read_parquet(DATA)
for c in CAT_BASE + ["brand_model"]:
    df[c] = df[c].astype(object).where(df[c].notna(), np.nan)
df["age_c2"] = (df.age - AGE_CENTER) ** 2
df["log_age_c2"] = (df.log_age - np.log(AGE_CENTER)) ** 2
train, test = df[df.split == "train"].reset_index(drop=True), df[df.split == "test"].reset_index(drop=True)
y_tr, y_te = train.mileage_km.to_numpy(), test.mileage_km.to_numpy()
print(f"train {len(train)}  test {len(test)}")
split_cmp = pd.DataFrame({"train": train.mileage_km.describe(), "test": test.mileage_km.describe()}).round(0)
print(split_cmp.T.to_string())
print("skew mileage:", round(train.mileage_km.skew(), 2), " skew log:", round(np.log(train.mileage_km).skew(), 2),
      " skew km/yr:", round((train.mileage_km / train.age).skew(), 2))
print("feature-matrix rows per feature (rough): ", round(len(train) / (len(NUM_BASE) * 2 + len(BIN) + 60), 0))

# ------------------------------------------------------- target & features EDA
fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
axs[0].hist(train.mileage_km, bins=60, color=BLUE)
axs[0].set_title(f"Пробег, км (skew {train.mileage_km.skew():.2f})")
axs[0].xaxis.set_major_formatter(mt.FuncFormatter(km_fmt))
axs[1].hist(np.log(train.mileage_km), bins=60, color=BLUE)
axs[1].set_title(f"log(пробег) (skew {np.log(train.mileage_km).skew():.2f})")
kmy = train.mileage_km / train.age
axs[2].hist(kmy, bins=60, color=BLUE)
axs[2].set_title(f"Пробег в год, км (skew {kmy.skew():.2f})")
axs[2].xaxis.set_major_formatter(mt.FuncFormatter(km_fmt))
for ax in axs:
    ax.set_yticks([])
plt.tight_layout()
save("01_target")

fig, axs = plt.subplots(2, 2, figsize=(11, 7.5))
g = train.groupby("year").mileage_km
yr = pd.DataFrame({"q25": g.quantile(.25), "med": g.median(), "q75": g.quantile(.75)})
ax = axs[0, 0]
ax.fill_between(yr.index, yr.q25, yr.q75, color=BLUE, alpha=0.18, lw=0, label="Q1-Q3")
ax.plot(yr.index, yr.med, color=BLUE, marker="o", ms=4, label="медиана")
ax.plot(yr.index, 14_000 * (2026 - yr.index), color=MUTED, ls=(0, (3, 3)), lw=1.2, label="14 тыс. км × возраст")
ax.yaxis.set_major_formatter(mt.FuncFormatter(km_fmt))
ax.set_title("Пробег по году выпуска"); ax.legend(fontsize=8.5)
ax = axs[0, 1]
m = train.log_price.notna()
ax.hexbin(np.exp(train.log_price[m]) / 87.45, train.mileage_km[m], gridsize=45, xscale="log", cmap=SEQ, mincnt=1, bins="log")
ax.set_title("Пробег vs цена (USD, лог.; цвет — плотность)")
ax.yaxis.set_major_formatter(mt.FuncFormatter(km_fmt))
ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"${v / 1000:g}k"))
ax = axs[1, 0]
for lab, sub, c in [("дилер", train[train.seller == "dealer"], ORANGE), ("частный", train[train.seller == "private"], BLUE)]:
    gg = sub.groupby("year").mileage_km.median()
    gg = gg[sub.groupby("year").size() >= 30]
    ax.plot(gg.index, gg.values, color=c, marker="o", ms=4, label=lab)
ax.yaxis.set_major_formatter(mt.FuncFormatter(km_fmt))
ax.set_title("Медианный пробег: дилеры vs частные (n≥30)"); ax.legend(fontsize=8.5)
ax = axs[1, 1]
fuel_lab = {"fuel_petrol": "бензин", "fuel_diesel": "дизель", "fuel_lpg": "газ", "fuel_hybrid": "гибрид", "fuel_ev": "электро"}
rows = []
for col, lab in fuel_lab.items():
    s = train[train[col] == 1]
    rows.append((lab, (s.mileage_km / s.age).median(), len(s)))
rows.sort(key=lambda r: r[1])
ax.barh(range(len(rows)), [r[1] for r in rows], color=BLUE, height=0.65)
ax.set_yticks(range(len(rows)), [f"{r[0]} (n={sp(r[2])})" for r in rows])
for i, r in enumerate(rows):
    ax.text(r[1] * 1.01, i, km_fmt(r[1]), va="center", fontsize=8.5, color=INK2)
ax.xaxis.set_major_formatter(mt.FuncFormatter(km_fmt))
ax.grid(axis="y", visible=False)
ax.set_title("Медианный годовой пробег по топливу")
plt.tight_layout()
save("02_target_vs_features")

num_corr = train[["mileage_km", "age", "log_age", "price_kgs", "log_price", "engine_l", "image_count"]].assign(
    log_mileage=np.log(train.mileage_km)).corr()
print("pearson corr with mileage:\n", num_corr["mileage_km"].round(3).to_string())
print("pearson corr with log_mileage:\n", num_corr["log_mileage"].round(3).to_string())
fig, ax = plt.subplots(figsize=(6.6, 5.6))
cm = DIV
im_ = ax.imshow(num_corr.values, cmap=cm, vmin=-1, vmax=1)
ax.set_xticks(range(len(num_corr)), num_corr.columns, rotation=50, ha="right")
ax.set_yticks(range(len(num_corr)), num_corr.columns)
ax.grid(False)
for i in range(len(num_corr)):
    for j in range(len(num_corr)):
        v = num_corr.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5, color="#ffffff" if abs(v) > 0.6 else INK)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title("Корреляции Пирсона (train)")
save("03_corr")

# =================================================================== experiments
BASE = dict(num=NUM_BASE, cat=CAT_BASE, bin=BIN)
LOGP = dict(num=["age", "engine_l", "log_price", "image_count"], cat=CAT_BASE, bin=BIN)
LOGAGE = dict(num=["log_age", "engine_l", "log_price", "image_count"], cat=CAT_BASE, bin=BIN)
EXPS = [
    dict(id="B0", title="Среднее (Dummy mean)", hyp="Планка: модель, которая ничего не знает", model="dummy_mean", target="raw", num=[], cat=[], bin=[]),
    dict(id="B1", title="Медиана (Dummy median)", hyp="Оптимальная константа для MAE", model="dummy_median", target="raw", num=[], cat=[], bin=[]),
    dict(id="B2", title="LR: только возраст", hyp="Один сильнейший признак: пробег ≈ a + b·возраст", model="lr", target="raw", num=["age"], cat=[], bin=[]),
    dict(id="E1", title="LR: все базовые признаки", hyp="Тех. характеристики, цена и тип продавца добавляют к возрасту", model="lr", target="raw", **BASE),
    dict(id="E2", title="E1 + log(цены)", hyp="H1: связь пробега с ценой мультипликативная (лог-цена линейнее)", model="lr", target="raw", **LOGP),
    dict(id="E3", title="E2 + log(пробега), log(возраста)", hyp="H2: пробег = возраст^b × интенсивность → линейно в логарифмах", model="lr", target="log", **LOGAGE),
    dict(id="E4", title="E2 + цель = пробег в год", hyp="H3: годовой пробег однороднее общего; прогноз = ставка × возраст", model="lr", target="rate", **LOGP),
]
results, fitted_cv = [], {}
cv = KFold(N_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def run(exp: dict):
    t0 = time.time()
    est = make_estimator(exp)
    r = cross_validate(est, train, y_tr, cv=cv, scoring=SCORING, return_train_score=True, return_estimator=True)
    row = {"id": exp["id"], "title": exp["title"], "hypothesis": exp["hyp"], "model": exp["model"], "target": exp["target"]}
    for k in SCORING:
        sign = 1 if k == "r2" else -1
        row[f"cv_{k}"] = sign * r[f"test_{k}"].mean()
        row[f"cv_{k}_std"] = r[f"test_{k}"].std()
        row[f"train_{k}"] = sign * r[f"train_{k}"].mean()
    fold0 = r["estimator"][0]
    if not exp["model"].startswith("dummy"):
        pipe = inner_pipeline(fold0)
        row["n_features"] = len(pipe.named_steps["prep"].get_feature_names_out())
        if exp["model"] == "ridge":
            row["alpha_median"] = float(np.median([inner_pipeline(e).named_steps["model"].alpha_ for e in r["estimator"]]))
    else:
        row["n_features"] = 0
    row["seconds"] = round(time.time() - t0, 1)
    results.append(row)
    fitted_cv[exp["id"]] = exp
    print(f"{exp['id']:<4} {exp['title']:<42} CV MAE {row['cv_mae']:>9,.0f} ± {row['cv_mae_std']:>6,.0f}  "
          f"train MAE {row['train_mae']:>9,.0f}  R² {row['cv_r2']:.3f}  n_feat {row.get('n_features', 0)}"
          + (f"  alpha {row['alpha_median']:.3g}" if "alpha_median" in row else ""))
    return row


print("\n=== CV experiments (5-fold on train) ===")
for e in EXPS:
    run(e)

# step 2: the best target/feature form among E2-E4 carries forward
best_form = min((r for r in results if r["id"] in ("E2", "E3", "E4")), key=lambda r: r["cv_mae"])["id"]
form = next(e for e in EXPS if e["id"] == best_form)
print(f"best form among E2-E4: {best_form}")
age_sq = "log_age_c2" if "log_age" in form["num"] else "age_c2"
E5 = dict(form, id="E5", title=f"{best_form} + Ridge (alpha по CV)", hyp="H4: L2-регуляризация стабилизирует десятки one-hot колонок", model="ridge")
E6 = dict(E5, id="E6", title="E5 + марка×модель (≥30 объявлений)", hyp="H5: конкретная модель (такси, коммерческие) задаёт интенсивность эксплуатации", cat=CAT_BASE + ["brand_model"])
E7 = dict(E6, id="E7", title="E6 + квадрат возраста", hyp="H6: годовой пробег меняется с возрастом нелинейно", num=form["num"] + [age_sq])
for e in (E5, E6, E7):
    EXPS.append(e)
    run(e)

res = pd.DataFrame(results)
res.to_csv(RES / "experiments.csv", index=False)
cand = res[res.id.str.startswith("E")]
best_id = cand.loc[cand.cv_mae.idxmin(), "id"]
best = next(e for e in EXPS if e["id"] == best_id)
b2 = res.set_index("id").loc["B2"]
b0 = res.set_index("id").loc["B0"]
rb = res.set_index("id").loc[best_id]
print(f"\nselected: {best_id} ({best['title']}); vs B2: {1 - rb.cv_mae / b2.cv_mae:.1%} better; vs B0: {1 - rb.cv_mae / b0.cv_mae:.1%}")

fig, ax = plt.subplots(figsize=(9.5, 5))
order = res.iloc[::-1]
y = np.arange(len(order))
cols = [NEUTRAL if i.startswith("B") else (ORANGE if i == best_id else BLUE) for i in order.id]
ax.barh(y, order.cv_mae, xerr=order.cv_mae_std, color=cols, height=0.62, error_kw=dict(ecolor=INK2, lw=1, capsize=3))
ax.scatter(order.train_mae, y, marker="|", s=220, color=INK, zorder=4, label="MAE на train (среднее по фолдам)")
ax.set_yticks(y, [f"{r.id}  {r.title}" for r in order.itertuples()], fontsize=8.5)
for i, r in enumerate(order.itertuples()):
    ax.text(r.cv_mae + r.cv_mae_std + 900, i, f"{r.cv_mae / 1000:.1f}k", va="center", fontsize=8.5, color=INK2)
ax.axvline(b2.cv_mae * (1 - BAR_VS_AGE_ONLY), color=MUTED, lw=1, ls=(0, (3, 3)))
ax.text(b2.cv_mae * (1 - BAR_VS_AGE_ONLY), len(order) - 0.4, " планка: B2 − 10%", fontsize=8, color=MUTED)
ax.xaxis.set_major_formatter(mt.FuncFormatter(km_fmt))
ax.set_xlabel("MAE, км (5-fold CV на train; усы — std между фолдами)")
ax.grid(axis="y", visible=False)
ax.set_xlim(0, res.cv_mae.max() * 1.15)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=8.5)
ax.set_title("Эксперименты: CV MAE (оранжевый — выбранная модель)")
save("04_cv_experiments")

# ============================================================ selected model: OOF
final_est = make_estimator(best)
oof = cross_val_predict(final_est, train, y_tr, cv=cv)
resid = y_tr - oof
print("OOF metrics:", {k: round(v, 3) for k, v in metrics(y_tr, oof).items()})
print(f"OOF predictions < 0: {(oof < 0).sum()}  < 10 000 km: {(oof < 10_000).sum()}  min {oof.min():,.0f}")

sizes, tr_sc, va_sc = learning_curve(make_estimator(best), train, y_tr, cv=cv, train_sizes=np.linspace(0.1, 1, 8),
                                     scoring="neg_mean_absolute_error", shuffle=True, random_state=RANDOM_STATE)
lc = pd.DataFrame({"n": sizes, "train": -tr_sc.mean(1), "cv": -va_sc.mean(1), "cv_std": va_sc.std(1)})
print("learning curve:\n", lc.round(0).to_string(index=False))
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.plot(lc.n, lc.train, color=MUTED, marker="o", ms=4, label="train")
ax.plot(lc.n, lc.cv, color=BLUE, marker="o", ms=4, label="CV")
ax.fill_between(lc.n, lc.cv - lc.cv_std, lc.cv + lc.cv_std, color=BLUE, alpha=0.15, lw=0)
ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v / 1000:.1f}k"))
ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: sp(v)))
ax.set_xlabel("объём обучающей выборки"); ax.set_ylabel("MAE, км")
ax.set_title(f"Кривая обучения ({best_id})"); ax.legend(fontsize=8.5)
save("05_learning_curve")

fig, ax = plt.subplots(figsize=(6.2, 5.6))
hb = ax.hexbin(y_tr, oof, gridsize=55, cmap=SEQ, mincnt=1, bins="log")
lim = [0, np.percentile(y_tr, 99.5)]
ax.plot(lim, lim, color=ORANGE, lw=1.5, ls=(0, (4, 3)))
ax.set_xlim(lim); ax.set_ylim(lim)
ax.xaxis.set_major_formatter(mt.FuncFormatter(km_fmt)); ax.yaxis.set_major_formatter(mt.FuncFormatter(km_fmt))
ax.set_xlabel("факт, км"); ax.set_ylabel("прогноз (out-of-fold), км")
ax.set_title("Факт vs прогноз (OOF на train)")
save("06_fact_vs_pred")

fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
ax = axs[0]
ax.hexbin(oof, resid, gridsize=55, cmap=SEQ, mincnt=1, bins="log")
bins_ = pd.qcut(oof, 20, duplicates="drop")
bm = pd.DataFrame({"p": oof, "r": resid}).groupby(bins_, observed=True).agg(p=("p", "median"), r=("r", "mean"), s=("r", "std"))
ax.plot(bm.p, bm.r, color=ORANGE, lw=2, label="среднее остатка по бинам")
ax.plot(bm.p, bm.s, color=INK2, lw=1.2, ls=(0, (3, 2)), label="std остатка по бинам")
ax.axhline(0, color=INK, lw=0.8)
ax.xaxis.set_major_formatter(mt.FuncFormatter(km_fmt)); ax.yaxis.set_major_formatter(mt.FuncFormatter(km_fmt))
ax.set_xlabel("прогноз, км"); ax.set_ylabel("остаток = факт − прогноз")
ax.set_title("Остатки vs прогноз"); ax.legend(fontsize=8, loc="upper left")
ax = axs[1]
ax.hist(resid, bins=80, color=BLUE)
ax.xaxis.set_major_formatter(mt.FuncFormatter(km_fmt)); ax.set_yticks([])
ax.set_title(f"Распределение остатков (skew {st.skew(resid):.2f})")
ax = axs[2]
target_resid = to_target_space(y_tr, train, best["target"]) - to_target_space(oof, train, best["target"])
(osm, osr), (slope, intercept, _) = st.probplot(target_resid, dist="norm")
ax.scatter(osm, osr, s=4, color=BLUE, alpha=0.5)
ax.plot(osm, slope * osm + intercept, color=ORANGE, lw=1.5)
ax.set_xlabel("теоретические квантили N(0,1)"); ax.set_ylabel("остатки в пространстве цели")
if best["target"] != "log":
    ax.yaxis.set_major_formatter(mt.FuncFormatter(km_fmt))
ax.set_title(f"Q-Q plot (цель: {best['target']})")
plt.tight_layout()
save("07_residuals")

seg = train.assign(err=np.abs(resid), res=resid,
                   age_band=pd.cut(train.age, [1, 4, 7, 11, 16], labels=["2022-24", "2019-21", "2015-18", "2010-14"]))
seg_tables = {}
for col in ["age_band", "seller", "currency", "region", "mileage_rule", "condition"]:
    t = seg.groupby(col, observed=True).agg(n=("err", "size"), mae=("err", "mean"), bias=("res", "mean"),
                                           median_km=("mileage_km", "median"))
    t["mae_pct_of_median"] = t.mae / t.median_km
    seg_tables[col] = t
    print(f"\nOOF by {col}:\n", t.round(3).to_string())
tb = seg[seg.brand.isin(seg.brand.value_counts().head(12).index)].groupby("brand").agg(
    n=("err", "size"), mae=("err", "mean"), bias=("res", "mean"), median_km=("mileage_km", "median")).sort_values("mae")
seg_tables["brand"] = tb
print("\nOOF by brand:\n", tb.round(0).to_string())

fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1, 0.8, 1.3]})
for ax, (col, ttl) in zip(axs, [("age_band", "по году выпуска"), ("seller", "по типу продавца"), ("brand", "по маркам (топ-12)")]):
    t = seg_tables[col]
    if col == "seller":
        t = t.rename(index={"dealer": "дилер", "private": "частный"})
    ax.barh(range(len(t)), t.mae, color=BLUE, height=0.62)
    ax.set_yticks(range(len(t)), [f"{i} (n={sp(n)})" for i, n in zip(t.index, t.n)], fontsize=8.5)
    for i, (m_, b_) in enumerate(zip(t.mae, t.bias)):
        ax.text(m_ * 1.01, i, f"{m_ / 1000:.1f}k · смещ. {b_ / 1000:+.1f}k", va="center", fontsize=8, color=INK2)
    ax.set_xlim(0, t.mae.max() * 1.55)
    ax.xaxis.set_major_formatter(mt.FuncFormatter(km_fmt))
    ax.grid(axis="y", visible=False)
    ax.set_title(f"MAE {ttl}")
plt.tight_layout()
save("08_errors_by_segment")

top_err = seg.assign(pred=oof).nlargest(10, "err")[["ad_id", "brand_model", "year", "mileage_km", "pred", "price_kgs", "seller", "condition", "mileage_rule"]]
print("\ntop-10 OOF errors:\n", top_err.round(0).to_string(index=False))

# ================================================ final fit on all train + test
final = make_estimator(best).fit(train, y_tr)
pipe = inner_pipeline(final)
names = pipe.named_steps["prep"].get_feature_names_out()
coef = pd.Series(pipe.named_steps["model"].coef_, index=names)
alpha = getattr(pipe.named_steps["model"], "alpha_", None)
print(f"\nfinal model {best_id}: n_features={len(names)} alpha={alpha}")

scaler = pipe.named_steps["prep"].named_transformers_["num"].named_steps["scale"]
num_used = best["num"]
per_unit = pd.Series(coef[[f"num__{c}" for c in num_used]].values / scaler.scale_, index=num_used)
print("numeric coefficients per natural unit (target space):", per_unit.round(4).to_dict())
print("numeric coefficients per 1 SD:", coef[[f"num__{c}" for c in num_used]].round(4).to_dict(), " SD:", dict(zip(num_used, scaler.scale_.round(3))))
coef_sorted = coef.reindex(coef.abs().sort_values(ascending=False).index)
print("top-25 |coef| (target space):\n", coef_sorted.head(25).round(4).to_string())

unit = {"log": "%", "rate": "км/год", "raw": "км"}[best["target"]]
show = coef_sorted.head(22)[::-1]
vals = (np.exp(show) - 1) * 100 if best["target"] == "log" else show
fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(range(len(show)), vals, color=[BLUE if v > 0 else ORANGE for v in vals], height=0.65)
ax.set_yticks(range(len(show)), [n.replace("num__", "").replace("cat__", "").replace("bin__", "").replace("miss__", "isna: ") for n in show.index], fontsize=8.5)
ax.axvline(0, color=INK, lw=0.8)
ax.grid(axis="y", visible=False)
if best["target"] != "log":
    ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v / 1000:+.0f}k" if v else "0"))
ref = "против базовой категории" if best["model"] == "lr" else "отклонение категории (Ridge, без базовой)"
ax.set_xlabel(f"эффект на пробег, {unit} (числовые — на 1 SD; категории — {ref})")
ax.set_title(f"Крупнейшие коэффициенты {best_id} (синий — пробег выше, оранжевый — ниже)")
save("09_coefficients")

pred_te = final.predict(test)
m_te = metrics(y_te, pred_te)
base_te = {}
for bid in ("B0", "B2"):
    e = next(x for x in EXPS if x["id"] == bid)
    base_te[bid] = metrics(y_te, make_estimator(e).fit(train, y_tr).predict(test))
gap = m_te["mae"] / rb.cv_mae - 1
print(f"\nTEST {best_id}: " + "  ".join(f"{k}={v:,.3f}" for k, v in m_te.items()))
print("TEST baselines:", {k: {m: round(v, 3) for m, v in d.items()} for k, d in base_te.items()})
print(f"test vs CV MAE gap: {gap:+.1%}")
print(f"TEST predictions < 0: {(pred_te < 0).sum()}  min {pred_te.min():,.0f}")
te_seg = test.assign(err=np.abs(y_te - pred_te), res=y_te - pred_te,
                     age_band=pd.cut(test.age, [1, 4, 7, 11, 16], labels=["2022-24", "2019-21", "2015-18", "2010-14"])
                     ).groupby("age_band", observed=True).agg(n=("err", "size"), mae=("err", "mean"), bias=("res", "mean"))
print("TEST by age band:\n", te_seg.round(0).to_string())

# save + reload check (checklist stage 14)
joblib.dump(final, RES / "final_model.joblib")
reloaded = joblib.load(RES / "final_model.joblib")
assert np.allclose(reloaded.predict(test), pred_te), "reloaded model predicts differently"
print("joblib reload: identical predictions")

# ====================================================== statsmodels diagnostics
# OLS / VIF / Cook need a full-rank design: same features, dummy coding with a reference level
diag_prep = make_prep(best["num"], best["cat"], best["bin"], drop="first").fit(train)
Xp = diag_prep.transform(train).astype(float)
dnames = diag_prep.get_feature_names_out()
keep = Xp.std(0) > 0
Xp, names_k = Xp[:, keep], dnames[keep]
yt = to_target_space(y_tr, train, best["target"])
ols = sm.OLS(yt, sm.add_constant(Xp)).fit()
(RES / "ols_summary.txt").write_text(str(ols.summary(xname=["const"] + list(names_k))))
print(f"\nOLS (unregularised, target space): R²={ols.rsquared:.3f} adjR²={ols.rsquared_adj:.3f} cond.no={ols.condition_number:.3g}")
pv = pd.Series(ols.pvalues[1:], index=names_k)
print("p<0.05:", int((pv < 0.05).sum()), "of", len(pv))
Xc = sm.add_constant(Xp)
vif = pd.Series([variance_inflation_factor(Xc, i) for i in range(1, Xc.shape[1])], index=names_k).sort_values(ascending=False)
print("VIF top-12:\n", vif.head(12).round(1).to_string())
print("VIF > 10:", int((vif > 10).sum()), " VIF > 5:", int((vif > 5).sum()))
infl = OLSInfluence(ols)
cook = infl.cooks_distance[0]
thr = 4 / len(yt)
print(f"Cook's D > 4/n ({thr:.2e}): {(cook > thr).sum()} ({(cook > thr).mean():.1%});  max D={cook.max():.3f}")
top_cook = train.assign(cook=cook).nlargest(5, "cook")[["ad_id", "brand_model", "year", "mileage_km", "price_kgs", "cook"]]
print(top_cook.to_string(index=False))

summary = {
    "versions": {"sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__, "statsmodels": sm.__version__},
    "n_train": len(train), "n_test": len(test), "selected": best_id, "selected_title": best["title"],
    "target": best["target"], "alpha": alpha, "n_features": int(len(names)),
    "cv": {k: float(rb[f"cv_{k}"]) for k in SCORING}, "cv_mae_std": float(rb.cv_mae_std),
    "test": m_te, "test_baselines": base_te, "test_vs_cv_mae_gap": gap,
    "bars": {"beats_B2_by": float(1 - rb.cv_mae / b2.cv_mae), "required": BAR_VS_AGE_ONLY,
             "test_gap_ok": abs(gap) <= BAR_TEST_GAP, "long_term_km": BAR_LONG_TERM_KM},
    "per_unit_numeric_coef": per_unit.to_dict(),
    "vif_gt10": int((vif > 10).sum()), "cook_gt_4n": int((cook > thr).sum()),
}
(RES / "pilot_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float))
print("\nfigures:", sorted(p.name for p in FIG.glob("*.png")))
