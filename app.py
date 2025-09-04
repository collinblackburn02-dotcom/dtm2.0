import streamlit as st
import pandas as pd
import numpy as np
import duckdb
from itertools import combinations
from utils import resolve_col

st.set_page_config(page_title="Heavenly Health — Customer Insights", layout="wide")
st.title("✨ Heavenly Health — Customer Insights")
st.caption("Fast, ranked customer segments powered by DuckDB (GROUPING SETS).")

# ---------- Sidebar ----------
with st.sidebar:
    uploaded = st.file_uploader("Upload merged CSV", type=["csv"])
    st.markdown("---")
    metric_choice = st.radio(
        "Sort / Map metric",
        ["Conversion %", "Purchases", "Visitors", "Revenue / Visitor"],
        index=0
    )
    max_depth = st.slider("Max combo depth", 1, 4, 2, 1)
    top_n = st.slider("Top N", 10, 1000, 50, 10)

@st.cache_data(show_spinner=False)
def load_df(file):
    df = pd.read_csv(file)
    df.columns = [str(c).strip() for c in df.columns]
    return df

def to_datetime_series(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.to_datetime(pd.Series([None]*len(s)))

if not uploaded:
    st.info("Upload the merged CSV to begin.")
    st.stop()

# ---------- Load Data ----------
df = load_df(uploaded)

# Resolve key columns (case-insensitive)
purchase_col = resolve_col(df, "Purchase")
date_col     = resolve_col(df, "DATE")        # optional
msku_col     = resolve_col(df, "SKU")
revenue_col  = resolve_col(df, "Revenue")

if purchase_col is None:
    st.error("Missing Purchase column.")
    st.stop()

# Purchases are already 0/1 in this dataset
df["_PURCHASE"] = df[purchase_col].fillna(0).astype(int)

# Dates (optional)
df["_DATE"] = to_datetime_series(df[date_col]) if date_col else pd.NaT

# Revenue (default 0 if missing)
if revenue_col:
    df["_REVENUE"] = pd.to_numeric(df[revenue_col], errors="coerce").fillna(0.0)
else:
    df["_REVENUE"] = 0.0

# ---------- Segmentation Attributes ----------
seg_map = {
    "Age Range":   resolve_col(df, "Age_Range"),
    "Income Range":resolve_col(df, "Income_Range"),
    "Net Worth":   resolve_col(df, "Net_Worth"),
    "Credit Rating":resolve_col(df, "Credit_Rating"),
    "Gender":      resolve_col(df, "Gender"),
    "Homeowner":   resolve_col(df, "Home_Owner"),
    "Married":     resolve_col(df, "Married"),
    "Children":    resolve_col(df, "Children"),
}
# keep only found columns
seg_map = {label: col for label, col in seg_map.items() if col is not None}
seg_cols = list(seg_map.values())

# ---------- Filters ----------
with st.expander("🔎 Filters", expanded=True):
    dff = df.copy()

    # Treat 'U' as missing for Gender and Credit before collapsing
    for label, col in seg_map.items():
        if label in ("Gender", "Credit Rating") and col in dff.columns:
            dff.loc[dff[col].astype(str).str.upper().str.strip() == "U", col] = pd.NA

    # Date filter (only if DATE exists)
    if not dff["_DATE"].dropna().empty:
        mind, maxd = pd.to_datetime(dff["_DATE"].dropna().min()), pd.to_datetime(dff["_DATE"].dropna().max())
        c1, c2 = st.columns(2)
        with c1:
            date_range = st.date_input("Date range", (mind.date(), maxd.date()))
        with c2:
            include_undated = st.checkbox("Include no-date rows", value=True)

        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            start, end = date_range
            mask = dff["_DATE"].between(pd.to_datetime(start), pd.to_datetime(end))
            if include_undated:
                mask = mask | dff["_DATE"].isna()
            dff = dff[mask]

    # SKU contains filter
    sku_search = st.text_input("Most Recent SKU contains (optional)")
    if msku_col and sku_search:
        dff = dff[dff[msku_col].astype(str).str.contains(sku_search, case=False, na=False)]

    # Attribute filters
    selections = {}
    include_flags = {}
    if seg_cols:
        st.markdown("**Attributes**")
        cols = st.columns(3)
        idx = 0
        for label, col in seg_map.items():
            with cols[idx % 3]:
                mode = st.selectbox(
                    f"{label}: mode",
                    options=["Include", "Do not include"],
                    index=0,
                    key=f"mode_{label}"
                )
                include_flags[col] = (mode == "Include")
                values = sorted([x for x in dff[col].dropna().unique().tolist() if str(x).strip()])
                sel = st.multiselect(label, options=values, default=[], help="Empty = All")
                if sel:
                    selections[col] = sel
            idx += 1
        for col, vals in selections.items():
            dff = dff[dff[col].isin(vals)]

    st.caption(f"Rows after filters: **{len(dff):,}** / {len(df):,}")

# Collapse NaN/empty/"None"/"U" → "Unknown" AFTER filters (pre-DuckDB)
for col in seg_cols:
    if col in dff.columns:
        dff[col] = (
            dff[col]
            .fillna("Unknown")
            .replace({"": "Unknown", "None": "Unknown", "U": "Unknown", "u": "Unknown"})
        )

# Included attributes and required ones
include_cols = [c for c in seg_cols if include_flags.get(c, True)]
required_cols = [col for col, vals in selections.items() if len(vals) > 0 and include_flags.get(col, True)]

# ---------- DuckDB GROUPING SETS ----------
con = duckdb.connect()
con.register("t", dff)

attrs = [c for c in include_cols if c in dff.columns]
req_set = set(required_cols)

sets = []
for d in range(1, max_depth + 1):
    for s in combinations(attrs, d):
        if req_set.issubset(set(s)):
            sets.append("(" + ",".join([f'"{c}"' for c in s]) + ")")

if not sets:
    if required_cols:
        sets.append("(" + ",".join([f'"{c}"' for c in required_cols]) + ")")
    else:
        if attrs:
            sets.append("(" + f'"{attrs[0]}"' + ")")
        else:
            sets.append("()")  # grand total only

grouping_sets_sql = ",\n".join(sets)

# Top SKUs (for per-segment purchase counts by SKU)
top_skus = []
if msku_col and msku_col in dff.columns:
    top_skus = con.execute(f'''
        SELECT "{msku_col}" AS sku, COUNT(*) AS c
        FROM t
        WHERE _PURCHASE=1 AND "{msku_col}" IS NOT NULL AND TRIM("{msku_col}")<>''
        GROUP BY 1
        ORDER BY c DESC
        LIMIT 11
    ''').fetchdf()["sku"].astype(str).tolist()

sku_sums = ""
if top_skus:
    pieces = []
    for sku in top_skus:
        s_escaped = sku.replace("'", "''")
        # Keep column name "SKU:<name>" in SQL; we'll rename to "<name>" after fetch
        pieces.append(
            f"SUM(CASE WHEN \"{msku_col}\"='{s_escaped}' AND _PURCHASE=1 THEN 1 ELSE 0 END) AS \"SKU:{s_escaped}\""
        )
    sku_sums = ",\n  " + ",\n  ".join(pieces)

# Depth (count of non-null attributes in the grouping)
depth_expr = " + ".join([f"CASE WHEN \"{c}\" IS NULL THEN 0 ELSE 1 END" for c in attrs]) if attrs else "0"

# Revenue / RPV aggregates
revenue_sql = (
    "SUM(_REVENUE) AS revenue,\n  1.0 * SUM(_REVENUE) / NULLIF(COUNT(*),0) AS rpv"
    if revenue_col else
    "0.0 AS revenue,\n  0.0 AS rpv"
)

# Build SQL with conditional HAVING (only when grouped)
having_clause = "HAVING COUNT(*) >= ?" if attrs else ""

sql = f"""
SELECT
  {(", ".join([f'"{c}"' for c in attrs]) + "," if attrs else "" )}
  COUNT(*) AS Visitors,
  SUM(_PURCHASE) AS Purchases,
  100.0 * SUM(_PURCHASE) / NULLIF(COUNT(*),0) AS conv_rate,
  ({depth_expr}) AS Depth,
  {revenue_sql}
  {sku_sums}
FROM t
{'GROUP BY GROUPING SETS (' + grouping_sets_sql + ')' if attrs else ''}
{having_clause}
"""

# ---------- Ranked Conversion Table ----------
st.subheader("🏆 Ranked Conversion Table")
min_rows = st.number_input("Minimum Visitors per group", min_value=1, value=30, step=1)

params = [int(min_rows)] if attrs else []
res = con.execute(sql, params).fetchdf()

# ---- Rename SKU:* columns to just the SKU name
sku_prefixed_cols = [c for c in res.columns if str(c).startswith("SKU:")]
sku_rename = {c: c.split("SKU:", 1)[1] for c in sku_prefixed_cols}
res = res.rename(columns=sku_rename)

# ---- Sort by selected metric
sort_key_map = {
    "Conversion %": "conv_rate",
    "Purchases": "Purchases",
    "Visitors": "Visitors",
    "Revenue / Visitor": "rpv",
}
sort_key = sort_key_map[metric_choice]
res = res.sort_values(sort_key, ascending=False).head(top_n).reset_index(drop=True)

# ---- Prepare display data
all_sku_cols = [sku_rename.get(c, c) for c in sku_prefixed_cols]  # renamed SKU columns
attrs_present = [c for c in attrs if c in res.columns]

disp = res.copy()
disp.insert(0, "Rank", np.arange(1, len(disp) + 1))
# pretty % and $ formats
disp["Conversion %"] = disp["conv_rate"].map(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")
if "rpv" in disp.columns:
    disp["Revenue / Visitor"] = disp["rpv"].map(lambda x: f"${x:,.2f}" if pd.notnull(x) else "")

# integers with no .0 for Purchases and SKU counts
for col in ["Visitors", "Purchases", "Depth"]:
    if col in disp.columns:
        disp[col] = disp[col].fillna(0).astype(int)
for col in all_sku_cols:
    if col in disp.columns:
        disp[col] = disp[col].fillna(0).astype(int)

# ensure Unknown is displayed consistently
for col in attrs_present:
    if col in disp.columns:
        disp[col] = (
            disp[col]
            .fillna("Unknown")
            .replace({"": "Unknown", "None": "Unknown"})
        )

# ---- Column order:
# Rank, Visitors, Purchases, Conversion %, Depth, Gender, Age Range, Homeowner, Married,
# Children, Income Range, Net Worth, Credit Rating, then SKUs (far right)
desired_attr_labels = [
    "Gender", "Age Range", "Homeowner", "Married", "Children",
    "Income Range", "Net Worth", "Credit Rating"
]
# translate labels -> actual column names found in data
desired_attr_cols = [seg_map[lbl] for lbl in desired_attr_labels if lbl in seg_map and seg_map[lbl] in disp.columns]

left_cols = ["Rank", "Visitors", "Purchases", "Conversion %", "Depth"]
middle_cols = desired_attr_cols
right_cols = [c for c in all_sku_cols if c in disp.columns]

table_cols = left_cols + middle_cols + right_cols

def highlight_conv(s):
    return ["font-weight: bold" if s.name == "Conversion %" else "" for _ in s]

st.dataframe(disp[table_cols].style.apply(highlight_conv, axis=0), use_container_width=True, hide_index=True)

# ---------- Download CSV ----------
csv_out = res.copy()
csv_out.insert(0, "Rank", np.arange(1, len(csv_out) + 1))

# keep numeric columns numeric in CSV; rename columns for clarity
csv_cols = ["Rank", "Visitors", "Purchases", "conv_rate", "Depth"] + desired_attr_cols + right_cols
csv_out = csv_out[csv_cols].rename(columns={
    "conv_rate": "Conversion % (0-100)"
})
st.download_button(
    "Download ranked combinations (CSV)",
    data=csv_out.to_csv(index=False).encode("utf-8"),
    file_name="ranked_combinations_duckdb.csv",
    mime="text/csv"
)
