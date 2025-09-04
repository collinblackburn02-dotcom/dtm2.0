import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations
from utils import resolve_col

# -------------------- Page & Brand --------------------
st.set_page_config(page_title="Heavenly Health — Customer Insights", layout="wide")

# Assets (RAW GitHub URLs)
DATA_URL = "https://raw.githubusercontent.com/collinblackburn02-dotcom/dtm2.0/main/Copy%20of%20DAN_HHS%20-%20Copy%20of%20MergedForDashboard.csv"
LOGO_URL = "https://raw.githubusercontent.com/collinblackburn02-dotcom/dtm2.0/main/Heavenly%20Health%20Logo.jpg"

# Feature flags
SHOW_GLOBAL_FILTERS = False     # hidden for now
EXCLUDE_STATE = True            # hide/ignore State everywhere

# Start these attributes UNCHECKED on first load
DEFAULT_UNCHECKED_ATTRS = {"Income Range", "Net Worth", "Credit Rating"}

# Header
st.title("Heavenly Health — Customer Insights")
st.caption("Fast, ranked customer segments.")

# --- Brand CSS ---
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Playfair+Display:wght@700&display=swap');

  :root{
    --ink:#2A2A2A;
    --bg:#F7F3EE;           /* warm cream */
    --panel:#FFFFFF;
    --accent:#6A3E2E;       /* cocoa */
    --accent2:#C67A57;      /* copper */
    --muted:#8D8379;
  }
  html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > .main {
    background: var(--bg) !important;
    color: var(--ink) !important;
    font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif !important;
  }
  h1,h2,h3 { font-family: 'Playfair Display', serif !important; color: var(--accent) !important; letter-spacing:.2px; }
  [data-testid="stSidebar"] { background: var(--panel) !important; border-right: 1px solid #ece3da !important; }

  /* Controls tint */
  .stButton > button { background: var(--accent) !important; color: #fff !important; border:0 !important; }
  .stButton > button:hover { background:#5a3326 !important; }
  .stRadio [role="radio"][aria-checked="true"] { color: var(--accent) !important; }
  .stMultiSelect [data-baseweb="tag"] { background: #f2e7df !important; }
  .stMultiSelect [data-baseweb="select"] > div { border-color: #e8ddd3 !important; }

  /* Attribute cards */
  .attr-card   { display:flex; flex-direction:column; gap:1px; min-height:0px; margin-bottom:0px; }
  .attr-header { display:flex; align-items:center; justify-content:space-between; }
  .attr-title  { font-weight:700; font-size:1.08rem; margin:0; }
  div[data-testid="stCheckbox"] { margin:0; display:flex; align-items:center; justify-content:flex-end; }
  .attr-body   { min-height:0px; display:flex; align-items:center; }

  /* Center st.dataframe headers + cells */
  [data-testid="stDataFrame"] th, [data-testid="stDataFrame"] td {
    text-align:center !important; vertical-align:middle !important;
  }
</style>
""", unsafe_allow_html=True)

# -------------------- Sidebar --------------------
with st.sidebar:
  st.image(LOGO_URL, use_container_width=True)
  st.markdown("---")
  metric_choice = st.radio(
      "Sort metric", ["Conversion %", "Purchases", "Visitors", "Revenue / Visitor"], index=0
  )
  max_depth = st.slider("Max combo depth", 1, 4, 4, 1)
  top_n = st.slider("Top N", 10, 1000, 50, 10)
  min_rows = st.number_input("Minimum Visitors per group", min_value=50, value=1000, step=50)

# -------------------- Data --------------------
@st.cache_data(show_spinner=False)
def load_df(path_or_url: str) -> pd.DataFrame:
  df = pd.read_csv(path_or_url)
  df.columns = [str(c).strip() for c in df.columns]
  return df

def to_datetime_series(s: pd.Series) -> pd.Series:
  try:
    return pd.to_datetime(s, errors="coerce")
  except Exception:
    return pd.to_datetime(pd.Series([None] * len(s)))

df = load_df(DATA_URL)

# Resolve key columns
purchase_col = resolve_col(df, "Purchase")
date_col     = resolve_col(df, "DATE")
msku_col     = resolve_col(df, "SKU")
revenue_col  = resolve_col(df, "Revenue")

# State (we'll ignore if EXCLUDE_STATE=True)
state_col = (
  resolve_col(df, "State")
  or resolve_col(df, "US_STATE")
  or resolve_col(df, "STATE_CODE")
  or resolve_col(df, "STATE_ABBR")
  or resolve_col(df, "Personal_State")
)

if purchase_col is None:
  st.error("Missing Purchase column."); st.stop()

# Core numeric columns
df["_PURCHASE"] = pd.to_numeric(df[purchase_col], errors="coerce").fillna(0).astype(int)
df["_DATE"] = to_datetime_series(df[date_col]) if date_col else pd.NaT
df["_REVENUE"] = pd.to_numeric(df[revenue_col], errors="coerce").fillna(0.0) if revenue_col else 0.0

# -------------------- Segmentation attributes --------------------
seg_map = {
  "Age Range":     resolve_col(df, "Age_Range"),
  "Income Range":  resolve_col(df, "Income_Range"),
  "Net Worth":     resolve_col(df, "Net_Worth"),
  "Credit Rating": resolve_col(df, "Credit_Rating"),
  "Gender":        resolve_col(df, "Gender"),
  "Homeowner":     resolve_col(df, "Home_Owner"),
  "Married":       resolve_col(df, "Married"),
  "Children":      resolve_col(df, "Children"),
  "State":         state_col,
}
# keep only found columns
seg_map = {label: col for label, col in seg_map.items() if col is not None}
seg_cols = list(seg_map.values())

# Drop State everywhere (flag)
if EXCLUDE_STATE and "State" in seg_map:
  del seg_map["State"]
  state_col = None
  seg_cols = list(seg_map.values())

# --- One-time init so those three attrs start unchecked and empty ---
if "init_unchecked_v1" not in st.session_state:
    st.session_state["init_unchecked_v1"] = True
    for lbl in DEFAULT_UNCHECKED_ATTRS:
        st.session_state[f"include_{lbl}"] = False   # Include unchecked
        st.session_state[f"filter_{lbl}"] = []       # No preselected values

# -------------------- Filters (global hidden) --------------------
dff = df.copy()
if SHOW_GLOBAL_FILTERS:
  with st.expander("🔎 Filters", expanded=False):
    # Date and SKU filters would go here (kept for future)
    pass

# -------------------- Attribute UI --------------------
selections = {}
include_flags = {}
if seg_cols:
  st.markdown("**Attributes**")
  cols = st.columns(3, gap="small")
  idx = 0
  for label, col in seg_map.items():
    with cols[idx % 3]:
      st.markdown('<div class="attr-card">', unsafe_allow_html=True)

      # header row
      left, right = st.columns([0.62, 0.38])
      with left:
        st.markdown(f'<div class="attr-title">{label}</div>', unsafe_allow_html=True)
      with right:
        default_include = False if label in DEFAULT_UNCHECKED_ATTRS else True
        include_flags[label] = st.checkbox("Include", value=default_include, key=f"include_{label}")

      # body
      st.markdown('<div class="attr-body">', unsafe_allow_html=True)
      if include_flags[label]:
        values = sorted([x for x in dff[col].dropna().unique().tolist() if str(x).strip()])
        sel = st.multiselect(
          "", options=values, default=[], placeholder="Choose options",
          label_visibility="collapsed", key=f"filter_{label}"
        )
        if sel:
          selections[col] = sel
      else:
        if f"filter_{label}" in st.session_state:
          st.session_state[f"filter_{label}"] = []
        st.markdown("&nbsp;", unsafe_allow_html=True)
      st.markdown('</div>', unsafe_allow_html=True)   # .attr-body
      st.markdown('</div>', unsafe_allow_html=True)   # .attr-card
    idx += 1

  # Apply value filters (only if the attribute is included)
  for label, col in seg_map.items():
    if include_flags.get(label):
      vals = st.session_state.get(f"filter_{label}", [])
      if vals:
        dff = dff[dff[col].isin(vals)]

# attributes to group by
group_attr_cols = [seg_map[lbl] for lbl, inc in include_flags.items() if inc]
required_raw_cols = list(selections.keys())
group_attr_cols = sorted(set(group_attr_cols).union(required_raw_cols))

st.caption(f"Rows after filters: **{len(dff):,}** / {len(df):,}")

# Normalize AFTER filters; collapse to 'Unknown' for stability; uppercase state if present
for col in seg_cols:
  if col in dff.columns:
    s = dff[col].astype("string").str.strip()
    if state_col and col == state_col:
      s = s.str.upper()
    dff[col] = s.fillna("Unknown").replace({"": "Unknown", "None": "Unknown", "U": "Unknown", "u": "Unknown"})

# -------------------- Top SKUs --------------------
top_skus = []
if msku_col and msku_col in dff.columns:
  top_skus = (
    dff.loc[dff["_PURCHASE"] == 1, msku_col]
      .astype(str).str.strip().replace({"": np.nan}).dropna()
      .value_counts().head(11).index.tolist()
  )

sku_ind_cols = {}
if top_skus:
  for sku in top_skus:
    colname = f"__SKU_{hash(sku)%10**9}"
    sku_ind_cols[sku] = colname
    dff[colname] = ((dff[msku_col].astype(str).str.strip() == sku) & (dff["_PURCHASE"] == 1)).astype(int)

# -------------------- Build combinations --------------------
attrs_available = [c for c in group_attr_cols if c in dff.columns]
req_set = set(required_raw_cols)
combo_sets = []

if len(attrs_available) == 0:
  combo_sets = [()]  # grand total
else:
  min_depth = max(1, len(req_set))
  max_d = max(max_depth, len(req_set))
  for d in range(min_depth, max_d + 1):
    for s in combinations(attrs_available, d):
      if req_set.issubset(s):
        combo_sets.append(list(s))
  if not combo_sets:
    if req_set.issubset(set(attrs_available)):
      combo_sets = [list(req_set)]
    else:
      combo_sets = [()]

# Unknown/U/blank should not create their own groups
def _df_for_combo(src: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
  if not keys: return src
  dst = src.copy()
  for k in keys:
    if k in dst.columns:
      dst[k] = dst[k].replace({"Unknown": np.nan, "unknown": np.nan, "U": np.nan, "u": np.nan, "": np.nan})
  return dst

# -------------------- Aggregate --------------------
rows = []
for combo in combo_sets:
  keys = list(combo)
  if keys:
    dtmp = _df_for_combo(dff, keys)

    size_df = dtmp.groupby(keys).size().rename("Visitors").reset_index()
    agg_dict = {"_PURCHASE": "sum"}
    if revenue_col:
      agg_dict["_REVENUE"] = "sum"
    for _, ind in sku_ind_cols.items():
      agg_dict[ind] = "sum"

    sums_df = dtmp.groupby(keys).agg(agg_dict).reset_index()
    g = size_df.merge(sums_df, on=keys, how="left")
  else:
    g = pd.DataFrame({
      "Visitors": [int(len(dff))],
      "_PURCHASE": [int(dff["_PURCHASE"].sum())],
      "_REVENUE": [float(dff["_REVENUE"].sum())] if revenue_col else [0.0],
      **{ind: [int(dff[ind].sum())] for ind in sku_ind_cols.values()}
    })

  # metrics
  g["Purchases"] = g["_PURCHASE"].astype(int)
  g["conv_rate"] = 100.0 * g["Purchases"] / g["Visitors"].replace(0, np.nan)
  if revenue_col:
    g["rpv"] = g["_REVENUE"] / g["Visitors"].replace(0, np.nan)
    g["revenue"] = g["_REVENUE"]
  else:
    g["rpv"] = 0.0
    g["revenue"] = 0.0
  g["Depth"] = len(keys)

  cols_order = keys + ["Visitors", "Purchases", "conv_rate", "Depth", "rpv", "revenue"] + list(sku_ind_cols.values())
  g = g[cols_order]
  rows.append(g)

res = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["Visitors","Purchases","conv_rate","Depth"])
res = res[res["Visitors"] >= int(min_rows)].copy()

# Replace temp SKU indicator names with actual SKU names
sku_cols = []
if sku_ind_cols:
  rename_map = {v: k for k, v in sku_ind_cols.items()}
  res = res.rename(columns=rename_map)
  sku_cols = list(rename_map.values())

# -------------------- Sort & limit --------------------
sort_key_map = {"Conversion %":"conv_rate","Purchases":"Purchases","Visitors":"Visitors","Revenue / Visitor":"rpv"}
sort_key = sort_key_map[metric_choice]
res = res.sort_values(sort_key, ascending=False).head(top_n).reset_index(drop=True)

# -------------------- Display table --------------------
friendly_attr = {v: k for k, v in seg_map.items()}
disp = res.copy()

# ints
for c in ["Visitors","Purchases","Depth"]:
  if c in disp.columns:
    disp[c] = pd.to_numeric(disp[c], errors="coerce").fillna(0).astype(int)
# SKUs: hide 0
for c in sku_cols:
  if c in disp.columns:
    disp[c] = pd.to_numeric(disp[c], errors="coerce").fillna(0).astype(int).replace({0:""})

# pretty formats
disp["Conversion %"] = disp["conv_rate"].map(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")
if "rpv" in disp.columns:
  disp["Revenue / Visitor"] = disp["rpv"].map(lambda x: f"${x:,.2f}" if pd.notnull(x) else "")

# attribute headers → friendly labels
disp = disp.rename(columns=friendly_attr)

# display-only: blank None/"Unknown"/"U"
attr_display_cols = [lbl for lbl in ["Gender","Age Range","Homeowner","Married","Children","Income Range","Net Worth","Credit Rating"] if lbl in disp.columns]
for c in attr_display_cols:
  s = disp[c].astype("string")
  mask = s.isna() | s.str.strip().str.lower().isin(["none","unknown","u"])
  disp[c] = s.mask(mask, "")

# rank
disp.insert(0, "Rank", np.arange(1, len(disp)+1))

desired_attr_labels = ["Gender","Age Range","Homeowner","Married","Children","Income Range","Net Worth","Credit Rating"]  # (no State)
middle_cols = [lbl for lbl in desired_attr_labels if lbl in disp.columns]
right_cols = [c for c in sku_cols if c in disp.columns]
disp["Purchasers"] = disp["Purchases"]
left_cols = ["Rank","Visitors","Purchasers","Conversion %","Revenue / Visitor","Depth"]
extra_attrs = [c for c in friendly_attr.values() if c in disp.columns and c not in middle_cols]
table_cols = [c for c in left_cols + middle_cols + extra_attrs + right_cols if c in disp.columns]

# dynamic bolding
display_metric_map = {"Conversion %":"Conversion %","Purchases":"Purchasers","Visitors":"Visitors","Revenue / Visitor":"Revenue / Visitor"}
selected_display_metric = display_metric_map.get(metric_choice,"Conversion %")

def highlight_selected_metric(s):
  return ["font-weight: bold" if s.name == selected_display_metric else "" for _ in s]

styled = disp[table_cols].style.apply(highlight_selected_metric, axis=0)
st.dataframe(styled, use_container_width=True, hide_index=True)

# -------------------- Download CSV --------------------
csv_out = res.copy()
csv_out.insert(0, "Rank", np.arange(1, len(csv_out)+1))
csv_out = csv_out.rename(columns=friendly_attr)

csv_attr_cols = [lbl for lbl in desired_attr_labels if lbl in csv_out.columns]
extra_csv_attrs = [c for c in friendly_attr.values() if c in csv_out.columns and c not in csv_attr_cols]
csv_sku_cols = [c for c in sku_cols if c in csv_out.columns]

csv_cols = ["Rank","Visitors","Purchases","conv_rate","Depth","rpv","revenue"] + csv_attr_cols + extra_csv_attrs + csv_sku_cols
csv_cols = [c for c in csv_cols if c in csv_out.columns]
csv_out = csv_out[csv_cols].rename(columns={"conv_rate":"Conversion % (0-100)","rpv":"Revenue / Visitor","revenue":"Revenue"})

int_like = ["Rank","Visitors","Purchases","Depth"] + csv_sku_cols
for c in int_like:
  if c in csv_out.columns:
    csv_out[c] = pd.to_numeric(csv_out[c], errors="coerce").fillna(0).astype("Int64")

st.download_button(
  "Download ranked combinations (CSV)",
  data=csv_out.to_csv(index=False).encode("utf-8"),
  file_name="ranked_combinations.csv",
  mime="text/csv"
)

# ---------- Static attribute conversion tables (from full dataset; not affected by selections) ----------

st.markdown("## Attribute Conversion Snapshots")

@st.cache_data(show_spinner=False)
def compute_attr_table(df_in: pd.DataFrame, col: str) -> pd.DataFrame:
    """Return a small table: value, visitors, purchasers, conv% for one attribute.
       Unknown/U/blank are removed from the output (but still counted overall)."""
    tmp = df_in[[col, "_PURCHASE"]].copy()
    # normalize labels
    s = tmp[col].astype("string").str.strip()
    s = s.fillna("Unknown").replace({"": "Unknown", "None": "Unknown", "U": "Unknown", "u": "Unknown"})
    tmp["_VAL"] = s

    # aggregate
    size_df = tmp.groupby("_VAL").size().rename("Visitors").reset_index()
    purch_df = tmp.groupby("_VAL")["_PURCHASE"].sum().rename("Purchasers").reset_index()
    g = size_df.merge(purch_df, on="_VAL", how="left")

    # drop Unknown-like rows from the display
    g = g[~g["_VAL"].str.lower().isin(["unknown", "u", "none", ""])]

    # conv%
    g["Conversion %"] = 100.0 * g["Purchasers"] / g["Visitors"].replace(0, np.nan)

    # tidy columns / sort
    g = g.rename(columns={"_VAL": "Value"})
    g = g[["Value", "Visitors", "Purchasers", "Conversion %"]].sort_values("Conversion %", ascending=False).reset_index(drop=True)

    # cast types for clean display / CSV
    g["Visitors"] = pd.to_numeric(g["Visitors"], errors="coerce").fillna(0).astype(int)
    g["Purchasers"] = pd.to_numeric(g["Purchasers"], errors="coerce").fillna(0).astype(int)
    return g

def _style_attr(df_small: pd.DataFrame):
    # bold Conversion %, center everything (global CSS may already center)
    def _bold_conv(s):
        return ["font-weight: bold" if s.name == "Conversion %" else "" for _ in s]
    return (
        df_small.style
        .format({"Conversion %": "{:.2f}%"})
        .apply(_bold_conv, axis=0)
    )

# decide order (reuse your desired_attr_labels; State was already excluded upstream)
_attr_order = [lbl for lbl in ["Gender", "Age Range", "Homeowner", "Married", "Children",
                               "Income Range", "Net Worth", "Credit Rating"]
               if lbl in seg_map]  # use seg_map so we only show attributes that exist

# lay out tables in a 2-column grid
cols = st.columns(2, gap="medium")
for i, label in enumerate(_attr_order):
    colname = seg_map[label]
    tbl = compute_attr_table(df, colname)  # uses full df (static)
    with cols[i % 2]:
        st.markdown(f"**{label}**")
        st.dataframe(_style_attr(tbl), use_container_width=True, hide_index=True)

