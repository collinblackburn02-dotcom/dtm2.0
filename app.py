import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations
from utils import resolve_col

st.set_page_config(page_title="Heavenly Health — Customer Insights", layout="wide")
st.title("✨ Heavenly Health — Customer Insights")
st.caption("Fast, ranked customer segments (Pandas-only, robust and simple).")

# --- Attribute cards: alignment + compact fixed heights ---
st.markdown("""
<style>
  .attr-card   { display:flex; flex-direction:column; gap:4px; min-height:110px; margin-bottom:12px; }
  .attr-header { display:flex; align-items:center; justify-content:space-between; }
  .attr-title  { font-weight:700; font-size:1.08rem; margin:0; }
  /* Align Streamlit checkbox inline with the title */
  div[data-testid="stCheckbox"] { margin:0; display:flex; align-items:center; justify-content:flex-end; }
  /* Reserve just enough space for one multiselect row so all cards line up */
  .attr-body   { min-height:44px; display:flex; align-items:center; }
</style>
""", unsafe_allow_html=True)





# ---------- Sidebar ----------
with st.sidebar:
    uploaded = st.file_uploader("Upload merged CSV", type=["csv"])
    st.markdown("---")
    metric_choice = st.radio(
        "Sort metric",
        ["Conversion %", "Purchases", "Visitors", "Revenue / Visitor"],
        index=0
    )
    # Max combo depth: up to 5, default 5
    max_depth = st.slider("Max combo depth", 1, 5, 5, 1)
    top_n = st.slider("Top N", 10, 1000, 50, 10)
    # Min visitors: default 500, min allowed 50
    min_rows = st.number_input("Minimum Visitors per group", min_value=50, value=500, step=50)

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

# Core numeric columns
df["_PURCHASE"] = pd.to_numeric(df[purchase_col], errors="coerce").fillna(0).astype(int)
df["_DATE"] = to_datetime_series(df[date_col]) if date_col else pd.NaT
df["_REVENUE"] = pd.to_numeric(df[revenue_col], errors="coerce").fillna(0.0) if revenue_col else 0.0

# ---------- Segmentation Attributes ----------
seg_map = {
    "Age Range":     resolve_col(df, "Age_Range"),
    "Income Range":  resolve_col(df, "Income_Range"),
    "Net Worth":     resolve_col(df, "Net_Worth"),
    "Credit Rating": resolve_col(df, "Credit_Rating"),
    "Gender":        resolve_col(df, "Gender"),
    "Homeowner":     resolve_col(df, "Home_Owner"),
    "Married":       resolve_col(df, "Married"),
    "Children":      resolve_col(df, "Children"),
}
# keep only found columns
seg_map = {label: col for label, col in seg_map.items() if col is not None}
seg_cols = list(seg_map.values())

# ---------- Filters + Include toggles ----------
with st.expander("🔎 Filters", expanded=True):
    dff = df.copy()

    # Treat 'U' as missing for Gender and Credit before collapsing
    for label, col in seg_map.items():
        if label in ("Gender", "Credit Rating") and col in dff.columns:
            mask_u = dff[col].astype(str).str.upper().str.strip().eq("U")
            dff.loc[mask_u, col] = pd.NA

    # Date filter (only if DATE exists)
    if not pd.isna(dff["_DATE"]).all():
        mind, maxd = pd.to_datetime(dff["_DATE"].dropna().min()), pd.to_datetime(dff["_DATE"].dropna().max())
        c1, c2 = st.columns(2)
        with c1:
            start_end = st.date_input("Date range", (mind.date(), maxd.date()))
        with c2:
            include_undated = st.checkbox("Include no-date rows", value=True)
        if isinstance(start_end, (list, tuple)) and len(start_end) == 2:
            start, end = start_end
            mask = dff["_DATE"].between(pd.to_datetime(start), pd.to_datetime(end))
            if include_undated:
                mask = mask | dff["_DATE"].isna()
            dff = dff[mask]

    # SKU contains filter
    sku_search = st.text_input("Most Recent SKU contains (optional)")
    if msku_col and sku_search:
        dff = dff[dff[msku_col].astype(str).str.contains(sku_search, case=False, na=False)]

# Attribute value filters + Include checkboxes (all included by default)
selections = {}
include_flags = {}  # label -> bool
if seg_cols:
    st.markdown("**Attributes**")
    cols = st.columns(3, gap="small")
    idx = 0
    for label, col in seg_map.items():
        with cols[idx % 3]:
            # Card wrapper (fixed height)
            st.markdown('<div class="attr-card">', unsafe_allow_html=True)

            # Header row: title (left) + Include (right), perfectly aligned
            left, right = st.columns([0.62, 0.38])
            with left:
                st.markdown(f'<div class="attr-title">{label}</div>', unsafe_allow_html=True)
            with right:
                include_flags[label] = st.checkbox("Include", value=True, key=f"include_{label}")

            # Body area: dropdown when included, otherwise a spacer of same height
            st.markdown('<div class="attr-body">', unsafe_allow_html=True)
            if include_flags[label]:
                values = sorted([x for x in dff[col].dropna().unique().tolist() if str(x).strip()])
                sel = st.multiselect(
                    "",
                    options=values,
                    default=[],
                    placeholder="Choose options",
                    label_visibility="collapsed",
                    key=f"filter_{label}"
                )
                if sel:
                    selections[col] = sel
            else:
                # keep previous choices cleared when unchecked
                if f"filter_{label}" in st.session_state:
                    st.session_state[f"filter_{label}"] = []
                # spacer keeps card height consistent
                st.markdown("&nbsp;", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)  # close .attr-body

            st.markdown('</div>', unsafe_allow_html=True)  # close .attr-card
        idx += 1

    # Apply value filters (only on attributes that are included)
    for label, col in seg_map.items():
        if include_flags.get(label):
            vals = st.session_state.get(f"filter_{label}", [])
            if vals:
                dff = dff[dff[col].isin(vals)]



# Build the list of attribute columns to group by
#  - Start from "Include" toggles
group_attr_cols = [seg_map[lbl] for lbl, inc in include_flags.items() if inc]
#  - Any attribute with a value filter becomes REQUIRED in all combos
required_raw_cols = list(selections.keys())
group_attr_cols = sorted(set(group_attr_cols).union(required_raw_cols))

st.caption(f"Rows after filters: **{len(dff):,}** / {len(df):,}")

# Collapse NaN/empty/"None"/"U" → "Unknown" AFTER filters (so groupings are stable)
for col in seg_cols:
    if col in dff.columns:
        dff[col] = (
            dff[col]
            .astype("string")
            .fillna("Unknown")
            .replace({"": "Unknown", "None": "Unknown", "U": "Unknown", "u": "Unknown"})
            .str.strip()
        )

# ---------- Top SKUs globally (among purchasers) ----------
top_skus = []
if msku_col and msku_col in dff.columns:
    top_skus = (
        dff.loc[dff["_PURCHASE"] == 1, msku_col]
        .astype(str).str.strip()
        .replace({"": np.nan})
        .dropna()
        .value_counts()
        .head(11)
        .index.tolist()
    )

# Precompute SKU indicator columns (per-row), so groupby can sum them quickly
sku_ind_cols = {}
if top_skus:
    for sku in top_skus:
        colname = f"__SKU_{hash(sku)%10**9}"  # collision-resistant temp name
        sku_ind_cols[sku] = colname
        dff[colname] = ((dff[msku_col].astype(str).str.strip() == sku) & (dff["_PURCHASE"] == 1)).astype(int)

# ---------- Build all grouping combinations (enforce required attributes) ----------
attrs_available = [c for c in group_attr_cols if c in dff.columns]
req_set = set(required_raw_cols)  # raw col names with value filters
combo_sets = []

if len(attrs_available) == 0:
    combo_sets = [()]  # grand total
else:
    # Depth must be at least the number of required attrs
    min_depth = max(1, len(req_set))
    max_d = max(max_depth, len(req_set))
    for d in range(min_depth, max_d + 1):
        for s in combinations(attrs_available, d):
            if req_set.issubset(s):
                combo_sets.append(list(s))
    if not combo_sets:
        # Fallback: just require-set itself if possible
        if req_set.issubset(set(attrs_available)):
            combo_sets = [list(req_set)]
        else:
            combo_sets = [()]  # extreme fallback

# ---------- Aggregate for each combination ----------
rows = []
for combo in combo_sets:
    combo = list(combo)
    if combo:
        # size (Visitors)
        size_df = dff.groupby(combo, dropna=False).size().rename("Visitors").reset_index()
        # numeric sums
        agg_dict = {"_PURCHASE": "sum"}
        if revenue_col:
            agg_dict["_REVENUE"] = "sum"
        for sku, ind in sku_ind_cols.items():
            agg_dict[ind] = "sum"
        sums_df = dff.groupby(combo, dropna=False).agg(agg_dict).reset_index()
        g = size_df.merge(sums_df, on=combo, how="left")
    else:
        # grand total
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
    g["Depth"] = len(combo)

    # keep columns in stable order: attributes → metrics → SKUs
    cols_order = combo + ["Visitors", "Purchases", "conv_rate", "Depth", "rpv", "revenue"] + list(sku_ind_cols.values())
    g = g[cols_order]
    rows.append(g)

# Concatenate all combinations
res = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["Visitors","Purchases","conv_rate","Depth"])

# Apply min_rows AFTER aggregation
res = res[res["Visitors"] >= int(min_rows)].copy()

# Replace temp SKU indicator names with actual SKU names
sku_cols = []
if sku_ind_cols:
    rename_map = {v: k for k, v in sku_ind_cols.items()}
    res = res.rename(columns=rename_map)
    sku_cols = list(rename_map.values())

# ---- Sort & limit
sort_key_map = {
    "Conversion %": "conv_rate",
    "Purchases": "Purchases",
    "Visitors": "Visitors",
    "Revenue / Visitor": "rpv",
}
sort_key = sort_key_map[metric_choice]
res = res.sort_values(sort_key, ascending=False).head(top_n).reset_index(drop=True)

# ---------- Prepare display
# Friendly attribute headers -> display names (without changing internal processing)
friendly_attr = {v: k for k, v in seg_map.items()}

disp = res.copy()
# Convert counts to ints (no .0)
for c in ["Visitors", "Purchases", "Depth"]:
    if c in disp.columns:
        disp[c] = pd.to_numeric(disp[c], errors="coerce").fillna(0).astype(int)
for c in sku_cols:
    if c in disp.columns:
        disp[c] = pd.to_numeric(disp[c], errors="coerce").fillna(0).astype(int)
        disp[c] = disp[c].replace({0: ""})  # blank out 0s for SKUs

# Pretty formats
disp["Conversion %"] = disp["conv_rate"].map(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")
if "rpv" in disp.columns:
    disp["Revenue / Visitor"] = disp["rpv"].map(lambda x: f"${x:,.2f}" if pd.notnull(x) else "")

# Rename attribute columns to friendly labels for display
disp = disp.rename(columns=friendly_attr)

# DISPLAY-ONLY: replace None/NaN and the literal "None" with ""
attr_display_cols = [
    lbl for lbl in
    ["Gender", "Age Range", "Homeowner", "Married", "Children", "Income Range", "Net Worth", "Credit Rating"]
    if lbl in disp.columns
]
for c in attr_display_cols:
    s = disp[c].astype(object)
    mask = s.isna() | s.astype(str).str.strip().str.lower().eq("none")
    disp[c] = s.mask(mask, "")

# Insert Rank
disp.insert(0, "Rank", np.arange(1, len(disp) + 1))

# ---- Column order (Purchasers + RPV in the table; SKUs on the right)
desired_attr_labels = [
    "Gender", "Age Range", "Homeowner", "Married", "Children",
    "Income Range", "Net Worth", "Credit Rating"
]
middle_cols = [lbl for lbl in desired_attr_labels if lbl in disp.columns]
right_cols = [c for c in sku_cols if c in disp.columns]  # SKUs to the right

# Add Purchasers (display alias)
disp["Purchasers"] = disp["Purchases"]
left_cols = ["Rank", "Visitors", "Purchasers", "Conversion %", "Revenue / Visitor", "Depth"]

# Some groups may lack attributes; keep any extra attribute cols at the end of middle section
extra_attrs = [c for c in friendly_attr.values() if c in disp.columns and c not in middle_cols]

table_cols = [c for c in left_cols + middle_cols + extra_attrs + right_cols if c in disp.columns]

# Bold the column that matches the selected sort metric
display_metric_map = {
    "Conversion %": "Conversion %",
    "Purchases": "Purchasers",
    "Visitors": "Visitors",
    "Revenue / Visitor": "Revenue / Visitor",
}
selected_display_metric = display_metric_map.get(metric_choice, "Conversion %")

def highlight_selected_metric(s):
    return ["font-weight: bold" if s.name == selected_display_metric else "" for _ in s]

st.dataframe(
    disp[table_cols].style.apply(highlight_selected_metric, axis=0),
    use_container_width=True,
    hide_index=True
)

# ---------- Download CSV ----------
csv_out = res.copy()
csv_out.insert(0, "Rank", np.arange(1, len(csv_out) + 1))

# Order CSV: include Purchases and numeric metrics; use friendly names for attributes
csv_out = csv_out.rename(columns=friendly_attr)

csv_attr_cols = [lbl for lbl in desired_attr_labels if lbl in csv_out.columns]
extra_csv_attrs = [c for c in friendly_attr.values() if c in csv_out.columns and c not in csv_attr_cols]
csv_sku_cols = [c for c in sku_cols if c in csv_out.columns]

csv_cols = ["Rank", "Visitors", "Purchases", "conv_rate", "Depth", "rpv", "revenue"] + csv_attr_cols + extra_csv_attrs + csv_sku_cols
csv_cols = [c for c in csv_cols if c in csv_out.columns]
csv_out = csv_out[csv_cols].rename(columns={
    "conv_rate": "Conversion % (0-100)",
    "rpv": "Revenue / Visitor",
    "revenue": "Revenue",
})

# Cast integer-like columns to Int64 to avoid .0 in CSV
int_like = ["Rank", "Visitors", "Purchases", "Depth"] + csv_sku_cols
for c in int_like:
    if c in csv_out.columns:
        csv_out[c] = pd.to_numeric(csv_out[c], errors="coerce").fillna(0).astype("Int64")

st.download_button(
    "Download ranked combinations (CSV)",
    data=csv_out.to_csv(index=False).encode("utf-8"),
    file_name="ranked_combinations.csv",
    mime="text/csv"
)
