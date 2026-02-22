import pandas as pd

train_path = "train_set.csv"
train = pd.read_csv(train_path)

METRICS = ["impressions","clicks","CTR","CPC","ad_spend","conversions","CPA","revenue","ROAS"]
for m in METRICS:
    if m not in train.columns:
        METRICS.remove(m)

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for col in METRICS:
        s = df[col].dropna()
        out[col] = {
            "mean": s.mean(),
            "median": s.median(),
            "p25": s.quantile(0.25),
            "p75": s.quantile(0.75),
        }
    return pd.DataFrame(out).T.reset_index().rename(columns={"index":"metric"})

# Overall benchmarks
overall = summarize(train)
overall.to_csv("train_benchmarks_overall.csv", index=False)

# By platform benchmarks
by_platform = (
    train.groupby("platform", dropna=False)
         .apply(summarize)
         .reset_index(drop=True)
)
# add platform label back
platforms = train["platform"].dropna().unique()
rows = []
for p in platforms:
    rows.append(summarize(train[train["platform"] == p]).assign(platform=p))
by_platform = pd.concat(rows, ignore_index=True)
by_platform.to_csv("train_benchmarks_by_platform.csv", index=False)

# Optional: top segments
SEGMENT_KEYS = [c for c in ["platform","country","campaign_type","industry"] if c in train.columns]
if SEGMENT_KEYS:
    seg = (train.groupby(SEGMENT_KEYS, dropna=False)
                .agg(
                    rows=("ROAS","size") if "ROAS" in train.columns else ("ad_spend","size"),
                    ad_spend=("ad_spend","sum") if "ad_spend" in train.columns else ("impressions","sum"),
                    revenue=("revenue","sum") if "revenue" in train.columns else ("clicks","sum"),
                    conversions=("conversions","sum") if "conversions" in train.columns else ("clicks","sum"),
                    ROAS=("ROAS","mean") if "ROAS" in train.columns else ("CPC","mean"),
                    CPA=("CPA","mean") if "CPA" in train.columns else ("CTR","mean"),
                )
                .reset_index()
          )
    seg = seg.sort_values(by="ROAS", ascending=False).head(50)
    seg.to_csv("train_top_segments.csv", index=False)