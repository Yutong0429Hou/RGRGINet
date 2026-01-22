import pandas as pd
import matplotlib.pyplot as plt

csv_path = "/root/autodl-tmp/renlian/results/inpaint/reliability_stats.csv"
df = pd.read_csv(csv_path)

print(df.head())
import numpy as np
import matplotlib.pyplot as plt

def binned_mean_std(x, y, bins):
    x = np.asarray(x); y = np.asarray(y)
    idx = np.digitize(x, bins) - 1
    xs, ym, ys = [], [], []
    for b in range(len(bins) - 1):
        m = idx == b
        if m.sum() == 0:
            continue
        xs.append(0.5 * (bins[b] + bins[b+1]))
        ym.append(y[m].mean())
        ys.append(y[m].std())
    return np.array(xs), np.array(ym), np.array(ys)

# 分箱：你现在 sim 很集中，建议用“分位数bins”
def quantile_bins(x, qs=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)):
    x = np.asarray(x)
    b = [np.quantile(x, q) for q in qs]
    b[-1] = max(b[-1], 1.0) + 1e-3
    # 去重
    out = [b[0]]
    for v in b[1:]:
        out.append(max(v, out[-1] + 1e-3))
    return np.array(out)

# matched
m_bins = quantile_bins(df["sim"].dropna().values)
mx, my, ms = binned_mean_std(df["sim"].dropna().values, df["gate"].dropna().values, m_bins)

# mismatch
mm_bins = quantile_bins(df["sim_mis"].dropna().values)
mmx, mmy, mms = binned_mean_std(df["sim_mis"].dropna().values, df["gate_mis"].dropna().values, mm_bins)

plt.figure(figsize=(5,4))
plt.plot(mx, my, marker="o", label="Matched")
plt.fill_between(mx, my-ms, my+ms, alpha=0.2)

plt.plot(mmx, mmy, marker="o", label="Mismatch")
plt.fill_between(mmx, mmy-mms, mmy+mms, alpha=0.2)

plt.xlabel("Reference Similarity (sim)")
plt.ylabel("Gating Value")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig("fig_reliability_curve.png", dpi=300)
plt.show()


plt.scatter(df["sim"], df["gate"], s=8, label="Matched")
plt.scatter(df["sim_mis"], df["gate_mis"], s=8, label="Mismatch", alpha=0.6)

plt.xlabel("Reference Similarity (sim)")
plt.ylabel("Gating Value")
plt.legend()
plt.show()
