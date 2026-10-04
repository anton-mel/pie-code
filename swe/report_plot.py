"""swe.png: time for the SWE-bench batch, and issues solved, per engine."""
import matplotlib.pyplot as plt
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
rows = [("pie\n(mini-swe-agent)", 34, "#2a78d6", "12 / 16 solved"), ("pie, branch and prune\n(co-designed agent)", 67, "#86b6ef", "12 / 16 solved"),
        ("llama.cpp\n(mini-swe-agent)", 136, "#6b6b66", "12 / 16 solved, 3 unfinished")]
plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 11, "text.color": INK})
fig, ax = plt.subplots(figsize=(10, 4.2), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
bars = ax.barh(range(3), [r[1] for r in rows][::-1], color=[r[2] for r in rows][::-1], height=0.55)
ax.set_yticks(range(3)); ax.set_yticklabels([r[0] for r in rows][::-1])
for bar, r in zip(bars, rows[::-1]):
    label = f"{r[1]} min" + ("+" if r[0].startswith("llama") else "") + f"  ·  {r[3]}"
    ax.annotate(label, (bar.get_width(), bar.get_y() + bar.get_height() / 2), xytext=(6, 0), textcoords="offset points", va="center")
ax.set_xlim(0, 200); ax.grid(axis="x", color=GRID); ax.set_axisbelow(True); ax.tick_params(colors=MUTED, length=0)
for side in ("top", "right", "left"): ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(GRID); ax.set_xlabel("minutes to finish 16 SWE-bench Verified tasks, 8 agents at once", color=MUTED)
fig.suptitle("Same issues solved, pie at least 4x faster than llama.cpp (M5 Max 48 GB, Qwen3.8-27B 4-bit)", x=0.02, ha="left", fontsize=13)
fig.subplots_adjust(left=0.22, right=0.97, top=0.85, bottom=0.17)
fig.savefig("swe.png", dpi=160, facecolor=SURFACE)
