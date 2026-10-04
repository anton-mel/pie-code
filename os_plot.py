"""Draws agent-os.png: how much sooner pie finishes agent workloads that use OS-style process calls.

    uv run --with matplotlib python os_plot.py
"""

from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE = "#2a78d6"

# Measured on an M5 Max (48 GB), Qwen3.8-27B 4-bit: (workload, pie seconds, other seconds).
VS_LLAMA = [
    ("Search: 4 branches forked,\nfirst YES wins (time to answer)", 0.66, 57.4),
    ("Team of 8: forked specialists,\npiped reports (total)", 61.0, 1334.0),
    ("Army of 8 agents, prompts\nshared by content (total)", 46.7, 153.7),
    ("Team of 4 (total)", 49.0, 144.7),
    ("Army of 4 (total)", 24.7, 68.1),
    ("Undo: 4 agents roll back\nevery failed step (total)", 32.2, 85.1),
]
VS_PLAIN = [
    ("Search, 4 branches", 0.66, 53.9),
    ("Undo, 4 agents", 32.2, 242.1),
    ("Army of 8", 46.7, 124.1),
    ("Team of 4", 49.0, 97.9),
]


def panel(axes, rows, title):
    axes.set_facecolor(SURFACE)
    names = [name for name, _, _ in rows][::-1]
    gains = [other / pie for _, pie, other in rows][::-1]
    bars = axes.barh(range(len(rows)), gains, color=BLUE, height=0.6)
    axes.set_xscale("log")
    axes.set_yticks(range(len(rows)))
    axes.set_yticklabels(names, color=INK, fontsize=10)
    for bar, gain, (_, pie, other) in zip(bars, gains, rows[::-1]):
        axes.annotate(f"{gain:.0f}x  ({pie:g} s vs {other:g} s)" if gain >= 10 else f"{gain:.1f}x  ({pie:g} s vs {other:g} s)",
                      (gain, bar.get_y() + bar.get_height() / 2), xytext=(6, 0), textcoords="offset points",
                      va="center", color=INK, fontsize=10)
    axes.set_xlim(1, max(gains) * 12)
    axes.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:g}x"))
    axes.grid(axis="x", color=GRID, linewidth=1)
    axes.set_axisbelow(True)
    axes.tick_params(colors=MUTED, length=0)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(GRID)
    axes.set_title(title, loc="left", fontsize=12, color=MUTED)


plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 11, "text.color": INK})
figure, (left, right) = plt.subplots(1, 2, figsize=(15, 6.2), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.35, 1]})
panel(left, VS_LLAMA, "pie as an agent OS vs llama.cpp tuned (a slot per agent)")
panel(right, VS_PLAIN, "pie as an agent OS vs pie used as a plain API")
figure.suptitle("Agents as processes: fork, pipes, shared memory, kill and rollback inside the engine", x=0.02, ha="left", fontsize=14, color=INK)
figure.text(0.02, 0.02, "M5 Max 48 GB, Qwen3.8-27B 4-bit, 6,000-8,000-token contexts, one run each. Answer accuracy is the same on every engine.",
            color=MUTED, fontsize=9)
figure.subplots_adjust(left=0.17, right=0.98, top=0.86, bottom=0.1, wspace=0.55)
figure.savefig(HERE / "agent-os.png", dpi=160, facecolor=SURFACE)
print(HERE / "agent-os.png")
