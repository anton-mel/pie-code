"""Per-task wall time, steps and exit status from mini-swe-agent trajectories."""
import json, sys, glob, statistics
for run in sys.argv[1:]:
    rows = {}
    for f in glob.glob(f"runs/{run}/*/*.traj.json"):
        t = json.load(open(f))
        stamps = [m.get("extra", {}).get("timestamp") for m in t.get("messages", []) if m.get("extra", {}).get("timestamp")]
        steps = sum(m.get("role") == "assistant" for m in t.get("messages", []))
        info = t.get("info", {})
        rows[t.get("instance_id", f)] = (round(stamps[-1] - stamps[0]) if len(stamps) > 1 else 0, steps, info.get("exit_status"), len(info.get("submission") or ""))
    secs = [r[0] for r in rows.values()]
    print(run, "tasks", len(rows), "median task s", statistics.median(secs) if secs else None, "sum s", sum(secs),
          "submitted", sum(r[2] == "Submitted" for r in rows.values()), "median steps", statistics.median([r[1] for r in rows.values()]) if rows else None)
    for k, v in sorted(rows.items()):
        print("  ", k[:34].ljust(34), v)
