"""Peak memory of an engine while a benchmark runs. Uses macOS `footprint`, which counts GPU memory.

    python3 mem.py release/pie -- uv run --project pie/python/client python bench.py --agents 12
"""

import re
import subprocess
import sys
import threading


def footprint_mb(pattern: str) -> float:
    """Summed footprint of every process whose command line matches `pattern`."""
    pids = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True).stdout.split()
    total = 0.0
    for pid in pids:
        out = subprocess.run(["footprint", "-p", pid, "--noCategories"], capture_output=True, text=True).stdout
        found = re.search(r"Footprint: ([\d.]+) (KB|MB|GB)", out)
        if found:
            total += float(found.group(1)) * {"KB": 1 / 1024, "MB": 1, "GB": 1024}[found.group(2)]
    return total


def system_mb() -> float:
    """Memory the whole Mac cannot give back: wired, anonymous and compressed pages."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(re.search(r"page size of (\d+)", out).group(1))
    kinds = ("Pages wired down", "Anonymous pages", "Pages occupied by compressor")
    return sum(int(re.search(rf"{kind}:\s+(\d+)", out).group(1)) for kind in kinds) * page / 2**20


class Peak:
    """Samples every `interval` seconds on a thread. `mb` is the highest footprint of
    `pattern`; `system` is the most the whole Mac grew over its level at the start."""

    def __init__(self, pattern: str, interval: float = 1.0):
        self.pattern, self.interval, self.mb = pattern, interval, 0.0
        self.base, self.system = system_mb(), 0.0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop.wait(self.interval):
            self.mb = max(self.mb, footprint_mb(self.pattern))
            self.system = max(self.system, system_mb() - self.base)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()


if __name__ == "__main__":
    pattern, command = sys.argv[1], sys.argv[sys.argv.index("--") + 1 :]
    with Peak(pattern) as peak:
        code = subprocess.run(command).returncode
    print(f'{{"peak_footprint_gb": {peak.mb / 1024:.2f}, "system_growth_gb": {peak.system / 1024:.2f}}}')
    sys.exit(code)
