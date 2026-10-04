WORDS = "amber birch cedar delta ember fjord grove heron iris jade kelp lotus".split()


def notebook(agent: int, tokens: int) -> str:  # bench.notebook, copied: MLX runs in its own environment
    lines = [f"Notebook of agent {agent}. Each item has a code word."]
    for item in range(tokens // 14):
        lines.append(f"Item {item}: the code word is {WORDS[(item * 7 + agent * 5) % len(WORDS)]}{agent}.")
    return "\n".join(lines)
