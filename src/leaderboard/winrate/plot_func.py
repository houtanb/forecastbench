import matplotlib.pyplot as plt
import numpy as np


# Define the function
def f(a, b):
    return 1 / (1 + np.exp(b - a))


# Sample data from your examples
examples = [
    {
        "row_model": "Claude-2.1 (scratchpad with news with...)",
        "eta_m_p": 990,
        "col_model": "Grok-2-1212 (zero shot)",
        "eta_m": 910,
    },
    {
        "row_model": "Claude-2.1 (scratchpad with news)",
        "eta_m_p": 980,
        "col_model": "Grok-2-1212 (zero shot)",
        "eta_m": 910,
    },
    {
        "row_model": "Claude-2.1 (superforecaster with news 1)",
        "eta_m_p": 960,
        "col_model": "Grok-2-1212 (zero shot)",
        "eta_m": 910,
    },
    {
        "row_model": "Qwen1.5-110B-Chat (zero shot)",
        "eta_m_p": 950,
        "col_model": "Grok-2-1212 (zero shot)",
        "eta_m": 910,
    },
    {
        "row_model": "Grok-2-1212 (scratchpad)",
        "eta_m_p": 930,
        "col_model": "Grok-2-1212 (zero shot)",
        "eta_m": 910,
    },
]

# Calculate function values
for example in examples:
    example["result"] = f(example["eta_m_p"], example["eta_m"])
    print(f"{example['row_model']} vs {example['col_model']}: {example['result']:.6f}")

# Prepare data for visualization
labels = [ex["row_model"].split(" ")[0] for ex in examples]
values = [ex["result"] for ex in examples]
scores = [ex["eta_m_p"] for ex in examples]

plt.figure(figsize=(12, 6))

# Plot 1: Function values for each model pair
plt.subplot(1, 2, 1)
bars = plt.bar(labels, values, color="skyblue")
plt.xlabel("Row Model")
plt.ylabel("f(eta_m_p, eta_m)")
plt.title("Function Values for Each Model Pair")
plt.xticks(rotation=45, ha="right")
plt.ylim(0, 1)
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Add value labels on top of each bar
for bar, value in zip(bars, values):
    plt.text(
        bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{value:.4f}", ha="center"
    )

# Plot 2: Showing relationship between score difference and function value
plt.subplot(1, 2, 2)
diff = [ex["eta_m_p"] - ex["eta_m"] for ex in examples]
plt.scatter(diff, values, s=100, c=range(len(examples)), cmap="viridis")

# Add labels for each point
for _, (d, v, l) in enumerate(zip(diff, values, labels)):
    plt.annotate(l, (d, v), xytext=(5, 5), textcoords="offset points")

# Plot the function for a range of differences
x = np.linspace(0, 100, 1000)
y = [f(d + 910, 910) for d in x]
plt.plot(x, y, "r-", alpha=0.7)

plt.xlabel("Score Difference (eta_m_p - eta_m)")
plt.ylabel("f(eta_m_p, eta_m)")
plt.title("Function Value vs. Score Difference")
plt.grid(True)

plt.tight_layout()
plt.show()
