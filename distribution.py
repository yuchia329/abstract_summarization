import polars as pl
import matplotlib.pyplot as plt
# from load_data import loadfile
import statistics
# Example Polars DataFrame with a text column
# df = loadfile('data/test.txt.src')

# Compute text length
# df = df.with_columns(pl.col("text").str.split(" ").arr.lengths().alias("text_length"))
# print(df)
with open('data/train.txt.src',mode='r') as f:
    length = []
    for line in f:
        line = line.split()
        length.append(len(line))

print(max(length))
print(statistics.mean(length))
plt.hist(length,bins=50)
plt.savefig("figures/sentence_distribution.png")
# Extract statistics
# max_length = df["text_length"].max()
# min_length = df["text_length"].min()
# avg_length = df["text_length"].mean()

# # Plot the distribution
# plt.figure(figsize=(8, 5))
# plt.hist(df["text_length"], bins=10, edgecolor="black", alpha=0.7)
# plt.xlabel("Text Length (Characters)")
# plt.ylabel("Frequency")
# plt.title("Text Length Distribution")
# plt.axvline(max_length, color='red', linestyle='dashed', linewidth=1, label=f"Max: {max_length}")
# plt.axvline(min_length, color='blue', linestyle='dashed', linewidth=1, label=f"Min: {min_length}")
# plt.axvline(avg_length, color='green', linestyle='dashed', linewidth=1, label=f"Avg: {avg_length:.2f}")
# plt.legend()
# plt.savefig("figures/sentence_distribution.png")


# # Print statistics
# print(f"Max Length: {max_length}")
# print(f"Min Length: {min_length}")
# print(f"Average Length: {avg_length:.2f}")
