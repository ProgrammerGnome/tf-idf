import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

file_path = "outputs/benchmark_scaling.txt"
if not os.path.exists(file_path):
    print(f"Hiba: A '{file_path}' nem található. Futtasd előbb a benchmark kódot!")
    exit()

df = pd.read_csv(file_path)

sns.set_theme(style="whitegrid")
colors = {
    "Sklearn_All": "#e74c3c",       # Piros
    "Sklearn_Batched": "#f39c12",   # Narancs
    "Custom_All": "#3498db",        # Kék
    "Custom_Batched": "#2ecc71"     # Zöld
}

# --- 1. Futásidő Diagram (CPU) ---
plt.figure(figsize=(10, 6))
for method, color in colors.items():
    subset = df[df["Method"] == method]
    plt.plot(subset["Limit"], subset["Time_sec"], marker='o', linewidth=2, label=method, color=color)

plt.title("Futásidő (Másodperc) vs. Dokumentumok száma", fontsize=14, pad=15)
plt.xlabel("Dokumentumok száma", fontsize=12)
plt.ylabel("Futásidő (mp)", fontsize=12)
plt.xticks(df["Limit"].unique(), rotation=45)
plt.legend(title="Módszerek", fontsize=10)
plt.tight_layout()
plt.savefig("outputs/plot_cpu_time.png", dpi=300)
plt.show()

# --- 2. Memóriahasználat Diagram (RAM) ---
plt.figure(figsize=(10, 6))
for method, color in colors.items():
    subset = df[df["Method"] == method]
    plt.plot(subset["Limit"], subset["RAM_MB"], marker='s', linewidth=2, label=method, color=color)

plt.title("Csúcs Memóriahasználat (MB) vs. Dokumentumok száma", fontsize=14, pad=15)
plt.xlabel("Dokumentumok száma", fontsize=12)
plt.ylabel("Memória (MB)", fontsize=12)
plt.xticks(df["Limit"].unique(), rotation=45)
plt.legend(title="Módszerek", fontsize=10)
plt.tight_layout()
plt.savefig("outputs/plot_memory.png", dpi=300)
plt.show()

print("A diagramok sikeresen legenerálva és mentve az 'outputs' mappába!")
