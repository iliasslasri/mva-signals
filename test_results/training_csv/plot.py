import glob
import re

import matplotlib.pyplot as plt
import pandas as pd


def moving_average(data, window_size=2):
    return data.rolling(window=window_size, min_periods=1, center=True).mean()


plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 17,
        "font.family": "serif",
        "figure.dpi": 200,
        "lines.linewidth": 2.2,
    }
)
limit = 400

snr_files = {}
for f in glob.glob("*snr_*dB.csv"):
    m = re.search(r"snr_(\d+)dB", f)
    if m:
        snr = int(m.group(1))
        snr_files[snr] = f

global_val_file = None
for f in glob.glob("*val_accuracy*.csv"):
    if "snr_" in f:
        global_val_file = f
        break

fig_combined, ax_combined = plt.subplots(figsize=(10, 5))

for snr, path in sorted(snr_files.items()):
    df = pd.read_csv(path, sep="\t|,|;", engine="python")
    df = df[df["Step"] <= limit]

    y_smooth = moving_average(df["Value"])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(df["Step"], y_smooth, label=f"{snr} dB", linewidth=2.5)
    ax.set_xlabel("Validation Step")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title(f"Validation Accuracy (SNR = {snr} dB)")
    ax.grid(True, linestyle="--", alpha=0.45)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"val_acc_snr_{snr}dB_ma.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    ax_combined.plot(df["Step"], y_smooth, label=f"{snr} dB")

if global_val_file:
    df_global = pd.read_csv(global_val_file, sep="\t|,|;", engine="python")
    df_global = df_global[df_global["Step"] <= limit]
    y_global_smooth = moving_average(df_global["Value"])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        df_global["Step"], y_global_smooth, color="black", label="Global", linewidth=2.5
    )
    ax.set_xlabel("Validation Step")
    ax.set_ylabel("Global Validation Accuracy")
    ax.set_title("Validation Accuracy (All SNRs Combined)")
    ax.grid(True, linestyle="--", alpha=0.45)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.savefig("val_acc_global_ma.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    ax_combined.plot(
        df_global["Step"],
        y_global_smooth,
        label="Global",
        # linestyle="--",
        linewidth=2.7,
        color="black",
    )

ax_combined.set_xlabel("Validation Step")
ax_combined.set_ylabel("Validation Accuracy")
ax_combined.set_title("Validation Accuracy for All SNRs")
ax_combined.grid(True, linestyle="--", alpha=0.45)
ax_combined.legend(title="SNR")
for spine in ["top", "right"]:
    ax_combined.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig("val_acc_all_snr_ma.png", dpi=300, bbox_inches="tight")
plt.show()


# --- Paths to your CSV files ---
train_csv = "20251121-131516_cnn_lstm_snr_augment_snr_validation_loss.csv"  # replace with your training loss CSV path
val_csv = "20251121-131516_cnn_lstm_snr_augment_snr_trainlossperepoch.csv"  # replace with your validation loss CSV path

# --- Load CSVs ---
df_train = pd.read_csv(train_csv)
df_val = pd.read_csv(val_csv)


# --- Optional: smooth with moving average ---
def moving_average(data, window_size=10):
    return data.rolling(window=window_size, min_periods=1, center=True).mean()


df_train["Value_smooth"] = moving_average(df_train["Value"], window_size=10)
df_val["Value_smooth"] = moving_average(df_val["Value"], window_size=10)

# --- Plot ---
plt.figure(figsize=(8, 5))
plt.plot(
    df_train["Step"],
    df_train["Value_smooth"],
    label="Training Loss",
    color="red",
    linewidth=2,
)
plt.plot(
    df_val["Step"],
    df_val["Value_smooth"],
    label="Validation Loss",
    color="blue",
    linewidth=2,
)

plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()

# --- Save figure ---
plt.savefig("training_validation_loss.png", dpi=300, bbox_inches="tight")
plt.show()
