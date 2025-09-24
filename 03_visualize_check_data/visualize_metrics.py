# visualize_accelerations.py
import pandas as pd
import matplotlib.pyplot as plt

CSV_FILES = {
    "UP": "../02_capture_data/data/up.csv",
    "DOWN": "../02_capture_data/data/down.csv",
    "LEFT": "../02_capture_data/data/left.csv",
    "RIGHT": "../02_capture_data/data/right.csv",
    "FRONT": "../02_capture_data/data/front.csv",
    "STILL": "../02_capture_data/data/still.csv",
}

SENSOR_COLUMNS = ["gx", "gy", "gz", "ax", "ay", "az"]

# Assign a fixed color per gesture
GESTURE_COLORS = {
    "UP": "tab:blue",
    "DOWN": "tab:orange",
    "LEFT": "tab:green",
    "RIGHT": "tab:red",
    "FRONT": "tab:purple",
    "STILL": "tab:brown",
}

def plot_signal(ax, signal, reps_to_plot=[1, 2, 3]):
    """Overlay all gestures for a given signal into one axis, same color per gesture."""
    for gesture, path in CSV_FILES.items():
        df = pd.read_csv(path)
        color = GESTURE_COLORS.get(gesture, None)
        for rep in reps_to_plot:
            rep_df = df[df["rep"] == rep].sort_values("t_ms")
            # use same color, but no legend for every rep (to avoid duplicates)
            ax.plot(rep_df["t_ms"], rep_df[signal], color=color, alpha=0.7)
        # add one legend entry per gesture
        ax.plot([], [], color=color, label=gesture)

    ax.set_title(signal)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(signal)

def main():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=False)

    fig.suptitle("Overlay of all gestures per signal (same color per gesture)", fontsize=16)

    axes = axes.flatten()
    
    for i, signal in enumerate(SENSOR_COLUMNS):
        plot_signal(axes[i], signal)

    fig.legend(loc="upper right", fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

if __name__ == "__main__":
    main()

