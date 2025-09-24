# visualize_all.py
import pandas as pd
import matplotlib.pyplot as plt

CSV_FILES = [
    "../02_capture_data/data/up.csv",
    "../02_capture_data/data/down.csv",
    "../02_capture_data/data/left.csv",
    "../02_capture_data/data/right.csv",
    "../02_capture_data/data/front.csv",
    "../02_capture_data/data/still.csv",
]

SENSOR_COLUMNS = ["gx", "gy", "gz", "ax", "ay", "az"]

def plot_reps(ax, df, signal, reps_to_plot=[1, 2, 3, 4, 5]):
    """Plot selected repetitions for one signal into provided axis."""
    for rep in reps_to_plot:
        rep_df = df[df["rep"] == rep].sort_values("t_ms")
        ax.plot(rep_df["t_ms"], rep_df[signal], label=f"rep {rep}")
    ax.set_ylabel(signal)
    ax.set_xlabel("Time (ms)")

def main():
    fig, axes = plt.subplots(
        len(SENSOR_COLUMNS), len(CSV_FILES),
        figsize=(4 * len(CSV_FILES), 2 * len(SENSOR_COLUMNS)),
        sharex=False
    )
    fig.suptitle("Gesture comparisons per signal", fontsize=16)

    for col_idx, csv in enumerate(CSV_FILES):
        gesture = csv.split("/")[-1].split(".")[0].upper()
        df = pd.read_csv(csv)

        for row_idx, signal in enumerate(SENSOR_COLUMNS):
            ax = axes[row_idx, col_idx]
            plot_reps(ax, df, signal)
            if row_idx == 0:
                ax.set_title(gesture)

    # Add legends only once (top-left subplot)
    axes[0, 0].legend(loc="upper right", fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

if __name__ == "__main__":
    main()

