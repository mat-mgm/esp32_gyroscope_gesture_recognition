# visualize_gestures.py
import pandas as pd
import matplotlib.pyplot as plt

CSV_FILES = ["../02_capture_data/data/up.csv", "../02_capture_data/data/down.csv", "../02_capture_data/data/left.csv", "../02_capture_data/data/right.csv"]

def plot_reps(axs, csv_file, gesture_name, reps_to_plot=[1,2,3]):
    """Plot selected repetitions for one gesture into provided axes."""
    df = pd.read_csv(csv_file)

    ax_g, ax_a = axs  # first row = gyro, second row = accel

    for rep in reps_to_plot:
        rep_df = df[df["rep"] == rep].sort_values("t_ms")

        # Gyroscope
        ax_g.plot(rep_df["t_ms"], rep_df["gx"], label=f"rep {rep} - gx")
        ax_g.plot(rep_df["t_ms"], rep_df["gy"], label=f"rep {rep} - gy")
        ax_g.plot(rep_df["t_ms"], rep_df["gz"], label=f"rep {rep} - gz")

        # Accelerometer
        ax_a.plot(rep_df["t_ms"], rep_df["ax"], label=f"rep {rep} - ax")
        ax_a.plot(rep_df["t_ms"], rep_df["ay"], label=f"rep {rep} - ay")
        ax_a.plot(rep_df["t_ms"], rep_df["az"], label=f"rep {rep} - az")

    ax_g.set_title(f"{gesture_name}")
    ax_g.set_ylabel("Gyro (°/s)")
    ax_a.set_ylabel("Accel (g)")
    ax_a.set_xlabel("Time (ms)")


def main():
    fig, axes = plt.subplots(2, len(CSV_FILES), figsize=(16, 6), sharex=False)
    fig.suptitle("Gesture comparisons (Gyro top / Accel bottom)", fontsize=14)

    for i, csv in enumerate(CSV_FILES):
        gesture = csv.split("/")[-1].split(".")[0]
        plot_reps((axes[0, i], axes[1, i]), csv, gesture)

    # Only show legends once to avoid clutter
    axes[0,0].legend(loc="upper right", fontsize=7)
    axes[1,0].legend(loc="upper right", fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    main()
