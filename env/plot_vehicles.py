"""
plot_vehicles.py

This script reads a CSV file containing vehicle position data over time and
plots the positions of the 12 vehicles at a specified time. The first 8 vehicles
are V2V (grouped in 4 pairs), and the last 4 vehicles are V2I.

Edit the CSV_FILE, TIME_POINT, X_LIM, and Y_LIM variables below, then click
“Run” in PyCharm. The plot will appear in its own window (using the TkAgg backend).
"""

import sys
import pandas as pd

# Force matplotlib to use an external window backend (TkAgg) instead of PyCharm’s SciView
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


# ——————————————
# User-configurable variables
# ——————————————

# Path to your CSV file (change as needed)
CSV_FILE = "SUMOData/Journal_4_4.csv"

# Time (in seconds, must match one of the “time” values in the CSV, e.g. 0.0, 0.1, 0.2, …)
TIME_POINT = 90.1  # example: plot at t = 10.6 s

# Manually set axis limits: (xmin, xmax) and (ymin, ymax)
# Change these values to zoom in or out as desired
X_LIM = (-1000, 1000)
Y_LIM = (0, 20)


def plot_positions(csv_file: str, time_point: float):
    # Read the entire CSV into a DataFrame
    df_all = pd.read_csv(csv_file)

    # Filter rows corresponding to the requested time
    df_time = df_all[df_all["time"] == time_point]
    if df_time.empty:
        print(f"No data found for time = {time_point} s. Available times are multiples of 0.1 s.")
        sys.exit(1)

    # Index by 'id' for easy lookup
    df_time = df_time.set_index("id")

    # Define V2V pairs (first 8 vehicles in groups of 2)
    v2v_pairs = [
        ["carflow0_0.0", "carflow0_1.0"],
        ["carflow1_0.0", "carflow1_1.0"],
        ["carflow2_0.0", "carflow2_1.0"],
        ["carflow3_0.0", "carflow3_1.0"],
    ]
    # Define V2I IDs (last 4 vehicles)
    v2i_ids = ["carflowV2I_0.0", "carflowV2I_1.0", "carflowV2I_2.0", "carflowV2I_3.0"]

    # Create figure + axes
    fig, ax = plt.subplots(figsize=(12, 5))  # wider figure to fit X range

    # Colors for each V2V pair
    pair_colors = ["tab:blue", "tab:red", "tab:green", "tab:orange"]

    #add centerline
    # Solid yellow center line at y = 10
    ax.axhline(y=10.2, color='gold', linestyle='-', linewidth=2)

    # Dashed black lane divider lines at 3.6, 7.2, 14.4, 16.0 (and optionally 21.6, etc.)
    lane_lines = [3.2, 6.6, 13.6, 16.8]
    for y in lane_lines:
        ax.axhline(y=y, color='black', linestyle='--', linewidth=1)

    # Plot each V2V pair
    for idx, pair in enumerate(v2v_pairs):
        tx_id, rx_id = pair
        try:
            x_tx, y_tx = df_time.loc[tx_id, ["x", "y"]]
            x_rx, y_rx = df_time.loc[rx_id, ["x", "y"]]
        except KeyError:
            print(f"Could not find IDs {pair} at time {time_point}. Skipping this pair.")
            continue

        # Plot receiver (circle) with legend label
        ax.scatter(
            [x_rx],
            [y_rx],
            c=pair_colors[idx],
            marker="o",
            s=100,
            edgecolors="k",
            linewidths=0.5,
            label=f"V2V {idx}",
        )

        # Plot transmitter as a "T" using text (no marker, no legend entry)
        ax.text(
            x_tx,
            y_tx,
            "T",
            fontsize=12,
            color=pair_colors[idx],
            ha="center",
            va="center",
            fontweight="bold",
        )

        # Add "R" at receiver location
        ax.text(
            x_rx,
            y_rx,
            "R",
            fontsize=10,
            color="white",  # white on top of colored marker
            ha="center",
            va="center",
            fontweight="bold",
        )

    # Plot V2I vehicles as numbered labels (0, 1, 2, 3)
    for idx, vid in enumerate(v2i_ids):
        try:
            xi, yi = df_time.loc[vid, ["x", "y"]]
        except KeyError:
            print(f"Could not find V2I ID {vid} at time {time_point}.")
            continue

        # Draw the index number at the vehicle's position
        ax.text(
            xi,
            yi,
            str(idx),
            fontsize=10,
            color="black",
            ha="center",
            va="center",
            fontweight="bold",
        )


    # # Plot all V2I vehicles as black 'X' markers
    # v2i_x = []
    # v2i_y = []
    # for vid in v2i_ids:
    #     try:
    #         xi, yi = df_time.loc[vid, ["x", "y"]]
    #     except KeyError:
    #         print(f"Could not find V2I ID {vid} at time {time_point}.")
    #         continue
    #     v2i_x.append(xi)
    #     v2i_y.append(yi)
    #
    # if v2i_x:
    #     ax.scatter(
    #         v2i_x,
    #         v2i_y,
    #         c="k",
    #         marker="X",
    #         s=100,
    #         label="V2I Vehicles",
    #         edgecolors="k",
    #         linewidths=0.5,
    #     )

    # Annotate each point with its ID (offset by +0.5 so it doesn’t overlap badly)
    # for vid in df_time.index:
    #     xi, yi = df_time.loc[vid, ["x", "y"]]
    #     ax.annotate(vid, (xi + 0.5, yi + 0.5), fontsize=8, alpha=0.75)

    # Apply manual axis limits BEFORE setting aspect
    ax.set_xlim(X_LIM)
    ax.set_ylim(Y_LIM)

    # If you still want equal scaling in both directions, uncomment this line:
    # ax.set_aspect('equal', adjustable='box')
    # Otherwise, leave aspect as 'auto' so that the X‐range and Y‐range are shown proportionally.

    # Formatting
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title(f"Vehicle Positions at time = {time_point:.1f} s")
    ax.legend(loc="upper right",ncol=5, fontsize='x-small')

    custom_yticks = [0, 1.6, 4.8, 8.4, 12, 15.2, 18.4]
    ax.set_yticks(custom_yticks)
    # ax.grid(True)

    # Show plot in a new window
    plt.tight_layout()
    plt.show()


def main():
    plot_positions(CSV_FILE, TIME_POINT)


if __name__ == "__main__":
    main()