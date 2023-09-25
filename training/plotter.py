"""
Create plots from the generated CSV files.
"""

import matplotlib.pyplot as plt
import csv
import argparse


def main():
    # load in the file
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default=None)
    args = parser.parse_args()
    if args.file is None:
        raise ValueError("Please specify a file to load")

    else:
        # load in the data
        data = []
        filepath = f"./data/logs/{args.file}.csv"
        with open(filepath, "r") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append(row)

        # plot the data
        plt.plot(
            [float(x["episode"]) for x in data],
            [float(x["avg_score"]) for x in data],
            label=f"{args.file}",
        )
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Reward over time")
        plt.legend()
        plt.savefig(f"./data/plots/{args.file}.png")

        plt.show()


if __name__ == "__main__":
    main()
