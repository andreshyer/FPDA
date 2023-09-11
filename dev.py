import re
from pathlib import Path
from pandas import DataFrame
from numpy import savetxt


def check_string(s):
    pattern = r'^(\d+)\s+(\d+)$'
    return re.match(pattern, s) is not None

def prfs_to_csv():
    root_dir = Path("drops/prfs")
    output_dir = Path("drops/real_drop_profiles")

    for file in root_dir.iterdir():

        data = []
        i = 0
        with open(file, "r") as f:
            for line in f.readlines():

                if line[0] == "[":
                    data = DataFrame(data, columns=["x", "z"])
                    data = data.astype(float)

                    data["z"] = - data["z"]

                    left_x = data["x"].tolist()[0]
                    right_x = data["x"].tolist()[-1]
                    mid_x = (right_x + left_x) / 2

                    data = data[data["x"] < mid_x]
                    data["x"] = - data["x"]

                    data["x"] = data["x"] - data["x"].min()
                    data["z"] = data["z"] - data["z"].min()

                    data = data.sort_values(by="z")

                    max_length = data.max().max()
                    data = data / max_length

                    savetxt(output_dir / (file.stem + f"_{i}.csv"), data, delimiter=",")
                    i += 1
                    data = []
                elif check_string(line):
                    split_line = line.split()
                    if split_line:
                        x, z = split_line
                        data.append([x, z])


if __name__ == "__main__":
    prfs_to_csv()
