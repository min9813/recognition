import os
import pathlib
import numpy as np


def split(data_dir, valid_ratio=0.1):
    data_dir = pathlib.Path(data_dir)
    all_path = list(data_dir.glob('*/*'))
    size = int(len(all_path) * valid_ratio)
    assert size > 5, f"size : {size}"
    valid_path = np.random.choice(all_path, size=size, replace=False)
    valid_path = set(valid_path)

    txt_file_dir = pathlib.Path(__file__).parent
    train_txt_file = os.path.join(txt_file_dir, "train_path.txt")
    valid_txt_file = os.path.join(txt_file_dir, "valid_path.txt")

    for path in [train_txt_file, valid_txt_file]:
        with open(str(path), "w") as f:
            f.write("")

    for path in all_path:
        dir_name = pathlib.Path(path.parent).name
        save_name = os.path.join(dir_name, path.name)
        if path in valid_path:
            with open(str(valid_txt_file), "a") as f:
                f.write(save_name+"\n")
        else:
            with open(str(train_txt_file), "a") as f:
                f.write(save_name+"\n")


def main():
    data_dir = "../data"
    valid_ratio = 0.2
    split(data_dir, valid_ratio)


if __name__ == "__main__":
    main()
