import os
import pathlib
import cv2
import numpy as np
import pandas as pd
import dataset.preprocess


class Dataset(object):

    def __init__(self, args, split, trans=None):
        self.split = split
        self.args = args
        assert self.split in ("train", "test")

        if self.split == "test":
            self.load_data(self.args.test_file_text)
        else:
            self.load_data(self.args.train_file_text)

        self.trans = trans
        print(f"Loaded {len(self.dataset)} samples !")

    def __getitem__(self, index):
        data = self.get_data(index)

        return data

    def get_data(self, index):
        path = self.dataset[index]
        image = self.annotations[path]["img"]
        score = self.annotations[path]["label"]

        if self.trans is not None:
            image = self.trans(image)

        data = {"data": image, "label": score}

        return data

    def load_data(self, text_file):
        with open(text_file, "r") as f:
            text = f.readlines()

        data = pd.read_excel(self.args.annot_file)
        data["file"] = data["file"].map(lambda x:"{:0>5}".format(x))
        ok_file_list = set(
            map(lambda x: "man/{:0>5}".format(x), data["file"].values.tolist()))
        # print(ok_file_list)

        self.annotations = {}
        self.dataset = []
        for file_name in text:
            # print(file_name)
            file_name = file_name.strip()
            file_path = os.path.join(self.args.root_folder, file_name)

            pathlib_format = pathlib.Path(file_name)
            check_name = str(pathlib_format.parent) +"/" + pathlib_format.stem
            if check_name not in ok_file_list:
                continue
            # print(data.columns)
            mask = data["file"] == pathlib_format.stem
            score = data.loc[mask, "score"].values.astype(float)
            score = (score - 50) / 50
            # print(data["file"], pathlib_format.stem)
            # raise ValueError
            self.dataset.append(file_path)

            img = cv2.imread(file_path)
            img = dataset.preprocess.preprocess_image(
                img, self.args.image_h, self.args.image_w)
            # self.dataset.append(img)

            self.annotations[file_path] = {"img": img, "label": score}

    def __len__(self):
        return len(self.dataset)
