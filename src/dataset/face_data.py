import os
import pathlib
import cv2
import numpy as np
import pandas as pd
import dataset.preprocess


class Dataset(object):

    def __init__(self, args, split, trans=None):
        self.split = split
        # utils/configuration.py で用意したconfigのやつ
        self.args = args
        assert self.split in ("train", "test")

        # annotationとデータから画像と対応する点数を読み込む
        if self.split == "test":
            self.load_data(self.args.test_file_text)
        else:
            self.load_data(self.args.train_file_text)

        self.trans = trans
        print(f"Loaded {len(self.dataset)} samples !")

    def __getitem__(self, index):
        # pytorchのDataLoaderクラスがここにアクセスしてくる。
        # なのでここがメイン部分
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
        # trainに使うデータとvalidに使うデータを分割してtextファイルに記述したので対応するものを読み込む
        with open(text_file, "r") as f:
            text = f.readlines()

        # これはスコアが書かれたエクセルファイルを読み込んでいる。
        data = pd.read_excel(self.args.annot_file)
        # エクセルを読み込むと、fileの欄が数字扱いされて00012 -> 12 となるので
        # ファイル名と合致するように0を左に埋める処理 {;0>5} をしている。
        data["file"] = data["file"].map(lambda x:"{:0>5}".format(x))
        # ここはファイル名を man/ファイル名, woman/ファイル名、にした方が見通しがいいので
        # そう改変してる。あとtrain, valid分割データを書いたファイルがその形式なのでそうしている
        # 今はアノテーションがmanしかないのでこんな感じ。
        # {:0>5} しているけどこれはいったん書いたやつを直しいないだけなので意味はない。
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
            # file_name は man/00012.png みたいに余計な拡張子がついているので
            # annotationファイルに合わせるために、それを消して、 man/00012 にしている。
            check_name = str(pathlib_format.parent) +"/" + pathlib_format.stem
            # annotationにないデータを削除
            if check_name not in ok_file_list:
                continue
            # 下2行はpandasの操作、エクセルから対応するファイルの行を出して、
            # そのスコアを出している。 .valudsでnumpyのarrayに、.astype で浮動小数点型に変換
            # 浮動小数点は小数点を扱えるようにしたものと思っておけばおk
            #　例えば整数型があって、それだと 3 /2 = 1 となってしまう。（pythonだとならないけど）
            # あと "0" は文字列型だけど floatにすることで 0.0になってくれる。
            mask = data["file"] == pathlib_format.stem
            score = data.loc[mask, "score"].values.astype(float)

            # スコアが -1 ~ 1になるように正規化、値が大きいと学習はうまく行きにくい。
            # モデルの予測値にこれの逆変換を施せば欲しいものが手に入るのでおk
            score = (score - 50) / 50

            # indexとしてfile_pathをアクセスさせて、対応する 画像, ラベルを返す実装が
            # 見通し良さそうだったので、self.dataset にfile pathを入れている。
            # __getitem__ のindexはこのdatasetにアクセスさせる。詳細は
            # __getitem__ 関数と get_data関数を見た方がいい。
            self.dataset.append(file_path)

            # 画像読み込み
            img = cv2.imread(file_path)
            img = dataset.preprocess.preprocess_image(
                img, self.args.image_h, self.args.image_w)
            # self.dataset.append(img)

            # 辞書にする
            self.annotations[file_path] = {"img": img, "label": score}

    # len(nannka) で返すもの
    def __len__(self):
        return len(self.dataset)
