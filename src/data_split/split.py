import os
import pathlib
import numpy as np


def split(data_dir, valid_ratio=0.1):
    # pathlibライブラリはディレクトリの操作をすごい楽にしてくれるのでpathlibリアブラリが提供する
    # pathlib.Path クラスにディレクトリを変換する。
    data_dir = pathlib.Path(data_dir)
    # glob("*") はディレクトリ以下の正規表現に該当するファイル一覧を取得する。
    # デフォルトではiteratorというので返ってくるので、それをリストに変換している。
    # 今 data_dir = ../data であるので、*/*は dataディレクトリの2階層下までの全てのファイルを検索してくる。
    # ../data/man/00001.png, ../data/man/00002.png, ... ,と../data/woman/00001.png, .... をとってきてくれる。
    # .glob("*")だと ../data/man, ../data/woman までしかとってきてくれない。
    all_path = list(data_dir.glob('*/*'))

    # all_path には print(all_path) してみればわかるけど全ての画像へのパスが入っている。
    # [./data/man/00001.png, ./data/man/00002.png , ... ] という感じ
    # 検証用のデータサイズを決める。これは 全てのパスに対して検証用の割合をかけて、
    # それを整数にしている。例えばデータ数 88 に対して 0.1かけると 8.8になってしまうので
    # これをint() で整数に丸めて、8にする感じ。
    size = int(len(all_path) * valid_ratio)
    assert size > 5, f"size : {size}" # これはサイズが5以上出なかったらエラーを出して止める感じ

    # これは all_pathに入っているものを size= で指定したサイズだけランダムに取り出している。
    # replace = Falsehは重複なしの取り出し
    # 例えば np.random.choie([1,2,3,4,5], size=2, replace=False)なら、
    # [1,2,3,4,5]からランダムに2個取り出す。(1,2)とか(5,1)とか,(3, 4)とか
    # numpyはこういうことが1行でできてしまうので便利
    valid_path = np.random.choice(all_path, size=size, replace=False)

    # setに変換することで、後ろの path in valid_path が高速になる。
    valid_path = set(valid_path)

    # __file__ はこの split.py のパスを指す。
    # それを pathlib.Path クラスに変換する。 .parentは親ディレクトリを返してくレル。
    # この場合だと ./data になる
    txt_file_dir = pathlib.Path(__file__).parent

    # train用のデータを書き込むテキストファイル。./data/train_path.txt に書く感じ
    train_txt_file = os.path.join(txt_file_dir, "train_path.txt")
    # valid用のデータを書き込むテキストファイル。./data/valid_path.txt に書く感じ
    valid_txt_file = os.path.join(txt_file_dir, "valid_path.txt")

    # 上のファイルたちが既にあるなら、後で書き足しモードで書いていくので、
    # 既に書き込まれた内容が邪魔になるのでそれを消す
    # wモードは新規ファイル作成モードなのでそれでしっかりと消している。
    for path in [train_txt_file, valid_txt_file]:
        with open(str(path), "w") as f:
            f.write("")

    for path in all_path:
        # path = ../data/man/00001.png 
        # path.parent = ../data/man (pathの親階層)
        # path.parent.name = man (path.parent の一番末尾の部分をとる)
        dir_name = pathlib.Path(path.parent).name
        # path.name = 00001.png (path の一番末尾の部分をとる)
        # dir_nameとつなげると man/00001.png となる。
        save_name = os.path.join(dir_name, path.name)
        # ../data/man/00001.png が validに入っているならvalid_txt_fileに書き込む。
        if path in valid_path:
            # 書き込みモードを "a" にすることで次々と書いてくれる。
            with open(str(valid_txt_file), "a") as f:
                # ファイル名を書き込んだら改行する。
                f.write(save_name+"\n")
        # そうでないなら train_txt_fileに書き込む
        else:
            with open(str(train_txt_file), "a") as f:
                f.write(save_name+"\n")


def main():
    # face 写真が入っている dataディレクトリ
    data_dir = "../data"
    # 検証用データの割合
    valid_ratio = 0.2
    split(data_dir, valid_ratio)


if __name__ == "__main__":
    main()
