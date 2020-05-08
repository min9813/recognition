import os
import time
import pathlib
import shutil
import torch
import numpy as np
import network.lenet
import utils.configuration
import utils.average_meter
import utils.aug_and_trans
import dataset.face_data
from tqdm import tqdm


def my_makedirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def main():
    # argument　という色々条件を外から指定してファイル実行できるものを作成、
    # これはutils/configuration.py にあるmake_config関数を読み込んでいる。
    args = utils.configuration.make_config()

    # transformというnumpy -> pytorchのtensorに変換してディープラーニング用のデータクラスに変換するものを用意
    # c_aug, s_augはdata augumentationというものでいまは使わないのでNoneになっている。
    # これは utils/aug_and_trans.py の get_augumentation_and_transformに入っている。
    trans, c_aug, s_aug = utils.aug_and_trans.get_augumentation_and_transform(False, False)

    # データセットを楽にロードしてくれるもの。
    # dataset/face_data.py の Dataset クラスが該当。
    # trainデータと validationデータを作成
    train_dataset = dataset.face_data.Dataset(args, "train", trans=trans)
    valid_dataset = dataset.face_data.Dataset(args, "test", trans=trans)

    # バッチ学習を楽にしてくれるものをpytorchが提供してくれているのでそれを使う。
    #　使い方は自分で作ったdataクラスをこのdataloaderに入れる。
    # num_workersは使うスレッド数なので自前のノートPCだと0でいいと思う。
    # shuffle はデータをロードする順番をシャッフルするかということ。
    # trainデータはshuffleして、validationデータはしないのが普通。
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    mid_channel = 64
    # モデル構造。networks/lenet.py のCNNクラス。

    net = network.lenet.CNN(mid_channel=mid_channel).float()

    # optimzier
    # モデルの重みアップデータ方法を定義してくれるもの
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            net.parameters(), lr=args.lr, weight_decay=5e-4)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    else:
        raise ValueError

    # 損失関数、回帰ならとりあえず二乗誤差のmean squared lossでいいかと
    if args.criterion == "mse":
        criterion = torch.nn.MSELoss()
    elif args.criterion == "sigmoid":
        criterion = torch.nn.Sigmoid()
    else:
        raise ValueError

    # 途中まで学習したモデルを途中から再開するかどうか。
    if args.resume:
        print('resume the model parameters from: ',
              args.model_path, args.margin_path)
        net.load_state_dict(torch.load(args.model_path)['net_state_dict'])

    best_loss = np.inf
    best_iter = -1
    st = time.time()

    # 実際の学習中身
    # epoch はデータ全体を１回学習することを表す。
    # args.total_epoch　回 データを学習する感じ。
    for epoch in range(1, args.total_epoch+1):

        # train
        train_result = train(net, train_loader, optimizer,
                             criterion, args, epoch)

        # valid、trainデータにオーバーフィットしていないか確認する。
        valid_result = valid(net, valid_loader, criterion, args, epoch)

        # ログ出力
        trn_msg = ""
        for key, value in train_result.items():
            trn_msg += f" {key}:{value} "

        # ログ出力
        val_msg = ""
        for key, value in valid_result.items():
            val_msg += f" {key}:{value} "

        # メインで監視するロスを出力
        monitor_value = valid_result["loss_total"]

        print("Train:", trn_msg)
        print("Valid:", val_msg)
        times = time.time() - st
        print("elapsed time:{}s".format(times))
        is_best = best_loss > monitor_value
        if is_best:
            best_loss = monitor_value
            best_iter = epoch
        msg = 'Saving checkpoint: {}'.format(epoch)
        print(msg)

        # ここから下はモデルとoptimizer保存
        # ベストの結果が出たものと最新のモデルのみ保存している。
        model_save_dir = args.model_save_dir # これは保存ディレクトりを作っている
        my_makedirs(model_save_dir)

        # 1エポック前に保存した最新モデルを削除したいのでそれを正規表現で探している
        old_model_path = list(pathlib.Path(
            args.model_save_dir).glob("latest*.ckpt"))

        # ここら辺はパスをつなげている。
        save_net_path = os.path.join(
            model_save_dir, 'latest_{:0>6}_amp_net.ckpt'.format(epoch))
        save_best_net_path = os.path.join(
            model_save_dir, 'best_amp_net.ckpt'.format(epoch))
        save_other_path = os.path.join(
            model_save_dir, 'latest_{:0>6}_opt.ckpt'.format(epoch))
        save_best_other_path = os.path.join(
            model_save_dir, 'best_opt.ckpt'.format(epoch))

        torch.save({
            'iters': epoch,
            'net': net.state_dict(),
            'score': monitor_value
        },
            save_net_path)
        torch.save({
            'iters': epoch,
            'optimizer': optimizer.state_dict(),
            # 'amp': amp.state_dict()
        },
            save_other_path)

        if is_best:
            shutil.copy(str(save_other_path),
                        str(save_best_other_path))
            shutil.copy(str(save_net_path), str(save_best_net_path))

        for p in old_model_path:
            os.remove(str(p))

        # デバッグ用。2エポック学習したら終わり。
        if args.debug:
            if epoch - args.start_epoch > 0:
                break
    print('Finally Best Accuracy: {:.4f} in iters: {} (Val Loss={:.4f})'.format(
        best_loss, best_iter, best_loss))
    print('finishing training')


def train(net, dataloader, optimizer, criterion, args, epoch):
    net.train()
    average_meter = utils.average_meter.AverageMeter()

    num_iter = len(dataloader)
    iter_st = time.time()
    data_st = time.time()
    for idx, data in enumerate(dataloader):
        time_data = time.time() - data_st
        average_meter.add_value("time_data", time_data)
        x_data = data["data"]
        label = data["label"].type(torch.float32)
        # print(x_data.shape, x_data.dtype)
        # print(net.layer1.weight.dtype, label.dtype)
        model_st = time.time()
        output = net(x_data)

        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        time_model = time.time() - model_st
        average_meter.add_value("time_model", time_model)

        average_meter.add_value("loss_total", loss)

        if idx % 10 == 0:
            msg = "[{}/{}]".format(idx, num_iter)

            summary = average_meter.get_summary()
            for key, value in summary.items():
                msg += " {}:{} ".format(key, value)

            print(msg)

        time_iter = time.time() - iter_st
        average_meter.add_value("time_iter", time_iter)
        iter_st = time.time()

    summary = average_meter.get_summary()

    return summary


def valid(net, dataloader, criterion, args, epoch):
    net.eval()
    average_meter = utils.average_meter.AverageMeter()

    num_iter = len(dataloader)
    with torch.no_grad():
        for idx, data in tqdm(enumerate(dataloader), total=num_iter):
            x_data = data["data"]
            label = data["label"]

            output = net(x_data)
            loss = criterion(output, label)

            average_meter.add_value("loss_total", loss)

    summary = average_meter.get_summary()

    return summary


if __name__ == "__main__":
    main()
