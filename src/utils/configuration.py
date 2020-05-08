import argparse

def str2bool(var):
    return 't' in var.lower()

def make_config():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--run_mode', default="train", choices=["train", "test"])

    parser.add_argument('--root_folder', type=str, default='../data/')
    parser.add_argument('--train_file_text', default='')
    parser.add_argument('--test_file_text', default='')
    parser.add_argument('--annot_file', default='../data/annotation.xlsx')

    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--optimizer', default="adam", choices=["adam", "sgd"])

    parser.add_argument('--criterion', default="mse", choices=["mse", "sigmoid"])

    parser.add_argument('--total_epoch', default=100, type=int)
    parser.add_argument('--start_epoch', default=1, type=int)

    parser.add_argument('--image_h', type=int, default=128)
    parser.add_argument('--image_w', type=int, default=128)

    parser.add_argument('--mid_channel', type=int, default=64)

    parser.add_argument('--resume', type=str2bool, default='f')

    parser.add_argument('--model_path', default='')
    parser.add_argument('--model_save_dir', default='./models')

    parser.add_argument('--debug', default='t',type=str2bool)

    return parser.parse_args()
