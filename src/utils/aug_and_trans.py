import torchvision.transforms as transforms

def get_augumentation_and_transform(use_color_aug, use_shape_aug):
    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(
            0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])

    # if use_color_aug:
    #     c_aug = A.Compose([
    #         A.RandomBrightnessContrast(
    #             p=0.7, brightness_limit=0.5, contrast_limit=0.5),
    #         A.CoarseDropout(p=0.5, max_holes=8, max_height=16,
    #                         max_width=16, min_height=8, min_width=8, fill_value=0),
    #         A.OneOf([
    #                 A.Blur(p=1, blur_limit=7),
    #                 A.MotionBlur(p=1, blur_limit=7),
    #                 A.MedianBlur(p=1, blur_limit=7),
    #                 A.GaussianBlur(p=1, blur_limit=7)
    #                 ], p=0.5),
    #         A.OneOf([
    #                 A.RandomGamma(p=1, gamma_limit=(80, 120)),
    #                 A.GaussNoise(p=1, var_limit=(10.0, 50.0)),
    #                 A.ISONoise(p=1, color_shift=(0.01, 0.05),
    #                            intensity=(0.1, 0.5)),
    #                 ], p=0.3),
    #         A.OneOf([
    #                 A.HueSaturationValue(p=1, hue_shift_limit=20,
    #                                      sat_shift_limit=30, val_shift_limit=20),
    #                 A.RGBShift(p=1, r_shift_limit=20,
    #                            g_shift_limit=20, b_shift_limit=20),
    #                 A.CLAHE(p=1, clip_limit=4.0, tile_grid_size=(8, 8)),
    #                 # A.ChannelShuffle(p=0.5),
    #                 # A.InvertImg(p=0.5),
    #                 A.Solarize(p=1, threshold=128),
    #                 ], p=0.5),
    #     ])
    # else:
    c_aug = None
    shape_aug = None

    # crop_size = (args.train_input_h, args.train_input_w)

    # if use_shape_aug:
    #     shape_aug = A.Compose([
    #         A.HorizontalFlip(p=0.3),
    #         A.ShiftScaleRotate(p=0.5),
    #     ])
    # else:
    #     shape_aug = None

    return transform, c_aug, shape_aug