from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, Normalize, RandomCrop, RandomScale
from albumentations import OneOf, Compose
import ever as er


TARGET_SET = 'URBAN'

source_dir = dict(
    image_dir=[
        # './Africa/all_images/Train/',
        './LoveDA_mixedDistricts/Train/Rural512_fine/images_png/',
        './LoveDA_mixedDistricts/Train/Urban512_fine/images_png/',
        # './LoveDA_mixedRuralUrban/Train/Edges200_300/',
        # './processed_slovenia/train/urban/img_finetune/',
        # './processed_slovenia/train/rural/img_finetune/',
    ],
    mask_dir=[
        # './Africa/all_labels/Train/',
        './LoveDA_mixedDistricts/Train/Rural512_fine/masks_png/',
        './LoveDA_mixedDistricts/Train/Urban512_fine/masks_png/',
        # './processed_slovenia/train/urban/mask_finetune/',
        # './processed_slovenia/train/rural/mask_finetune/',
    ],
)
target_dir = dict(
    image_dir=[
        # './Africa/all_images/Val/',
        './LoveDA_mixedDistricts/Val/Urban512/images_png/',
        './LoveDA_mixedDistricts/Test/Urban512/images_png/',
        # './LoveDA_mixedRuralUrban/Val/Edges200_300/',
        # './processed_slovenia/val/urban/img/',
        # './processed_slovenia/test/urban/img/'
        # './processed_slovenia/val/rural/img/',
    ],
    mask_dir=[
        # './Africa/all_labels/Val/',
        './LoveDA_mixedDistricts/Val/Urban512/masks_png/',
        './LoveDA_mixedDistricts/Test/Urban512/masks_png/'
        # './processed_slovenia/val/urban/mask/',
        # './processed_slovenia/test/urban/mask/'
        # './processed_slovenia/val/rural/mask/',
    ],
)

target_dir_rural = dict(
    image_dir=[
        # './Africa/all_images/Val/',
        './LoveDA_mixedDistricts/Val/Rural512/images_png/',
        './LoveDA_mixedDistricts/Test/Rural512/images_png/',
        #'./processed_slovenia/val/rural/img/',
        #'./processed_slovenia/test/rural/img/',
    ],
    mask_dir=[
        # './Africa/all_labels/Val/',
        './LoveDA_mixedDistricts/Val/Rural512/masks_png/',
        './LoveDA_mixedDistricts/Test/Rural512/masks_png/'
        #'./processed_slovenia/val/rural/mask/',
        #'./processed_slovenia/test/rural/mask/',
    ],
)

target_test_dir = dict(
    image_dir=[
        # './Africa/all_images/Test/',
        #'./LoveDA_mixedDistricts/Test/Urban512/images_png/',
        # './LoveDA_mixedRuralUrban/Test/Edges200_300/',
        # './processed_slovenia/test/urban/img/',
    ],
    mask_dir=[
        # './Africa/all_labels/Test/',
        #'./LoveDA_mixedDistricts/Test/Urban512/masks_png/',
        # './processed_slovenia/test/urban/mask/',
    ],
)
source_test_dir = dict(
    image_dir=[
        #'./LoveDA_mixedDistricts/Test/Rural512/images_png/',
        # './LoveDA_mixedRuralUrban/Test/Edges200_300/',
        # './processed_slovenia/test/rural/img/',
    ],
    mask_dir=[
        #'./LoveDA_mixedDistricts/Test/Rural512/masks_png/',
        # './processed_slovenia/test/rural/mask/',
    ],
)


# mixed_ruralUrban
MEAN=(123.675, 116.28, 103.53)
STD=(58.395, 57.12, 57.375)
#MEAN=(74.041, 79.532, 76.112, 18.895)
#STD=(37.095, 33.228, 31.477, 1)
Edge=False

# mixed_districts
# Normalize(mean=(, , ),
#           std=(, , ),


# # Slovenia
#MEAN = (0.1273, 0.0967, 0.0868)
#STD = (0.1009, 0.1021, 0.1196)
#Edge=False

#MEAN = (52.523, 61.087, 83.399)
#STD = (31.867, 30.521, 38.968)
# Edge=False

SOURCE_DATA_CONFIG = dict(
    image_dir=source_dir['image_dir'],
    mask_dir=source_dir['mask_dir'],
    transforms=Compose([
        RandomCrop(512, 512),
        OneOf([
            HorizontalFlip(True),
            VerticalFlip(True),
            RandomRotate90(True)
        ], p=0.75),
        Normalize(mean=MEAN,
                  std=STD,
                  max_pixel_value=1, always_apply=True),
        er.preprocess.albu.ToTensor()

    ]),
    CV=dict(k=10, i=-1),
    training=True,
    batch_size=8,
    num_workers=2,
    edge_channel=Edge,
)


TARGET_DATA_CONFIG = dict(
    image_dir=target_dir['image_dir'],
    mask_dir=target_dir['mask_dir'],
    transforms=Compose([
        RandomCrop(512, 512),
        OneOf([
            HorizontalFlip(True),
            VerticalFlip(True),
            RandomRotate90(True)
        ], p=0.75),
        Normalize(mean=MEAN,
                  std=STD,
                  max_pixel_value=1, always_apply=True),
        er.preprocess.albu.ToTensor()

    ]),
    CV=dict(k=10, i=-1),
    training=True,
    batch_size=2,
    num_workers=2,
)

EVAL_DATA_CONFIG = dict(
    image_dir=target_dir['image_dir'],
    mask_dir=target_dir['mask_dir'],
    transforms=Compose([
        Normalize(mean=MEAN,
                  std=STD,
                  max_pixel_value=1, always_apply=True),
        er.preprocess.albu.ToTensor()

    ]),
    CV=dict(k=10, i=-1),
    training=False,
    batch_size=2,
    num_workers=0,
    num_classes=6,
    edge_channel=Edge,
)
EVAL_DATA_RURAL_CONFIG = dict(
    image_dir=target_dir_rural['image_dir'],
    mask_dir=target_dir_rural['mask_dir'],
    transforms=Compose([
        Normalize(mean=MEAN,
                  std=STD,
                  max_pixel_value=1, always_apply=True),
        er.preprocess.albu.ToTensor()

    ]),
    CV=dict(k=10, i=-1),
    training=False,
    batch_size=2,
    num_workers=0,
    num_classes=6,
    edge_channel=Edge,
)
TEST_DATA_CONFIG = dict(
    image_dir=target_test_dir['image_dir'],
    mask_dir=target_test_dir['mask_dir'],
    transforms=Compose([
        Normalize(mean=MEAN,
                  std=STD,
                  max_pixel_value=1, always_apply=True),
        er.preprocess.albu.ToTensor()

    ]),
    CV=dict(k=10, i=-1),
    training=False,
    batch_size=2,
    num_workers=0,
    edge_channel=Edge,
)

TEST_DATA_CONFIG_SOURCE = dict(
    image_dir=source_test_dir['image_dir'],
    mask_dir=source_test_dir['mask_dir'],
    transforms=Compose([
        Normalize(mean=MEAN,
                  std=STD,
                  max_pixel_value=1, always_apply=True),
        er.preprocess.albu.ToTensor()

    ]),
    CV=dict(k=10, i=-1),
    training=False,
    batch_size=2,
    num_workers=0,
    edge_channel=Edge,
)
