# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 10:09:18 2023

@author: linwang
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 15:46:18 2023

@author: linwang
"""

# %% import library
import os, cv2, math, csv
import numpy as np
import pandas as pd
import glob

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import warnings

warnings.filterwarnings("ignore")

import torch
import albumentations as album
import argparse
from PIL import Image

import segmentation_models_pytorch as smp

import segmentation_models_pytorch.utils


# crop image
def crop_image(image, true_dimensions):
    return album.CenterCrop(p=1, height=true_dimensions[0], width=true_dimensions[1])(
        image=image
    )


# viz
# Perform colour coding on the reverse-one-hot outputs
def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.
    # Arguments
        image: single channel array where each value represents the class key.
        label_values

    # Returns
        Colour coded image for segmentation visualization
    """
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x


# Perform reverse one-hot-encoding on labels / preds
def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.
    # Arguments
        image: The one-hot format image

    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified
        class key.
    """
    x = np.argmax(image, axis=-1)
    return x


#  prediction dataset
class PredictionDataset(torch.utils.data.Dataset):
    """Stanford Background Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        df (str): DataFrame containing images / labels paths
        class_rgb_values (list): RGB values of select classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    def __init__(
        self,
        df,
        class_rgb_values=None,
        augmentation=None,
        preprocessing=None,
    ):
        self.image_paths = df["image_path"].tolist()

        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read images and masks
        # print(self.image_paths[i])
        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
        # names = self.image_paths[i].rsplit('/', 1)[-1].split('.')[0]
        names = (
            self.image_paths[i].split("/", 3)[-1].split(".")[0]
        )  # split start from the wave

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image)
            image = sample["image"]

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample["image"]

        return image, names

    def __len__(self):
        # return length of
        return len(self.image_paths)


def get_training_augmentation():
    train_transform = [
        # album.PadIfNeeded(min_height=550, min_width=660, always_apply=True, border_mode=0),
        # LW height and width change from 832 to 1000 to 1984 to 2720
        album.RandomCrop(height=1024, width=1024, always_apply=True),
        album.OneOf(
            [
                album.HorizontalFlip(p=1),
                album.VerticalFlip(p=1),
                album.RandomRotate90(p=1),
            ],
            p=0.5,
        ),
    ]
    return album.Compose(train_transform)


def get_validation_augmentation():
    # Add sufficient padding to ensure image is divisible by 32
    test_transform = [
        # LW size should be square and can be devided by 32
        # LW height and width change from 992 to 1120 to 2752
        album.PadIfNeeded(
            min_height=1024,
            min_width=1024,
            always_apply=True,
            border_mode=0,
            value=(0, 0, 0),
        ),
    ]
    return album.Compose(test_transform)


def get_test_augmentation():
    # Add sufficient padding to ensure image is divisible by 32
    test_transform = [
        # LW change min_height of 1120 to 6016 to 2016, same as val
        # LW change min_width of 992 to 4000 to 1120 for checking images instead of images_test
        album.PadIfNeeded(
            min_height=256,
            min_width=256,
            always_apply=True,
            border_mode=0,
            value=(0, 0, 0),
        ),
    ]
    return album.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype("float32")


def get_preprocessing(preprocessing_fn=None):
    """Construct preprocessing transform
    Args:
        preprocessing_fn (callable): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """
    _transform = []
    if preprocessing_fn:
        _transform.append(album.Lambda(image=preprocessing_fn))
    _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))

    return album.Compose(_transform)


# crop images
def crop_img(image_path, crop_folder, save_path):
    title = ["wave", "scanner", "plant", "frame", "layer_ind"]
    data = np.zeros([1, 5])
    dataframe = pd.DataFrame(data, columns=title)
    layer_boundary_threshold = 20
    window_size = 5  # sliding windows for the moving average

    wave_list = [
        f for f in os.listdir(image_path) if os.path.isdir(os.path.join(image_path, f))
    ]
    for wave in wave_list:
        scanner_list = [
            f
            for f in os.listdir(os.path.join(image_path, wave))
            if os.path.isdir(os.path.join(image_path, wave, f))
        ]

        for scanner in scanner_list:

            startY = (
                180 if scanner.endswith("SlowScanner") else 219
            )  # based on the 1030 Y for slowscanner, 1069Y for FastScannner
            startX = 526 if scanner.endswith("SlowScanner") else 386
            height = 850
            width = 970

            plant_list = [
                f
                for f in os.listdir(os.path.join(image_path, wave, scanner))
                if os.path.isdir(os.path.join(image_path, wave, scanner, f))
            ]

            for plant in plant_list:
                print(f"cropping wave: {wave}, scanner: {scanner}, plant: {plant}")
                img_list = [
                    f
                    for f in os.listdir(os.path.join(image_path, wave, scanner, plant))
                    if f.endswith(".png")
                ]
                sorted_img_list = sorted(img_list, key=lambda x: int(x.split(".")[0]))

                crop_folder = os.path.join(save_path, "crop", wave, scanner, plant)
                if not os.path.exists(crop_folder):
                    os.makedirs(crop_folder)

                for img in sorted_img_list:
                    frame = img.split(".")[0]
                    image = cv2.imread(
                        os.path.join(image_path, wave, scanner, plant, img)
                    )

                    # get the boundary layer
                    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    img_crop = img_gray[
                        startY : startY + height, startX + 50 : startX + width - 50
                    ]  # remove the left and right 50 pixels
                    img_crop_v = np.mean(img_crop, axis=1)
                    # Use np.convolve to calculate the moving average
                    moving_averages = np.convolve(
                        img_crop_v, np.ones(window_size) / window_size, mode="valid"
                    )
                    ind = (
                        np.argmin(moving_averages[150:-50])
                        + startY
                        + int(window_size / 2)
                        + 150
                    )  # filter out the first 150 rows and last 50 rows
                    data = np.reshape(
                        np.array([wave, scanner, plant, frame, ind]), (1, 5)
                    )
                    df_new = pd.DataFrame(data, columns=title)
                    dataframe = pd.concat([dataframe, df_new], ignore_index=True)
    ind_data = dataframe[1:]
    ind_data.to_csv(os.path.join(save_path, "output_layer_index.csv"), index=False)


# main
def main():
    parser = argparse.ArgumentParser(description="Segmentation Model Training Pipeline")
    parser.add_argument("--image_path", required=True, help="Training images path")
    parser.add_argument("--save_path", required=True, help="Segmentation path")
    parser.add_argument("--model_name", required=True, help="Training model name")

    args = parser.parse_args()

    image_path = args.image_path
    save_path = args.save_path
    model_name = args.model_name

    # add device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load best saved model checkpoint from the current run
    if os.path.exists(f"./{model_name}.pth"):
        best_model = torch.load(f"./{model_name}.pth", map_location=DEVICE)
        print("Loaded UNet model from this run.")
    else:
        raise ValueError("Model not available!")

    # segment crop images
    crop_folder = os.path.join(save_path, "crop")
    crop_img(image_path, crop_folder, save_path)

    # generate metedata file
    image_extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.gif",
    ]  # Add more extensions as needed

    # Use glob to find image files in the specified directory and its subdirectories
    subimage_list = []
    for extension in image_extensions:
        subimage_list.extend(glob.glob(f"{crop_folder}/**/{extension}", recursive=True))

    metadata_row = []
    for i in range(len(subimage_list)):
        image_path_i = subimage_list[i]
        label_path_i = subimage_list[i]
        metadata_row.append([str(i + 1), image_path_i, label_path_i])

    metadata_file = "./metadata_tem.csv"

    header = ["image_id", "image_path", "label_colored_path"]
    with open(metadata_file, "w") as csvfile:
        writer = csv.writer(csvfile, lineterminator="\n")
        writer.writerow([g for g in header])
        for x in range(len(metadata_row)):
            writer.writerow(metadata_row[x])
    print("Finish writing meta data!")

    # set up segmentation patch folder
    sample_preds_folder = os.path.join(save_path, "segment")
    if not os.path.exists(sample_preds_folder):
        os.mkdir(sample_preds_folder)
    files = os.listdir(sample_preds_folder)
    print(files)
    if len(files) > 0:
        for items in files:
            os.remove(os.path.join(sample_preds_folder, items))

    # setup model parameters
    ENCODER = "resnet101"
    ENCODER_WEIGHTS = "imagenet"
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    # check the color
    class_dict = pd.read_csv("./label_class_dict_lr.csv")
    class_names = class_dict["name"].tolist()
    class_rgb_values = class_dict[["r", "g", "b"]].values.tolist()

    select_classes = ["background", "root"]
    select_class_indices = [class_names.index(cls.lower()) for cls in select_classes]
    select_class_rgb_values = np.array(class_rgb_values)[select_class_indices]

    # setup dataset
    metadata_df = pd.read_csv("metadata_tem.csv")
    test_dataset = PredictionDataset(
        metadata_df,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        class_rgb_values=select_class_rgb_values,
    )
    test_dataset_vis = PredictionDataset(
        metadata_df,
        class_rgb_values=select_class_rgb_values,
    )

    print(len(test_dataset))
    # predict patch segmentation
    for idx in range(1):  # len(test_dataset)
        image, names = test_dataset[idx]
        subfolder = names.rsplit("/", 1)[0][5:]  # get the subfolders without 'crop/'
        # subfolder = names.rsplit('/',1)[0]
        if not os.path.exists(os.path.join(sample_preds_folder, subfolder)):
            os.makedirs(os.path.join(sample_preds_folder, subfolder))
        if np.mod(idx, 72) == 1:
            print(subfolder)
        img_name = names.rsplit("/", 1)[-1]
        image_vis = test_dataset_vis[idx][0].astype("uint8")
        true_dimensions = image_vis.shape
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        # Predict test image
        pred_mask = best_model(x_tensor)
        pred_mask = pred_mask.detach().squeeze().cpu().numpy()
        # Convert pred_mask from `CHW` format to `HWC` format
        pred_mask = np.transpose(pred_mask, (1, 2, 0))
        # Get prediction channel corresponding to foreground
        # pred_foreground_heatmap = crop_image(pred_mask[:,:,select_classes.index('root')], true_dimensions)['image']
        pred_mask = crop_image(
            colour_code_segmentation(
                reverse_one_hot(pred_mask), select_class_rgb_values
            ),
            true_dimensions,
        )["image"]
        pred_mask[np.all(pred_mask == [128, 0, 0], axis=-1)] = [255, 255, 255]
        cv2.imwrite(
            os.path.join(sample_preds_folder, subfolder, f"{img_name}.png"), pred_mask
        )
    print("Finish prediction!")


if __name__ == "__main__":
    main()
