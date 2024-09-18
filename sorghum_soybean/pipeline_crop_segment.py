# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 15:46:18 2023

@author: linwang
"""

# %% import library
import os, cv2, math, csv
import numpy as np
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import warnings

warnings.filterwarnings("ignore")

import torch
import albumentations as album
import argparse

import segmentation_models_pytorch as smp

import segmentation_models_pytorch.utils


# %% add pading and crop images to patch size
def crop_images_folder(bbox, image_folder, save_path):
    images = [
        os.path.relpath(os.path.join(root, file), image_folder)
        for root, _, files in os.walk(image_folder)
        for file in files
        if (file.endswith(".PNG") or file.endswith(".png")) and not file.startswith(".")
    ]
    for image_name in images:
        image_path = os.path.join(image_folder, image_name)
        image = cv2.imread(image_path)
        print(f"cropping image: {image_path}")
        if "Fast" in image_path:
            bbox2 = bbox["Fast"]
            startX, startY, width, height = bbox2
        else:
            bbox2 = bbox["Slow"]
            startX, startY, width, height = bbox2
        # startX, startY, width, height = bbox
        new_image = image[startY : startY + height, startX : startX + width, :]

        # save new_image
        # print(image_name.split("/")[:-1])
        save_folder = os.path.join(save_path, "/".join(image_name.split("/")[:-1]))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        new_name = os.path.join(save_path, image_name)
        cv2.imwrite(new_name, new_image)


def crop_image(image, true_dimensions):
    return album.CenterCrop(p=1, height=true_dimensions[0], width=true_dimensions[1])(
        image=image
    )


# %% viz
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


# %% prediction dataset
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
        names = self.image_paths[i].rsplit("/", 1)[-1].split(".")[0]
        path_name = self.image_paths[i]

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image)
            image = sample["image"]

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample["image"]

        return image, names, path_name

    def __len__(self):
        # return length of
        return len(self.image_paths)


def get_training_augmentation():
    train_transform = [
        # album.PadIfNeeded(min_height=550, min_width=660, always_apply=True, border_mode=0),
        # LW height and width change from 832 to 1000 to 1984 to 2720
        # album.RandomCrop(height=1024, width=1024, always_apply=True),
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
            min_height=256, min_width=256, always_apply=True, border_mode=0
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


def create_clear_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

    files_temp = [
        os.path.relpath(os.path.join(root, file), folder)
        for root, _, files in os.walk(folder)
        for file in files
        if (file.endswith(".PNG") or file.endswith(".png")) and not file.startswith(".")
    ]
    if len(files_temp) > 0:
        for items in files_temp:
            os.remove(os.path.join(folder, items))


def generate_metafile(image_path_crop, label_path_crop):
    subimage_list = [
        os.path.relpath(os.path.join(root, file), image_path_crop)
        for root, _, files in os.walk(image_path_crop)
        for file in files
        if (file.endswith(".PNG") or file.endswith(".png")) and not file.startswith(".")
    ]

    metadata_row = []
    for i in range(len(subimage_list)):
        image_path_i = os.path.join(image_path_crop, subimage_list[i])
        label_path_i = os.path.join(label_path_crop, subimage_list[i])
        metadata_row.append([str(i + 1), image_path_i, label_path_i])

    metadata_file = "./metadata_tem.csv"

    header = ["image_id", "image_path", "label_colored_path"]
    with open(metadata_file, "w") as csvfile:
        writer = csv.writer(csvfile, lineterminator="\n")
        writer.writerow([g for g in header])
        for x in range(len(metadata_row)):
            writer.writerow(metadata_row[x])
    return metadata_file


# %% main
def main():
    parser = argparse.ArgumentParser(description="Segmentation Model Training Pipeline")
    parser.add_argument("--patch_size", default=1024, help="Cropped image patch size")
    parser.add_argument("--image_path", required=True, help="Training images path")
    parser.add_argument("--save_path", required=True, help="Segmentation path")
    parser.add_argument("--model_name", required=True, help="Training model name")

    args = parser.parse_args()

    patch_size = args.patch_size
    # overlap_size = args.overlap_size
    image_path = args.image_path
    save_path = args.save_path
    model_name = args.model_name

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load best saved model checkpoint from the current run
    if os.path.exists(f"{model_name}.pth"):
        best_model = torch.load(f"{model_name}.pth", map_location=DEVICE)
        print("Loaded UNet model from this run.")
    else:
        raise ValueError("Model not available!")

    # crop images
    # bbox = (617, 190, 970, 850)  # (590, 56, 1024, 1024) crop images
    bbox = {"Fast": (350, 56, 1024, 1024), "Slow": (520, 56, 1024, 1024)}
    image_path_crop = os.path.join(save_path, "crop")
    # create_clear_folder(image_path_crop)
    crop_images_folder(bbox, image_path, image_path_crop)

    # generate metedata file
    metadata_file = generate_metafile(image_path_crop, image_path_crop)

    # set up segmentation patch folder
    sample_preds_folder = os.path.join(save_path, "Segmentation")
    create_clear_folder(sample_preds_folder)

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
    metadata_df = pd.read_csv(metadata_file)
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

    print(f"{len(test_dataset)} images in metadata.")

    # predict patch segmentation
    for idx in range(len(test_dataset)):  # len(test_dataset)
        image, names, path_name = test_dataset[idx]
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
        # get the subfolder name
        path_parts = path_name.split("/")
        index = path_parts.index("crop")
        subfolder = "/".join(path_parts[index + 1 : -1])

        save_path = os.path.join(sample_preds_folder, subfolder)
        print(f"predicting : {save_path}")

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cv2.imwrite(os.path.join(save_path, f"{names}.png"), pred_mask)
    print("Finish prediction!")


if __name__ == "__main__":
    main()
