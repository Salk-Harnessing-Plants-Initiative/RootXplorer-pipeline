import os
import pandas as pd
import numpy as np
import cv2
import argparse
import seaborn as sns
from scipy import stats

import matplotlib.pyplot as plt


def cylinder_penetrate_analysis(image_path, layer_index_csv, save_path):
    layer_index = pd.read_csv(layer_index_csv)
    wave_list = [
        f for f in os.listdir(image_path) if os.path.isdir(os.path.join(image_path, f))
    ]
    layer_boundary_threshold = 20
    root_count_depth = 5

    title = [
        "wave",
        "scanner",
        "plant",
        "frame",
        "upper_root_area",
        "upper_total_area",
        "upper_root_count",
        "bottom_root_area",
        "bottom_total_area",
        "bottom_root_count",
    ]
    data = np.zeros([1, len(title)])
    dataframe = pd.DataFrame(data, columns=title)

    for wave in wave_list:
        scanner_list = [
            f
            for f in os.listdir(os.path.join(image_path, wave))
            if os.path.isdir(os.path.join(image_path, wave, f))
        ]

        for scanner in scanner_list:
            plant_list = [
                f
                for f in os.listdir(os.path.join(image_path, wave, scanner))
                if os.path.isdir(os.path.join(image_path, wave, scanner, f))
            ]

            top = (
                180 if scanner.endswith("SlowScanner") else 219
            )  # based on the 1030 Y for slowscanner, 1069Y for FastScannner
            left = 526 if scanner.endswith("SlowScanner") else 386
            height = 850
            width = 970
            for plant in plant_list:
                print(f"wave: {wave}, scanner: {scanner}, plant: {plant}")
                img_list = [
                    f
                    for f in os.listdir(os.path.join(image_path, wave, scanner, plant))
                    if f.endswith(".png")
                ]
                sorted_img_list = sorted(img_list, key=lambda x: int(x.split(".")[0]))

                for img in sorted_img_list:
                    frame = img.split(".")[0]
                    image = cv2.imread(
                        os.path.join(image_path, wave, scanner, plant, img)
                    )

                    img_loc = (
                        (layer_index["wave"] == wave)
                        & (layer_index["scanner"] == scanner)
                        & (layer_index["plant"] == plant)
                        & (layer_index["frame"] == int(frame))
                    )
                    layer_boundary = layer_index.loc[img_loc, "layer_ind"].iloc[0] - top

                    # get the upper and bottom layer
                    upper_layer = image[: layer_boundary - layer_boundary_threshold, :]

                    value, count = np.unique(upper_layer[:, :, 0], return_counts=True)
                    if len(count) > 1:
                        upper_root = count[1]
                        upper_total = count[0] + count[1]
                    else:
                        upper_root = 0
                        upper_total = count[0]

                    # get upper layer root count
                    upper_layer_count = upper_layer[-root_count_depth:, :, 0]

                    contours, stats = cv2.findContours(
                        upper_layer_count, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    if len(contours) > 0:
                        contour_areas = [
                            cv2.contourArea(contour) for contour in contours
                        ]
                        upper_root_count = (
                            len(contours)
                            if np.max(contour_areas) < upper_layer_count.size
                            else len(contours) - 1
                        )
                    else:
                        upper_root_count = 0

                    # get the bottom layer and statistics
                    bottom_layer = image[layer_boundary + layer_boundary_threshold :, :]

                    value, count = np.unique(bottom_layer[:, :, 0], return_counts=True)
                    if len(count) > 1:
                        bottom_root = count[1]
                        bottom_total = count[0] + count[1]
                    else:
                        bottom_root = 0
                        bottom_total = count[0]

                    # get bottom layer root count
                    bottom_layer_count = bottom_layer[:root_count_depth, :, 0]

                    contours, stats = cv2.findContours(
                        bottom_layer_count, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    if len(contours) > 0:
                        contour_areas = [
                            cv2.contourArea(contour) for contour in contours
                        ]
                        bottom_root_count = (
                            len(contours)
                            if np.max(contour_areas) < bottom_layer_count.size
                            else len(contours) - 1
                        )
                    else:
                        bottom_root_count = 0

                    # save data to the dataframe
                    data = np.reshape(
                        np.array(
                            [
                                wave,
                                scanner,
                                plant,
                                frame,
                                upper_root,
                                upper_total,
                                upper_root_count,
                                bottom_root,
                                bottom_total,
                                bottom_root_count,
                            ]
                        ),
                        (1, len(title)),
                    )
                    df_new = pd.DataFrame(data, columns=title)
                    dataframe = pd.concat([dataframe, df_new], ignore_index=True)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    result_data = dataframe[1:]
    result_data.to_csv(
        os.path.join(save_path, "arab_cylinder_results.csv"), index=False
    )


def get_statistics(save_path):
    # import csv data
    data_path = os.path.join(save_path, "arab_cylinder_results.csv")
    # r"Y:\Lin_Wang\Ara_cylinder\root_penetrate\GWAS_SCREEN_AUG23_test\analysis\arab_cylinder_results.csv"
    data = pd.read_csv(data_path)

    # add ratio of root area and root count
    data["root_area_ratio"] = data["bottom_root_area"] / data["upper_root_area"]
    data["root_count_ratio"] = data["bottom_root_count"] / data["upper_root_count"]
    # data.columns

    data = data[~data.isin([np.nan, np.inf, -np.inf]).any(axis=1)]
    # data.to_csv(os.path.join(save_path,'arab_cylinder_results_removeoutlier.csv'),index=False)
    data_plant = (
        data.groupby(["wave", "scanner", "plant"])[
            ["root_area_ratio", "root_count_ratio"]
        ]
        .mean()
        .reset_index()
    )

    # data_plant.columns
    # remove outlier
    data_wave = data_plant.groupby("wave")
    filtered_df = pd.DataFrame()
    z_score_threshold = 2
    for name, group in data_wave:
        # Calculate the z-scores for 'root_count_ratio' and 'root_area_ratio' columns within each wave
        z_scores_count = np.abs(stats.zscore(group["root_count_ratio"]))
        z_scores_area = np.abs(stats.zscore(group["root_area_ratio"]))

        # Create boolean masks to filter out the outliers for each column
        outlier_mask_count = z_scores_count <= z_score_threshold
        outlier_mask_area = z_scores_area <= z_score_threshold

        # Combine the outlier masks for both columns using the logical AND operation
        combined_outlier_mask = outlier_mask_count & outlier_mask_area

        # Append the non-outlier rows to the filtered DataFrame
        filtered_df = pd.concat([filtered_df, group[combined_outlier_mask]])
    filtered_df["wave_num"] = filtered_df["wave"].str.extract("(\d+)").astype(int)
    filtered_df = (
        filtered_df.sort_values(by="wave_num")
        .drop("wave_num", axis=1)
        .reset_index(drop=True)
    )
    filtered_df.to_csv(
        os.path.join(save_path, "arab_cylinder_results_removeoutlier.csv"), index=False
    )

    # box plot of wave
    plt.figure(figsize=(10, 6))  # Optional: set the figure size
    sns.boxplot(x="wave", y="root_count_ratio", hue="wave", data=filtered_df)
    sns.stripplot(
        x="wave",
        y="root_count_ratio",
        data=filtered_df,
        color="black",
        size=3,
        jitter=True,
    )
    plt.savefig(os.path.join(save_path, "root_count_ratio.png"))

    plt.figure(figsize=(10, 6))  # Optional: set the figure size
    sns.boxplot(x="wave", y="root_area_ratio", hue="wave", data=filtered_df)
    sns.stripplot(
        x="wave",
        y="root_area_ratio",
        data=filtered_df,
        color="black",
        size=3,
        jitter=True,
    )
    plt.savefig(os.path.join(save_path, "root_area_ratio.png"))

    statistics = (
        filtered_df.groupby("wave")[["root_count_ratio", "root_area_ratio"]]
        .agg(["mean", "median", "std", "min", "max"])
        .reset_index()
    )
    statistics.to_csv(
        os.path.join(save_path, "arab_cylinder_results_summary.csv"), index=False
    )


def main():
    parser = argparse.ArgumentParser(
        description="Arab cylinder root penetration analysis"
    )
    parser.add_argument("--seg_path", required=True, help="Segmentation images path")
    parser.add_argument("--save_path", required=True, help="Result save path")
    parser.add_argument(
        "--layer_index_csv", required=True, help="csv file with boundary index"
    )

    args = parser.parse_args()

    seg_path = args.seg_path
    save_path = args.save_path
    layer_index_csv = args.layer_index_csv

    cylinder_penetrate_analysis(seg_path, layer_index_csv, save_path)
    get_statistics(save_path)


if __name__ == "__main__":
    main()
