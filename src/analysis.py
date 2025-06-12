import numpy as np
import pandas as pd
import argparse
import os
import cv2
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt


def get_layer_boundary(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    region1 = img_gray[:, -200:-100]  # Last 200 to last 100 columns
    region2 = img_gray[:, 100:200]  # Columns 100 to 200

    # Concatenate them side by side
    img_crop = np.hstack((region1, region2))

    gradient_y = cv2.Sobel(img_crop, cv2.CV_64F, 0, 1, ksize=5)
    gradient_avgy = np.mean(gradient_y, axis=1)

    top = 0  # the index from cropped location instead of original image
    start_filter_ind = 350
    ind = (
        np.argmax(gradient_avgy[start_filter_ind:-550]) + top + start_filter_ind
    )  # filter out the first 350 rows and last 550 rows
    return ind


def get_layer_boundary_fodler(image_folder, save_path):
    images = [
        os.path.relpath(os.path.join(root, file), image_folder)
        for root, _, files in os.walk(image_folder)
        for file in files
        if (file.endswith(".PNG") or file.endswith(".png")) and not file.startswith(".")
    ]

    ind_df = pd.DataFrame()
    for img in images:

        image_name = os.path.join(image_folder, img)
        plant = os.path.dirname(image_name)
        frame = os.path.splitext(os.path.basename(image_name))[0]

        if img.startswith("T0R") or img.startswith("T0.0R"):
            ind = np.nan
        else:
            image = cv2.imread(image_name)
            ind = get_layer_boundary(image)

        ind_df = pd.concat(
            [
                ind_df,
                pd.DataFrame(
                    {
                        "image_name": [img],
                        "plant": [plant],
                        "frame": [frame],
                        "layer_ind": [ind],
                    }
                ),
            ],
            ignore_index=True,
        )

    # replace the concentration of 0 frames with median value
    ind_df["layer_ind"] = ind_df["layer_ind"].fillna(ind_df["layer_ind"].median())

    csv_name = os.path.join(save_path, "layer_index.csv")
    ind_df.to_csv(csv_name, index=False)
    return ind_df


def get_area(seg_image, index_median, threshold_area):
    upper_layer = seg_image[170 : index_median - threshold_area, :, 0]
    value, count = np.unique(upper_layer[:, :], return_counts=True)
    upper_area = count[1] if len(count) > 1 else 0

    bottom_layer = seg_image[index_median + threshold_area : -5, :, 0]
    value, count = np.unique(bottom_layer[:, :], return_counts=True)
    bottom_area = count[1] if len(count) > 1 else 0
    return upper_area, bottom_area


def get_count(seg_image, index_median, threshold_area, threshold_count):
    upper_layer = seg_image[
        index_median - threshold_area - threshold_count : index_median - threshold_area,
        :,
        0,
    ]
    contours, stats = cv2.findContours(
        upper_layer, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) > 0:
        contour_areas = [cv2.contourArea(contour) for contour in contours]
        upper_root_count = (
            len(contours)
            if np.max(contour_areas) < upper_layer.size
            else len(contours) - 1
        )
    else:
        upper_root_count = 0

    bottom_layer = seg_image[
        index_median + threshold_area : index_median + threshold_area + threshold_count,
        :,
        0,
    ]
    contours, stats = cv2.findContours(
        bottom_layer, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) > 0:
        contour_areas = [cv2.contourArea(contour) for contour in contours]
        bottom_root_count = (
            len(contours)
            if np.max(contour_areas) < bottom_layer.size
            else len(contours) - 1
        )
    else:
        bottom_root_count = 0
    return upper_root_count, bottom_root_count


def get_statistics_frames(df_filtered, save_path):
    data = df_filtered
    data = data[~data.isin([np.nan, np.inf, -np.inf]).any(axis=1)]

    # filter out outliers > 2
    z_score_threshold = 2
    filtered_df_count = pd.DataFrame()
    filtered_df_area = pd.DataFrame()
    data_plant = data.groupby("plant")
    for name, group in data_plant:
        # Calculate the z-scores for 'root_count_ratio' and 'root_area_ratio' columns within each wave
        group = group.dropna()  # drop nan values
        # Identify all-zero rows for root_count_ratio and root_area_ratio
        zero_count_mask = group["root_count_ratio"] == 0
        zero_area_mask = group["root_area_ratio"] == 0

        # Calculate z-scores (only if standard deviation is not zero)
        if group["root_count_ratio"].std() == 0:
            outlier_mask_count = pd.Series([False] * len(group), index=group.index)
        else:
            z_scores_count = np.abs(stats.zscore(group["root_count_ratio"]))
            outlier_mask_count = z_scores_count <= z_score_threshold

        if group["root_area_ratio"].std() == 0:
            outlier_mask_area = pd.Series([False] * len(group), index=group.index)
        else:
            z_scores_area = np.abs(stats.zscore(group["root_area_ratio"]))
            outlier_mask_area = z_scores_area <= z_score_threshold

        # Combine the masks with the zero masks using logical OR
        final_mask_count = outlier_mask_count | zero_count_mask
        final_mask_area = outlier_mask_area | zero_area_mask

        # Append filtered data
        filtered_df_count = pd.concat([filtered_df_count, group[final_mask_count]])
        filtered_df_area = pd.concat([filtered_df_area, group[final_mask_area]])
    filtered_df_summary_count = (
        filtered_df_area.groupby("plant")[
            ["root_count_ratio", "upper_root_count", "bottom_root_count"]
        ]
        .agg(
            root_count_ratio=("root_count_ratio", "mean"),
            upper_root_count=("upper_root_count", "mean"),
            bottom_root_count=("bottom_root_count", "mean"),
            frame_number_count=("root_count_ratio", "size"),  # Count of each group
        )
        .reset_index()
    )
    filtered_df_summary_count = filtered_df_summary_count.rename(
        columns={"plant": "plant_path"}
    )

    filtered_df_summary_area = (
        filtered_df_area.groupby("plant")[
            ["root_area_ratio", "upper_area", "bottom_area"]
        ]
        .agg(
            root_area_ratio=("root_area_ratio", "mean"),
            upper_area=("upper_area", "mean"),
            bottom_area=("bottom_area", "mean"),
            frame_number_area=("root_area_ratio", "size"),  # Count of each group
        )
        .reset_index()
    )
    filtered_df_summary_area = filtered_df_summary_area.rename(
        columns={"plant": "plant_path"}
    )

    # combine the area and count
    filtered_df_summary = pd.merge(
        filtered_df_summary_count,
        filtered_df_summary_area,
        on="plant_path",
        how="outer",
    )

    filtered_df_summary.to_csv(
        os.path.join(save_path, "traits_filteredframes_summary.csv"), index=False
    )

    return filtered_df_count, filtered_df_area, filtered_df_summary


def get_statistics_plants(save_path, master_data, plant_group):
    data_path = os.path.join(save_path, "traits_filteredframes_summary.csv")
    data = pd.read_csv(data_path)

    # get plant name based on plant_path
    data["plant_name"] = data["plant_path"].apply(lambda x: x.split("/")[-1])

    # link the master data to get concentration or genotype/accession experimental design
    data = data.merge(
        master_data[["barcode", plant_group]],
        left_on="plant_name",
        right_on="barcode",
        how="left",
    )
    data = data.drop(columns="barcode")

    z_score_threshold = 2
    filtered_df_count = pd.DataFrame()
    filtered_df_area = pd.DataFrame()
    data_plant = data.groupby(plant_group)
    for name, group in data_plant:
        # Calculate the z-scores for 'root_count_ratio' and 'root_area_ratio' columns within each wave
        group = group.dropna()  # drop nan values
        # Calculate z-scores safely, avoiding errors if standard deviation is zero
        if group["root_count_ratio"].std() == 0:
            z_scores_count = pd.Series([0] * len(group), index=group.index)
        else:
            z_scores_count = np.abs(stats.zscore(group["root_count_ratio"]))

        if group["root_area_ratio"].std() == 0:
            z_scores_area = pd.Series([0] * len(group), index=group.index)
        else:
            z_scores_area = np.abs(stats.zscore(group["root_area_ratio"]))

        # Create masks: keep values within threshold OR values equal to 0
        outlier_mask_count = (z_scores_count <= z_score_threshold) | (
            group["root_count_ratio"] == 0
        )
        outlier_mask_area = (z_scores_area <= z_score_threshold) | (
            group["root_area_ratio"] == 0
        )

        # Filter and append to result DataFrames
        filtered_df_count = pd.concat([filtered_df_count, group[outlier_mask_count]])
        filtered_df_area = pd.concat([filtered_df_area, group[outlier_mask_area]])
    filtered_df_summary_count = (
        filtered_df_count.dropna(subset=["root_count_ratio"])
        .groupby(plant_group)[
            ["root_count_ratio", "upper_root_count", "bottom_root_count"]
        ]
        .agg(
            root_count_ratio_mean=("root_count_ratio", "mean"),
            upper_root_count_mean=("upper_root_count", "mean"),
            bottom_root_count_mean=("bottom_root_count", "mean"),
            plant_number_count=("root_count_ratio", "size"),  # Count of each group
        )
        .reset_index()
    )
    print(f"filtered_df_summary_count: {filtered_df_summary_count}")

    filtered_df_summary_area = (
        filtered_df_area.groupby(plant_group)[
            ["root_area_ratio", "upper_area", "bottom_area"]
        ]
        .agg(
            root_area_ratio_mean=("root_area_ratio", "mean"),
            upper_area_mean=("upper_area", "mean"),
            bottom_area_mean=("bottom_area", "mean"),
            plant_number_area=("root_area_ratio", "size"),  # Count of each group
        )
        .reset_index()
    )
    print(f"filtered_df_summary_area: {filtered_df_summary_area}")

    # save the filtered data: combine the area and count
    filtered_df = pd.merge(
        filtered_df_count[
            [
                "plant_path",
                "plant_name",
                "trt",
                "root_count_ratio",
                "upper_root_count",
                "bottom_root_count",
                "frame_number_count",
            ]
        ],
        filtered_df_area[
            [
                "plant_path",
                "plant_name",
                "trt",
                "root_area_ratio",
                "upper_area",
                "bottom_area",
                "frame_number_area",
            ]
        ],
        on="plant_name",
        how="outer",
    )

    # Fill missing values in plant_path and accession from filtered_df_area
    filtered_df["plant_path"] = filtered_df["plant_path_x"].fillna(
        filtered_df["plant_path_y"]
    )
    filtered_df["trt"] = filtered_df["trt_x"].fillna(filtered_df["trt_y"])

    # Drop unnecessary duplicate columns created during merge
    filtered_df.drop(
        columns=["plant_path_x", "plant_path_y", "trt_x", "trt_y"],
        inplace=True,
    )
    # change column order
    column_order = ["trt", "plant_name", "plant_path"] + [
        col
        for col in filtered_df.columns
        if col not in ["trt", "plant_name", "plant_path"]
    ]
    filtered_df = filtered_df[column_order]
    # row ordered by treatment
    filtered_df = filtered_df.sort_values(by="trt", ascending=True)
    filtered_df.to_csv(
        os.path.join(save_path, "traits_filteredplants.csv"), index=False
    )

    # combine the area and count
    filtered_df_summary = pd.merge(
        filtered_df_summary_count,
        filtered_df_summary_area,
        on=plant_group,
        how="outer",
    )
    filtered_df_summary.to_csv(
        os.path.join(save_path, "traits_filteredplants_summary.csv"), index=False
    )

    return filtered_df_summary_count, filtered_df_summary_area, filtered_df_summary


def get_traits(seg_folder, ind_df, save_path):
    traits_df = ind_df
    for i in range(len(ind_df)):
        # get layer index and image path
        image_path = os.path.join(seg_folder, ind_df["image_name"][i])
        seg_image = cv2.imread(image_path)
        index_frame = int(ind_df["layer_ind"][i])

        # get areas
        threshold_area = 50
        upper_area, bottom_area = get_area(seg_image, index_frame, threshold_area)
        # get counts
        threshold_count = 5
        upper_root_count, bottom_root_count = get_count(
            seg_image, index_frame, threshold_area, threshold_count
        )
        traits_df.at[i, "upper_area"] = upper_area
        traits_df.at[i, "bottom_area"] = bottom_area
        traits_df.at[i, "upper_root_count"] = upper_root_count
        traits_df.at[i, "bottom_root_count"] = bottom_root_count
        traits_df.at[i, "root_area_ratio"] = np.divide(bottom_area, upper_area)
        traits_df.at[i, "root_count_ratio"] = np.divide(
            bottom_root_count, upper_root_count
        )
    save_name = os.path.join(save_path, "traits.csv")
    traits_df.to_csv(save_name, index=False)
    return traits_df


def remove_frame_outlier_0_upper(data, write_csv, output_dir):
    """Remove frames with 0 upper_root_count."""
    filter = data["upper_root_count"] == 0
    removed = data[filter]
    new_data = data[~filter]
    if write_csv:
        csv_path = os.path.join(output_dir, "removed_0upper.csv")
        removed.to_csv(csv_path, index=False)
    return new_data


def remove_frame_outlier_0_bottom(data, threshold, output_dir):
    """Remove outliers for less than 50% with 0 bottom_root_count."""
    # Group by 'plant' and calculate the percentage of frames with value 0
    frame_count_with_zeros = (
        data[data["bottom_root_count"] == 0].groupby("plant")["frame"].count()
    )
    total_frame_count = data.groupby("plant")["frame"].count()
    percentage_zeros = frame_count_with_zeros.div(total_frame_count, fill_value=0)

    # Get the plants where less than a threshold of frames have value 0
    plants_to_remove = percentage_zeros[percentage_zeros < threshold].index

    # Remove rows for the identified plants
    df_filtered = data[
        ~((data["plant"].isin(plants_to_remove)) & (data["bottom_root_count"] == 0))
    ]
    df_removed = data[
        ~(~((data["plant"].isin(plants_to_remove)) & (data["bottom_root_count"] == 0)))
    ]

    # save the filtered data
    filtered_path = os.path.join(output_dir, "filtered_72frames_0upper_0bottom.csv")
    df_filtered.to_csv(filtered_path, index=False)

    # save the removed data
    removed_path = os.path.join(output_dir, "removed_0bottom.csv")
    df_removed.to_csv(removed_path, index=False)
    return df_filtered, df_removed


def main():
    parser = argparse.ArgumentParser(
        description="Traits extraction and analysis Pipeline"
    )
    parser.add_argument(
        "--experiment", required=True, help="Experimental design folder path"
    )

    args = parser.parse_args()

    experiment = args.experiment

    image_folder = os.path.join("./Segmentation", experiment, "crop")
    seg_folder = os.path.join("./Segmentation", experiment, "Segmentation")
    save_path = os.path.join("./Segmentation", experiment, "analysis")

    image_path = os.path.join("./images", experiment)
    master_data_csv = [
        file
        for file in os.listdir(image_path)
        if (file.endswith(".csv") and not file.startswith("."))
    ]
    master_data = pd.read_csv(os.path.join(image_path, master_data_csv[0]))
    plant_group = "trt"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # get the layer index of each cropped image
    ind_df = get_layer_boundary_fodler(image_folder, save_path)

    # get traits
    traits_df = get_traits(seg_folder, ind_df, save_path)

    # delete frames with 0 in upper layer
    write_csv = True  # save the filtered data
    remove_0 = remove_frame_outlier_0_upper(traits_df, write_csv, save_path)

    # remove outliers for less than a threshold with 0 bottom_root_count.
    # the default threshold is 50% (0.5)
    # CHANGE the threshold if needed
    threshold = 0.5
    df_filtered, df_removed = remove_frame_outlier_0_bottom(
        remove_0, threshold, save_path
    )

    # remove frame outliers based on frames of each plant
    filtered_df_count, filtered_df_area, filtered_df_summary = get_statistics_frames(
        df_filtered, save_path
    )

    # remove plant outliers based on concentration or genotype
    filtered_df_summary_count, filtered_df_summary_area, filtered_df_summary = (
        get_statistics_plants(save_path, master_data, plant_group)
    )


if __name__ == "__main__":
    main()
