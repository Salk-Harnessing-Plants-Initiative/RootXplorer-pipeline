import pandas as pd
import os
import numpy as np
from scipy import stats
import argparse


def check_plant_72frames(data, output_dir, count_name, count_wave_name):
    """Check whether 72 frames for each plant/cylinder."""
    # get the count for each plant/cylinder
    plant_count = data["plant"].value_counts().reset_index()
    # get the plant with counts not equal to 72
    plant_count.columns = ["plant", "count"]
    plant_count_not72 = pd.DataFrame(plant_count[plant_count["count"] != 72])

    # get the plants with wave and traits
    print(f"plant_count_not72: {plant_count_not72['plant'].dtype}")
    print(f"data: {data['plant'].dtype}")
    plant_count_not72_wave_all = pd.merge(
        plant_count_not72, data, how="inner", on="plant"
    )  # inner
    count_all_path = os.path.join(output_dir, count_name + ".csv")
    plant_count_not72_wave_all.to_csv(count_all_path, index=False)

    # get plant barcode and wave without duplications of 72 frames
    plant_count_not72_wave = plant_count_not72_wave_all.drop_duplicates(
        subset=["plant", "wave"], keep="first"
    ).loc[:, :"wave"]
    # count_plant_wave_path = os.path.join(output_dir, count_wave_name + ".csv")
    # plant_count_not72_wave.to_csv(count_plant_wave_path, index=False)
    print(f"{len(plant_count_not72_wave)} plants should be removed!")

    return plant_count_not72_wave, plant_count_not72_wave_all


def drop_w8_day20(data, output_dir):
    removed_data = data[data["scanner"] == "Day20_10-11-2023_FastScanner"]
    dropped_data = data[data["scanner"] != "Day20_10-11-2023_FastScanner"]
    print(f"dropped {len(removed_data)} plants")
    return dropped_data


def drop_dup_cylinders(data):
    new_data = data[data["scanner"] != "Day_20_10-4-2023_SlowScanner"]
    new_data = new_data[
        ~(
            (new_data["scanner"] == "Day_20_9-6-23_FastScanner")
            & (new_data["plant"] == "K84TKXC1YK")
        )
    ]
    return new_data


def remove_frames_great72(data, write_csv, output_dir):
    """Remove images with frame index more than 72."""
    remove = data["frame"] > 72
    new_data = data[~remove]
    if write_csv:
        csv_path = os.path.join(output_dir, "data_no_duplications.csv")
        new_data.to_csv(csv_path, index=False)
    return new_data


def remove_frame_outlier_0_upper(data, write_csv, output_dir):
    """Remove frames with 0 upper_root_count."""
    filter = data["upper_root_count"] == 0
    removed = data[filter]
    print(f"Removed {len(removed)} frames with 0 upper layer")
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
    print(f"Removed {len(df_removed)} frames with less than 0.5 has 0 bottom counts")

    # save the filtered data
    filtered_path = os.path.join(output_dir, "filtered_72frames_0upper_0bottom.csv")
    df_filtered.to_csv(filtered_path, index=False)

    # save the removed data
    removed_path = os.path.join(output_dir, "removed_0bottom.csv")
    df_removed.to_csv(removed_path, index=False)
    return df_filtered, df_removed


def get_ratios(data, output_dir):
    data_copy = data.copy()
    data_copy["root_area_ratio"] = data["bottom_root_area"] / data["upper_root_area"]
    data_copy["root_count_ratio"] = data["bottom_root_count"] / data["upper_root_count"]
    data = data_copy

    # columns_to_drop = data.columns[
    #     data.columns.get_loc("frame") : data.columns.get_loc("bottom_root_count") + 1
    # ]

    # data.drop(columns=columns_to_drop, inplace=True)
    data.to_csv(os.path.join(output_dir, "Original_ratios.csv"), index=False)
    return data


def remove_outlier_trait_std(data, group_by, frame_plant, trait_name, output_dir):
    plant_df = data.groupby(group_by)
    # filtered_df = pd.DataFrame()
    z_score_threshold = 2
    filtered_df = pd.DataFrame()
    removed_df = pd.DataFrame()
    for name, group in plant_df:
        # Calculate the z-scores the trait
        z_scores = np.abs(stats.zscore(group[trait_name]))
        # Create boolean masks to filter out the outliers for each column
        outlier_mask = z_scores <= z_score_threshold
        # Append the non-outlier rows to the filtered DataFrame
        filtered_df = pd.concat([filtered_df, group[outlier_mask]])
        removed_df = pd.concat([removed_df, group[~outlier_mask]])

    filtered_df.to_csv(
        os.path.join(output_dir, f"final_filtered_{frame_plant}_{trait_name}.csv"),
        index=False,
    )
    removed_df.to_csv(
        os.path.join(output_dir, f"removed_{frame_plant}_{trait_name}.csv"),
        index=False,
    )

    # get groupby columns
    # first_group = next(iter(plant_df.groups.keys()))
    # group_indices = plant_df.groups[first_group]
    # group_columns = plant_df.obj.columns[group_indices]

    for name, group in plant_df:
        # get the average value
        if type(group_by) == str:
            exclude_columns = [group_by, trait_name]
        else:
            exclude_columns = group_by + [trait_name]
        drop_columns = [col for col in group.columns if col not in exclude_columns]
        # drop_columns = [
        #     "plant",
        #     "scanner",
        #     "wave",
        #     "ID",
        #     "Name",
        #     "R",
        # ]
        grouped_columns = [trait_name]
        aggregation_dict = {
            col: "mean" for col in grouped_columns if col not in drop_columns
        }
        for col in drop_columns:
            if col not in trait_name and col not in group_by:
                aggregation_dict[col] = "first"
        # aggregation_dict["scanner"] = "first"
        # aggregation_dict["wave"] = "first"
        # aggregation_dict["ID"] = "first"
        # aggregation_dict["Name"] = "first"
        # aggregation_dict["R"] = "first"
        plant_mean_df = (
            filtered_df.groupby(group_by).agg(aggregation_dict).reset_index()
        )

    plant_mean_df.to_csv(
        os.path.join(output_dir, f"{frame_plant}_mean_{trait_name}.csv"), index=False
    )
    return filtered_df, plant_mean_df


def main():
    parser = argparse.ArgumentParser(description="Crop images")
    parser.add_argument("--indexing_csv", required=True, help="original image path")
    args = parser.parse_args()
    indexing_csv = args.indexing_csv

    # CHANGE to your data folder (where the original csv file located)
    data_dir = "C:/Users/linwang/Box/Work/3_Root_penetration/remove_outlier/code_data"
    data_path = os.path.join(data_dir, "arab_cylinder_results.csv")
    # import original dataset
    data = pd.read_csv(data_path)
    print(f"data.shape: {data.shape}")
    print(f"data columns: {data.columns}")

    # CHANGE to your folder where you'd like to save the filtered data
    output_dir = (
        "C:/Users/linwang/Box/Work/3_Root_penetration/remove_outlier/code_data/output"
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # check whether 72 frames per plant/cylinder
    count_wave_name = "plants_not_72frames"
    count_name = "plants_not_72frames_all"
    check_plant_72frames(data, output_dir, count_name, count_wave_name)

    # drop wave 8
    data_drop8 = drop_w8_day20(data, output_dir)

    # remove frames greater than 72
    write_csv = False  # save the filtered data
    remove_72frames = remove_frames_great72(data_drop8, write_csv, output_dir)

    # drop duplicated cylinders
    new_data = drop_dup_cylinders(remove_72frames)

    # check whether 72 frames per plant/cylinder AGAIN
    data = new_data
    count_wave_name = "plants_filtered"
    count_name = "plants_filtered_frames"
    check_plant_72frames(data, output_dir, count_name, count_wave_name)

    # remove frames greater than 72
    write_csv = True  # save the filtered data
    remove_72frames = remove_frames_great72(data, write_csv, output_dir)

    # merge the barcode with accession name
    indexing = pd.read_csv(os.path.join(data_dir, indexing_csv))
    remove_72frames = remove_72frames.merge(
        indexing, left_on="plant", right_on="Barcode"
    )
    remove_72frames.to_csv(
        os.path.join(output_dir, "Original_col_data.csv"), index=False
    )

    # print the shape of outlier removing
    print("=========================================================")
    print(
        f"Before removing outliers, the dataset has {remove_72frames.shape[0]} frames"
    )
    # get the ratios
    ratios = get_ratios(remove_72frames, output_dir)

    # remove frames with 0 upper_root_count
    write_csv = True  # save the filtered data
    remove_0 = remove_frame_outlier_0_upper(ratios, write_csv, output_dir)

    # remove outliers for less than a threshold with 0 bottom_root_count.
    # the default threshold is 50% (0.5)
    # CHANGE the threshold if needed
    threshold = 0.5
    df_filtered, df_removed = remove_frame_outlier_0_bottom(
        remove_0, threshold, output_dir
    )

    # remove the frames out of 2 std of each barcode
    group_by = "plant"
    trait_name = "root_area_ratio"
    frame_plant = "frames"
    area_filtered_df, area_plant_mean_df = remove_outlier_trait_std(
        df_filtered, group_by, frame_plant, trait_name, output_dir
    )
    removed_frames = len(df_filtered) - len(area_filtered_df)
    print(f"Removed {removed_frames} frames based on the 2 std of root_area_ratio")

    trait_name = "root_count_ratio"
    count_filtered_df, count_plant_mean_df = remove_outlier_trait_std(
        df_filtered, group_by, frame_plant, trait_name, output_dir
    )
    removed_frames = len(df_filtered) - len(count_filtered_df)
    print(f"Removed {removed_frames} frames based on the 2 std of root_count_ratio")

    # area_plant_mean_df = pd.read_csv(
    #     os.path.join(output_dir, "plants_mean_root_area_ratio.csv")
    # )
    # count_plant_mean_df = pd.read_csv(
    #     os.path.join(output_dir, "plants_mean_root_count_ratio.csv")
    # )

    # concat the two dataframes
    count_area_plant_mean_df = pd.merge(
        count_plant_mean_df.drop(columns="root_area_ratio"),
        area_plant_mean_df[["plant", "root_area_ratio"]],
        on="plant",
        how="outer",
    )

    # change the column order
    count_area_plant_mean_df = count_area_plant_mean_df[
        [
            "plant",
            "root_count_ratio",
            "root_area_ratio",
            "scanner",
            "wave",
            "ID",
            "Name",
            "R",
        ]
    ]
    count_area_plant_mean_df.to_csv(
        os.path.join(output_dir, "plant_mean_traits.csv"), index=False
    )

    # remove plant outlier per accession/wave
    # remove the plants out of 2 std of each barcode
    group_by = ["wave", "Name"]
    trait_name = "root_area_ratio"
    frame_plant = "plants"
    area_filtered_plant_df, area_accession_mean_df = remove_outlier_trait_std(
        count_area_plant_mean_df, group_by, frame_plant, trait_name, output_dir
    )
    removed_plants = len(count_area_plant_mean_df) - len(area_filtered_plant_df)
    print(f"Removed {removed_plants} plants based on the 2 std of root_area_ratio")
    # print(f"area_filtered_plant_df shape: {area_filtered_plant_df.shape}")

    trait_name = "root_count_ratio"
    count_filtered_plant_df, count_accession_mean_df = remove_outlier_trait_std(
        count_area_plant_mean_df, group_by, frame_plant, trait_name, output_dir
    )
    removed_plants = len(count_area_plant_mean_df) - len(count_filtered_plant_df)
    print(f"Removed {removed_plants} plants based on the 2 std of root_count_ratio")

    ## merge filtered area and count
    count_area_plant_df = pd.merge(
        count_filtered_plant_df.drop(columns="root_area_ratio"),
        area_filtered_plant_df[["plant", "root_area_ratio"]],
        on="plant",
        how="outer",
    )

    map_traits = ["scanner", "wave", "ID", "Name", "R"]
    for trait in map_traits:
        count_area_plant_df[trait] = count_area_plant_df[trait].fillna(
            count_area_plant_df["plant"].map(
                area_filtered_plant_df.set_index("plant")[trait]
            )
        )

    # change the column order
    count_area_plant_df = count_area_plant_df[
        [
            "plant",
            "root_count_ratio",
            "root_area_ratio",
            "scanner",
            "wave",
            "ID",
            "Name",
            "R",
        ]
    ]
    count_area_plant_df.to_csv(
        os.path.join(output_dir, "filtered_plant_mean_traits.csv"), index=False
    )

    ##  merge mean area and count
    count_area_accession_mean_df = pd.merge(
        count_accession_mean_df.drop(columns="root_area_ratio"),
        area_accession_mean_df[["plant", "root_area_ratio"]],
        on="plant",
        how="outer",
    )

    map_traits = ["scanner", "wave", "ID", "Name", "R"]
    for trait in map_traits:
        count_area_accession_mean_df[trait] = count_area_accession_mean_df[
            trait
        ].fillna(
            count_area_accession_mean_df["plant"].map(
                area_accession_mean_df.set_index("plant")[trait]
            )
        )

    # change the column order
    count_area_accession_mean_df = count_area_accession_mean_df[
        [
            "plant",
            "root_count_ratio",
            "root_area_ratio",
            "scanner",
            "wave",
            "ID",
            "Name",
            "R",
        ]
    ]
    count_area_accession_mean_df.to_csv(
        os.path.join(output_dir, "accession_mean_traits.csv"), index=False
    )


if __name__ == "__main__":
    main()
