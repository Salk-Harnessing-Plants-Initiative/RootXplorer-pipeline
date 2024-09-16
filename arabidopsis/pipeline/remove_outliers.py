def remove_outlier_frames(traits_df, output_dir):
    plant_df = traits_df.groupby("plant")
    filtered_df = pd.DataFrame()
    z_score_threshold = 2
    for name, group in plant_df:
        # Calculate the z-scores for 'root_count_ratio' and 'root_area_ratio' columns within each wave
        z_scores_count = np.abs(stats.zscore(group["count_ratio"]))
        z_scores_area = np.abs(stats.zscore(group["area_ratio"]))

        # Create boolean masks to filter out the outliers for each column
        outlier_mask_count = z_scores_count <= z_score_threshold
        outlier_mask_area = z_scores_area <= z_score_threshold

        # Combine the outlier masks for both columns using the logical AND operation
        combined_outlier_mask = outlier_mask_count & outlier_mask_area

        # Append the non-outlier rows to the filtered DataFrame
        filtered_df = pd.concat([filtered_df, group[combined_outlier_mask]])

        # get the average value
        drop_columns = ["plant", "scan_ind_path", "frame"]
        grouped_columns = filtered_df.drop(columns=drop_columns).columns
        print(grouped_columns)
        aggregation_dict = {
            col: "mean" for col in grouped_columns if col not in drop_columns
        }
        aggregation_dict["scan_ind_path"] = "first"
        aggregation_dict["frame"] = "count"
        plant_mean_df = filtered_df.groupby("plant").agg(aggregation_dict).reset_index()
    filtered_df.to_csv(
        os.path.join(output_dir, "traits_filtered_frames.csv"), index=False
    )
    plant_mean_df.to_csv(
        os.path.join(output_dir, "traits_plants_mean.csv"), index=False
    )
    return filtered_df, plant_mean_df
