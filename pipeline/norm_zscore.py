import numpy as np
import pandas as pd
import os


def merge_traits(area_path, count_path):
    area = pd.read_csv(area_path)
    count = pd.read_csv(count_path)
    area = area[["plant", "root_area_ratio", "scanner", "wave", "ID", "Name"]]
    count = count[["plant", "root_count_ratio", "scanner", "wave", "ID", "Name"]]
    traits = pd.merge(
        area[["plant", "root_area_ratio"]], count, how="outer", on="plant"
    )

    # map the "scanner", "wave", "ID", "Name"
    map_traits = ["scanner", "wave", "ID", "Name"]
    for trait in map_traits:
        traits[trait] = traits[trait].fillna(
            traits["plant"].map(area.set_index("plant")[trait])
        )
    return traits


def get_zscore(traits_col, traits_wave, trait):
    # get the zscore of root count ratio
    avg = traits_col["root_count_ratio"].mean()
    std = traits_col["root_count_ratio"].std()
    print(f"root count - avg: {avg}, std: {std}")
    traits_wave_no_na = traits_wave.dropna(subset=["root_count_ratio"])
    traits_wave["zscore_root_count_ratio"] = (
        traits_wave_no_na["root_count_ratio"] - avg
    ) / std

    # get the zscore of root area ratio
    avg = traits_col["root_area_ratio"].mean()
    std = traits_col["root_area_ratio"].std()
    print(f"root area - avg: {avg}, std: {std}")
    traits_wave_no_na = traits_wave.dropna(subset=["root_area_ratio"])
    traits_wave["zscore_root_area_ratio"] = (
        traits_wave_no_na["root_area_ratio"] - avg
    ) / std

    return traits_wave


def norm_traits(traits_path, save_path):
    traits = pd.read_csv(traits_path)
    waves = traits["wave"].unique()
    waves = sorted(waves, key=lambda x: int(x[1:]))
    traits_waves = pd.DataFrame()
    for wave in waves:
        traits_wave = traits[traits["wave"] == wave]
        traits_col = traits_wave[traits_wave["Name"] == "Col-0"]
        trait = "root_count_ratio"
        traits_wave = get_zscore(traits_col, traits_wave, trait)
        traits_wave = traits_wave.sort_values(by=["Name", "R"])
        traits_wave = traits_wave.reset_index(drop=True)

        traits_waves = pd.concat([traits_waves, traits_wave], axis=0)
    save_name = os.path.join(save_path, "zscore_traits.csv")
    traits_waves.to_csv(save_name, index=False)


area_path = r"C:\Users\linwang\Box\Work\3_Root_penetration\4_normalize\final_filtered_plants_root_area_ratio.csv"
count_path = r"C:\Users\linwang\Box\Work\3_Root_penetration\4_normalize\final_filtered_plants_root_count_ratio.csv"

traits_path = r"C:\Users\linwang\Box\Work\3_Root_penetration\4_normalize\filtered_plant_mean_traits.csv"
save_path = r"C:\Users\linwang\Box\Work\3_Root_penetration\4_normalize"

# traits = merge_traits(area_path, count_path)
norm_traits(traits_path, save_path)
