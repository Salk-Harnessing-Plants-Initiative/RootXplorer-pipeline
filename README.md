# SALK_projects

## Purpose and background
**Purpose**: segmenting cylinder images and extracting traits from segmentation results for Arabidopsis, rice, soybean, and sorghum.

**Background**: to be updated with manuscript. 

## Installation

1. **Clone the repository**:  
   Clone the repository to the local drive.
   ```
   git clone https://github.com/Salk-Harnessing-Plants-Initiative/cylinder-penetration.git
   ```
2. **Navigate to the cloned directory**:  
   
   ```
   cd cylinder-penetration
   ```
3. **Create new conda environments**:
   ```
   conda env create -f environment.yml
   ```
4. **Activate conda environment**
    ```
    conda activate segmentation-analysis
    ```

## Running the pipeline

1. **crop and segment crop images**:
   ```
   python pipeline_crop_segment.py --image_path ../Images_test_v03 --save_path ../Segmentation_v03_test  --model_name best_model_crop_cylinder_unetpp_resnet101_1024patch_1batch_40epoch_02_27
   ```
   Change the `image_path` (../Images_test_v03) to your folder name where you save the cylinder images;
   
   (optional) Change the `save_path` (../Images_test_v03) to a folder name where you'd like to save the cropped images and segmentation. The new folder will be created automatically, you don't have to create a new one by yourself.
   
   Change the `model_name` (best_model_crop_cylinder_unetpp_resnet101_1024patch_1batch_40epoch_02_27) if needed. Arabidopsis model is `best_model_unet_plusplus_resnet101_cylinder_0124`; rice model is `best_model_rice_seminal_cylinder_unetpp_resnet101_1024patch_4batch_100epoch_05_23`; soybean and sorghum model is `best_model_crop_cylinder_unetpp_resnet101_1024patch_1batch_40epoch_02_27`.

2. **get traits and remove outlier**:
   ```
   python pipeline_analysis.py --image_folder ../Segmentation_v03_test/crop --seg_folder ../Segmentation_v03_test/Segmentation --save_path ../Segmentation_v03_test/analysis --master_data_csv ../MasterData_May2024.csv --plant_group accession
   ```
   (optional) Change the `image_folder` (../Segmentation_v03_test/crop) to your folder name where you save the cropped cylinder images in previous step, it is the same one with `save_path` in previous step plus '/crop' subfolder;
   
   (optional) Change the `seg_folder` (../Segmentation_v03_test/Segmentation) to the folder name where you save the segmentation in previous step;
   
   (optional) Change the `save_path` (../Segmentation_v03_test/analysis) to the folder name where you'd like to save the original traits and traits after outlier removal;
   
   Change the `master_data_csv` (../MasterData_May2024.csv) to the master data file;
   
   Change the `plant_group` (accession) to the column in `master_data_csv` you'd like to remove outlier (based on accession or concentration).

