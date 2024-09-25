# crop sorghum images
# python pipeline_crop.py --image_folder ../images --save_path ../Segmentation/crop

# crop and segment crop images (sorghum)
python pipeline_crop_segment.py --image_path ../Images_test --save_path ../Segmentation_v02_test  --model_name best_model_crop_cylinder_unetpp_resnet101_1024patch_1batch_40epoch_02_27
python pipeline_crop_segment.py --image_path ../Images_test_v03 --save_path ../Segmentation_v03_test  --model_name best_model_crop_cylinder_unetpp_resnet101_1024patch_1batch_40epoch_02_27

# analysis
python pipeline_analysis.py --image_folder ../Segmentation_v02_test/crop --seg_folder ../Segmentation_v02_test/Segmentation --save_path ../Segmentation_v02_test/analysis --master_data_csv ../MasterData_May2024.csv --plant_group accession
python pipeline_analysis.py --image_folder ../Segmentation_v03_test/crop --seg_folder ../Segmentation_v03_test/Segmentation --save_path ../Segmentation_v03_test/analysis --master_data_csv ../MasterData_May2024.csv --plant_group accession

# viz the traits with cropped images
python pipeline_viz_traits.py

