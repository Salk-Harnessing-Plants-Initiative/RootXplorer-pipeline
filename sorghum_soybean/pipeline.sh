# crop sorghum images
# python pipeline_crop.py --image_folder ../images --save_path ../Segmentation/crop

# segment crop images (sorghum)
python pipeline_crop_segment.py --image_path ../Images --save_path ../Segmentation_v01  --model_name best_model_crop_cylinder_unetpp_resnet101_1024patch_1batch_40epoch_02_27

# analysis
# python pipeline_analysis.py --image_folder ../Segmentation/Crop_demo --seg_folder ../segment_crop_v01 --save_path ../segment_crop_v01_analysis
# python pipeline_analysis.py --image_folder ../Segmentation/Crop --seg_folder ../segment_crop_v01 --save_path ../segment_crop_v01_analysis
python pipeline_analysis.py --image_folder ../Segmentation_v01/crop --seg_folder ../Segmentation_v01/Segmentation --save_path ../Segmentation_v01/analysis

# viz the traits with cropped images
python pipeline_viz_traits.py

