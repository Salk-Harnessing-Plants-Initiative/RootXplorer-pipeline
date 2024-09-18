# crop sorghum images
# python pipeline_crop.py --image_folder ../images --save_path ../Segmentation/crop

# segment crop images (sorghum)
python pipeline_crop_segment.py --image_path ../Images_test --save_path ../Segmentation_v02_test  --model_name best_model_crop_cylinder_unetpp_resnet101_1024patch_1batch_40epoch_02_27

# analysis
python pipeline_analysis.py --image_folder ../Segmentation_v02_test/crop --seg_folder ../Segmentation_v02_test/Segmentation --save_path ../Segmentation_v02_test/analysis

# viz the traits with cropped images
python pipeline_viz_traits.py

