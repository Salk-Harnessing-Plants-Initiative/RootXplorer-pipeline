# crop images, segmentation, and get layer index
# just change the image_path (folder with new images)
python pipeline_segment_cropped_cylinder.py --image_path GWAS_SCREEN_AUG23/Images/Raw_Images --save_path GWAS_SCREEN_AUG23_seg_v01 --model_name best_model_unet_plusplus_resnet101_cylinder_0124

# analyze the root penetration traits, outlier removal, box plots for waves
python pipeline_analysis.py --seg_path GWAS_SCREEN_AUG23_seg_v01/segment --save_path GWAS_SCREEN_AUG23_seg_v01/analysis --layer_index_csv GWAS_SCREEN_AUG23_seg_v01/output_layer_index.csv
