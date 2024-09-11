# crop images and get layer index
# python crop_img.py --image_path ./col-0/Images --save_path ./col-0/Images_crop --demo_path ./col-0/Images_demo
# crop and segment the images
# python pipeline_segment_cropped_cylinder.py --image_path ./col-0/Images --save_path ./col-0/segment_v02_12W --model_name best_model_unet_plusplus_resnet101_cylinder_0124
python pipeline_segment_cropped_cylinder.py --image_path GWAS_SCREEN_AUG23/Images/Raw_Images --save_path GWAS_SCREEN_AUG23_seg_v01 --model_name best_model_unet_plusplus_resnet101_cylinder_0124
python pipeline_segment_cropped_cylinder.py --image_path GWAS_SCREEN_AUG23/Images/Raw_Images --save_path GWAS_SCREEN_AUG23_seg_v02 --model_name best_model_unet_plusplus_resnet101_cylinder_0124

# analyze the root penetration traits
# python pipeline_analysis.py --seg_path ./col-0/segment_v02_12W/segment --save_path ./col-0/segment_v02_12W/analysis --layer_index_csv ./col-0/Images_crop/output_layer_index.csv
python pipeline_analysis.py --seg_path GWAS_SCREEN_AUG23_seg_v01/segment --save_path GWAS_SCREEN_AUG23_seg_v01/analysis --layer_index_csv GWAS_SCREEN_AUG23_seg_v01/output_layer_index.csv

# remove the outliers
# remove outliers for col
python remove_frame_outlier.py --indexing_csv "barcode_accession_col.csv"
python remove_frame_outlier.py --indexing_csv "acc_cyl_wave_barcodes_gwas.csv"

