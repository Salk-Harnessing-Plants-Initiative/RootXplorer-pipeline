# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 10:01:37 2023

@author: linwang
"""

import os
import cv2
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

 
#%% batch crop and save crop  boundary and layer boundary       
image_path = "Y:/Lin_Wang/Ara_cylinder/root_penetrate/GWAS_SCREEN_AUG23_test/Images/Raw_Images"
save_path = "Y:/Lin_Wang/Ara_cylinder/root_penetrate/GWAS_SCREEN_AUG23_test/analysis"
demo_path = "Y:/Lin_Wang/Ara_cylinder/root_penetrate/GWAS_SCREEN_AUG23_test/images_location"

def seperate_layers_cylinder_penetrate(image_path,save_path,demo_path):
    wave_list = [f for f in os.listdir(image_path) if os.path.isdir(os.path.join(image_path, f))]
    
    layer_boundary_threshold = 20
        
    window_size = 5 # sliding windows for the moving average
    
    line_color = (0, 0, 128)  # Red color (BGR)
    line_thickness = 2
    
    title = ['wave','scanner','plant', 'frame','layer_ind']
    data = np.zeros([1,5])
    dataframe = pd.DataFrame(data, columns=title)
    
    for wave in wave_list:
        scanner_list = [f for f in os.listdir(os.path.join(image_path,wave)) if os.path.isdir(os.path.join(image_path, wave,f))]
        
        for scanner in scanner_list:
            plant_list = [f for f in os.listdir(os.path.join(image_path,wave, scanner)) if os.path.isdir(os.path.join(image_path, wave,scanner,f))]
            top = 180 if scanner.endswith('SlowScanner') else 219 # based on the 1030 Y for slowscanner, 1069Y for FastScannner
            left = 526 if scanner.endswith('SlowScanner') else 386
            height = 850
            width = 970
            for plant in plant_list:
                print(f"wave: {wave}, scanner: {scanner}, plant: {plant}")
                img_list = [f for f in os.listdir(os.path.join(image_path,wave, scanner, plant)) if f.endswith('.png')]
                sorted_img_list = sorted(img_list, key=lambda x: int(x.split('.')[0]))
                
                save_folder = os.path.join(save_path,wave, scanner, plant)
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                    
                demo_folder = os.path.join(demo_path,wave, scanner, plant)
                if not os.path.exists(demo_folder):
                    os.makedirs(demo_folder)
                
                for img in sorted_img_list:
                    frame = img.split('.')[0]
                    image = cv2.imread(os.path.join(image_path,wave, scanner, plant,img))
                    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    
                    img_crop = img_gray[top:top+height, left+50:left+width-50] # remove the left and right 50 pixels
                    img_crop_v = np.mean(img_crop,axis=1)
                    
                    # Use np.convolve to calculate the moving average
                    moving_averages = np.convolve(img_crop_v, np.ones(window_size)/window_size, mode='valid')
                    ind = np.argmin(moving_averages[150:-50]) + top + int(window_size/2) + 150 # filter out the first 150 rows and last 50 rows
                    
                    # Draw the horizontal line seperating the two gel layers
                    cv2.line(image, (0, ind), (img_gray.shape[1], ind), line_color, line_thickness)
                    
                    # Draw two boundary lines next to the seperating line
                    cv2.line(image, (0, ind - layer_boundary_threshold), (img_gray.shape[1], ind - layer_boundary_threshold), (0,128,128), line_thickness)
                    cv2.line(image, (0, ind + layer_boundary_threshold), (img_gray.shape[1], ind + layer_boundary_threshold), (0,128,128), line_thickness)
                    
                    # draw the bbox of cropping
                    start_point = (left, top)
                    end_point = (left+width, top+height)
                    cv2.rectangle(image, start_point, end_point, (128,0,0), line_thickness)
                    cv2.imwrite(os.path.join(demo_path,wave, scanner, plant,img),image)
                    
                    
                    data = np.reshape(np.array([wave, scanner, plant, frame, ind]),(1,5))
                    df_new = pd.DataFrame(data, columns=title)
                    dataframe = pd.concat([dataframe, df_new], ignore_index=True)
                
    ind_data = dataframe[1:]
    ind_data.to_csv(os.path.join(save_path,'output_layer_index.csv'),index=False)


seperate_layers_cylinder_penetrate(image_path,save_path, demo_path)

#%% overlay segmentation and original images
import os
import cv2
from fpdf import FPDF

image_path = "Y:/Lin_Wang/Ara_cylinder/root_penetrate/GWAS_SCREEN_AUG23_test/crop"
seg_path = "Y:/Lin_Wang/Ara_cylinder/root_penetrate/GWAS_SCREEN_AUG23_test/segment"
save_path = "Y:/Lin_Wang/Ara_cylinder/root_penetrate/GWAS_SCREEN_AUG23_test/segment_viz"

def overlay_seg(image_path, seg_path):
    transparency = 0.3
    target_pixel_value = [255, 255, 255]
    replacement_pixel_value = [128, 0, 0]

    wave_list = [f for f in os.listdir(image_path) if os.path.isdir(os.path.join(image_path, f))]
        
    for wave in wave_list:
        scanner_list = [f for f in os.listdir(os.path.join(image_path,wave)) if os.path.isdir(os.path.join(image_path, wave,f))]
        
        for scanner in scanner_list:
            plant_list = [f for f in os.listdir(os.path.join(image_path,wave, scanner)) if os.path.isdir(os.path.join(image_path, wave,scanner,f))]
            
            pdf_file = os.path.join(save_path,f"{wave}_{scanner}.pdf")
            custom_page_width = 210
            custom_page_height = int(210/970*850)
            pdf = FPDF()
            
            for plant in plant_list:
                print(f"wave: {wave}, scanner: {scanner}, plant: {plant}")
                img_list = [f for f in os.listdir(os.path.join(image_path,wave, scanner, plant)) if f.endswith('.png')]
                sorted_img_list = sorted(img_list, key=lambda x: int(x.split('.')[0]))
                
                save_folder = os.path.join(save_path,wave,scanner,plant)
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                
                for img in sorted_img_list:
                    frame = int(img.split('.')[0])
                    if frame==1:
                        image = cv2.imread(os.path.join(image_path,wave, scanner, plant,img))
                        if os.path.exists(os.path.join(seg_path,wave, scanner, plant,img)):
                            seg = cv2.imread(os.path.join(seg_path,wave, scanner, plant,img))
                            indices_to_replace = np.all(seg == target_pixel_value, axis=-1)
                            seg[indices_to_replace] = replacement_pixel_value
                            add_image = cv2.addWeighted(image,1-transparency,seg,transparency,0)
                        else:
                            add_image = image
                        path = os.path.join(save_folder,"overlay_"+img)
                        cv2.imwrite(path, add_image)
                        
                        # convert to pdf
                        pdf.add_page(orientation='L', format=(custom_page_height,custom_page_width))
                        pdf.image(path, x=0, y=0, w=custom_page_width, h=custom_page_height)
                        
                        pdf.set_text_color(255,0,255)
                        pdf.set_xy(10, int(custom_page_height - 20))  # Adjust the Y position for the text int(custom_page_height - 10)
                        pdf.set_font("Arial", size=12)
                        pdf.cell(0, 0, txt=f"wave: {wave},  scanner: {scanner},  plant: {plant},  frame: {frame}", align="L")
                        
            pdf.output(pdf_file)  
                        
overlay_seg(image_path, seg_path)

#%% viz in a PDF
from fpdf import FPDF

image_path = "Y:/Lin_Wang/Ara_cylinder/root_penetrate/GWAS_SCREEN_AUG23_test/images_location"

def viz_pdf(image_path):
    wave_list = [f for f in os.listdir(image_path) if os.path.isdir(os.path.join(image_path, f))]
        
    for wave in wave_list:
        scanner_list = [f for f in os.listdir(os.path.join(image_path,wave)) if os.path.isdir(os.path.join(image_path, wave,f))]
        
        for scanner in scanner_list:
            plant_list = [f for f in os.listdir(os.path.join(image_path,wave, scanner)) if os.path.isdir(os.path.join(image_path, wave,scanner,f))]
            
            
            pdf_file = os.path.join(image_path,f"{wave}_{scanner}.pdf")
            custom_page_width = 210
            custom_page_height = int(210/2048*1024)
            pdf = FPDF()
            
            
            for plant in plant_list:
                print(f"wave: {wave}, scanner: {scanner}, plant: {plant}")
                img_list = [f for f in os.listdir(os.path.join(image_path,wave, scanner, plant)) if f.endswith('.png')]
                sorted_img_list = sorted(img_list, key=lambda x: int(x.split('.')[0]))
                
                for img in sorted_img_list:
                    frame = int(img.split('.')[0])
                    if frame==1:
                        path = os.path.join(image_path,wave, scanner, plant,img)
                        # image = cv2.imread(os.path.join(image_path,wave, scanner, plant,img))
                        
                        # convert to pdf
                        pdf.add_page(orientation='L', format=(custom_page_height,custom_page_width))
                        pdf.image(path, x=0, y=0, w=custom_page_width, h=custom_page_height)
                        
                        pdf.set_text_color(255,0,255)
                        pdf.set_xy(10, int(custom_page_height - 20))  # Adjust the Y position for the text int(custom_page_height - 10)
                        pdf.set_font("Arial", size=12)
                        pdf.cell(0, 0, txt=f"wave: {wave},  scanner: {scanner},  plant: {plant},  frame: {frame}", align="L")
                        #pdf.ln(custom_page_height)
            pdf.output(pdf_file)                     

viz_pdf(image_path)


#%% analysis
# root area and root count

import os
import pandas as pd
import numpy as np
import cv2

import matplotlib.pyplot as plt

image_path = "Y:/Lin_Wang/Ara_cylinder/root_penetrate/GWAS_SCREEN_AUG23_test/segment"
save_path = "Y:/Lin_Wang/Ara_cylinder/root_penetrate/GWAS_SCREEN_AUG23_test/analysis"
layer_index_csv = "Y:/Lin_Wang/Ara_cylinder/root_penetrate/GWAS_SCREEN_AUG23_test/output_layer_index.csv"


def cylinder_penetrate_analysis(image_path,layer_index_csv,save_path):
    layer_index = pd.read_csv(layer_index_csv)
    wave_list = [f for f in os.listdir(image_path) if os.path.isdir(os.path.join(image_path, f))]
    layer_boundary_threshold = 20
    root_count_depth = 5
    
    title = ['wave','scanner','plant', 'frame','upper_root_area','upper_total_area',
             'upper_root_count','bottom_root_area','bottom_total_area','bottom_root_count']
    data = np.zeros([1,len(title)])
    dataframe = pd.DataFrame(data, columns=title)
        
    for wave in wave_list:
        scanner_list = [f for f in os.listdir(os.path.join(image_path,wave)) if os.path.isdir(os.path.join(image_path, wave,f))]
        
        for scanner in scanner_list:
            plant_list = [f for f in os.listdir(os.path.join(image_path,wave, scanner)) if os.path.isdir(os.path.join(image_path, wave,scanner,f))]
            
            top = 180 if scanner.endswith('SlowScanner') else 219 # based on the 1030 Y for slowscanner, 1069Y for FastScannner
            left = 526 if scanner.endswith('SlowScanner') else 386
            height = 850
            width = 970
            for plant in plant_list:
                print(f"wave: {wave}, scanner: {scanner}, plant: {plant}")
                img_list = [f for f in os.listdir(os.path.join(image_path,wave, scanner, plant)) if f.endswith('.png')]
                sorted_img_list = sorted(img_list, key=lambda x: int(x.split('.')[0]))
                
                for img in sorted_img_list:
                    frame = img.split('.')[0]
                    image = cv2.imread(os.path.join(image_path,wave, scanner, plant,img))
                    
                    img_loc = (layer_index['wave']==wave) & (layer_index['scanner']==scanner) & (layer_index['plant']==plant) & (layer_index['frame']==int(frame))
                    layer_boundary = layer_index.loc[img_loc,'layer_ind'].iloc[0] - top
                    
                    # get the upper and bottom layer
                    upper_layer = image[:layer_boundary - layer_boundary_threshold,:]
                                        
                    value,count = np.unique(upper_layer[:,:,0],return_counts=True)
                    if len(count)>1:
                        upper_root = count[1]
                        upper_total = count[0]+count[1]
                    else:
                        upper_root = 0
                        upper_total = count[0]
                    
                    # get upper layer root count
                    upper_layer_count = upper_layer[-root_count_depth:,:,0]
                    
                    contours, stats = cv2.findContours(upper_layer_count, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if len(contours)>0:
                        contour_areas = [cv2.contourArea(contour) for contour in contours]
                        upper_root_count = len(contours) if np.max(contour_areas)<upper_layer_count.size else len(contours) - 1
                    else:  
                        upper_root_count = 0
                    
                    # get the bottom layer and statistics
                    bottom_layer = image[layer_boundary + layer_boundary_threshold:,:]
                    
                    value,count = np.unique(bottom_layer[:,:,0],return_counts=True)
                    if len(count)>1:
                        bottom_root = count[1]
                        bottom_total = count[0]+count[1]
                    else:
                        bottom_root = 0
                        bottom_total = count[0]
                    
                    # get bottom layer root count
                    bottom_layer_count = bottom_layer[:root_count_depth,:,0]
                    
                    contours, stats = cv2.findContours(bottom_layer_count, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if len(contours)>0:
                        contour_areas = [cv2.contourArea(contour) for contour in contours]
                        bottom_root_count = len(contours) if np.max(contour_areas)<bottom_layer_count.size else len(contours) - 1
                    else:
                        bottom_root_count = 0
                    
                    # save data to the dataframe
                    data = np.reshape(np.array([wave, scanner, plant, frame, upper_root,
                                                upper_total,upper_root_count,bottom_root,
                                                bottom_total,bottom_root_count]),(1,len(title)))
                    df_new = pd.DataFrame(data, columns=title)
                    dataframe = pd.concat([dataframe, df_new], ignore_index=True)
    result_data = dataframe[1:]
    result_data.to_csv(os.path.join(save_path,'arab_cylinder_results.csv'),index=False)           
        

cylinder_penetrate_analysis(image_path,layer_index_csv,save_path)
                    

#%% arab penetration analysis
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


def remove_outlier(df,column_name):
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define a lower and upper bound for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Detect and remove outliers
    # outliers = df[(df[column_name] < lower_bound) | (df[column_name] > upper_bound)]
    df_no_outliers = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]
    return df_no_outliers


# import csv data
data_path = r"Y:\Lin_Wang\Ara_cylinder\root_penetrate\GWAS_SCREEN_AUG23_test\analysis\arab_cylinder_results.csv"
data = pd.read_csv(data_path)

# add ratio of root area and root count
data['root_area_ratio'] = data['bottom_root_area'] / data['upper_root_area']
data['root_count_ratio'] = data['bottom_root_count'] / data['upper_root_count']
data.columns

data = data[~data.isin([np.nan, np.inf, -np.inf]).any(axis=1)]
data_plant = data.groupby(['wave', 'scanner', 'plant'])['root_area_ratio','root_count_ratio'].mean().reset_index()

data_plant.columns
# remove outlier
data_plant_remove_outlier = remove_outlier(data_plant,'root_area_ratio')
data_plant_remove_outlier = remove_outlier(data_plant_remove_outlier,'root_count_ratio')


# box plot of wave, scanner
plt.figure(figsize=(8, 6))  # Optional: set the figure size
sns.boxplot(x='wave', y='root_count_ratio', data=data_plant_remove_outlier)

plt.figure(figsize=(8, 6))  # Optional: set the figure size
sns.boxplot(x='wave', y='root_area_ratio', data=data_plant_remove_outlier)

statistics = data_plant_remove_outlier.groupby('wave')[['root_count_ratio', 'root_area_ratio']].agg(['mean', 'median', 'std', 'min', 'max']).reset_index()


#%% add experimental design of wave 1
design = pd.read_csv(r"C:\Users\linwang\Box\Work\3_Root_penetration\GWAS_SCREEN_AUG23\Experimental_Design\gwas_waves\GWAS_waves_experimental_design.csv")

merged_data = data_plant_remove_outlier.merge(design, left_on='plant', right_on='QR', how='left')

# wave 1
merged_data_W1 = merged_data[merged_data['wave']=='W1']
statistics = merged_data_W1.groupby('Name')[['root_count_ratio', 'root_area_ratio']].agg(['count','mean', 'median', 'std', 'min', 'max']).reset_index()


plt.figure(figsize=(8, 8))  # Optional: set the figure size
sns.boxplot(x='root_count_ratio', y='Name', data=merged_data_W1, orient="h")
sns.stripplot(x='root_count_ratio', y='Name', data=merged_data_W1, color='black', size=3, jitter=True)


plt.figure(figsize=(8, 8))  # Optional: set the figure size
sns.boxplot(x='root_area_ratio', y='Name', data=merged_data_W1, orient="h")
sns.stripplot(x='root_area_ratio', y='Name', data=merged_data_W1, color='black', size=3, jitter=True)

# wave 2
merged_data_W2 = merged_data[merged_data['wave']=='W2']
statistics = merged_data_W2.groupby('Name')[['root_count_ratio', 'root_area_ratio']].agg(['count','mean', 'median', 'std', 'min', 'max']).reset_index()


plt.figure(figsize=(8, 8))  # Optional: set the figure size
sns.boxplot(x='root_count_ratio', y='Name', data=merged_data_W2, orient="h")
sns.stripplot(x='root_count_ratio', y='Name', data=merged_data_W2, color='black', size=3, jitter=True)


plt.figure(figsize=(8, 8))  # Optional: set the figure size
sns.boxplot(x='root_area_ratio', y='Name', data=merged_data_W2, orient="h")
sns.stripplot(x='root_area_ratio', y='Name', data=merged_data_W2, color='black', size=3, jitter=True)
