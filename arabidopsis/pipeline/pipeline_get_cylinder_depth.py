import argparse
import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_top_bottom_index(img_center,index):
    """Get the top and bottom index of a left or right slice."""
    # get image slice
    img_ind = img_center[:,index-50:index+50]
    # apply filter to do edge detection
    gradient_y = cv2.Sobel(img_ind, cv2.CV_64F, 0, 1, ksize=5)
    gradient_avgy = np.mean(gradient_y, axis=1)
    top = gradient_avgy[:int(len(gradient_avgy)/3)]
    bottom = gradient_avgy[int(len(gradient_avgy)/3*2):]
    #print(f"max value: {np.max(top)} at {np.argmax(top)}")
    #print(f"min value: {np.min(bottom)} at {np.argmin(bottom)+int(len(gradient_avgy)/3*2)}")
    top_ind = np.argmax(top)
    bottom_ind = np.argmin(bottom)+int(len(gradient_avgy)/3*2)
    return top_ind, bottom_ind

def get_avg_index(img_center):
    # get number of columns
    columns = img_center.shape[1]
    # get the 1st and 3rd quarters align the x-axis
    left_ind, right_ind = int(columns/4), int(columns/4*3)
    # get the top and bottom index of left and right slice
    top_ind_left, bottom_ind_left = get_top_bottom_index(img_center,left_ind)
    top_ind_right, bottom_ind_right = get_top_bottom_index(img_center,right_ind)
    top_ind = int(np.mean([top_ind_left, top_ind_right]))
    bottom_ind = int(np.mean([bottom_ind_left, bottom_ind_right]))
    return top_ind, bottom_ind
    
def get_index_batch(folder):
    data = pd.DataFrame()
    waves = [file for file in os.listdir(folder)]
    waves = sorted(waves, key=lambda x: int(''.join(filter(str.isdigit, x))))
    for wave in waves:
        scanners = [file for file in os.listdir(os.path.join(folder,wave))]
        for scanner in scanners:
            plants = [file for file in os.listdir(os.path.join(folder,wave,scanner))]
            for plant in plants:
                print(f"wave: {wave}, scanner: {scanner}, plant: {plant}")
                plant_path = os.path.join(folder, wave, scanner, plant)
                imgs = [file for file in os.listdir(plant_path)]
                imgs = sorted(imgs, key=lambda x: int(''.join(filter(str.isdigit, x))))
                for image in imgs:
                    frame = os.path.splitext(image)[0]
                    left = 460 if scanner.endswith('SlowScanner') else 300 # based on the 1030 Y for slowscanner, 1069Y for FastScannner
                    img_path = os.path.join(plant_path, image)
                    img = cv2.imread(img_path)
                    img_center = img[:,left:left+1150,0]
                    top_ind, bottom_ind = get_avg_index(img_center)
                    data_new = pd.DataFrame([{'wave':wave,'scanner':scanner,'plant':plant,'frame':frame,'top_ind':top_ind,'bottom_ind':bottom_ind}])
                    data = pd.concat([data,data_new],ignore_index=True)
    return data

folder = r"Y:\Lin_Wang\Ara_cylinder\root_penetrate\col-0\Images"
data = get_index_batch(folder)
data.to_csv(r"Y:\Lin_Wang\Ara_cylinder\root_penetrate\col-0\top_bottom_ind.csv", index=False)

