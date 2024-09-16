import argparse
import os
import cv2
import numpy as np
import pandas as pd

def seperate_layers_cylinder_penetrate(image_path,save_path,demo_path):
    # batch crop and save crop  boundary and layer boundary
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
                    image_crop = image[top:top+height, left:left+width,:]
                    cv2.imwrite(os.path.join(save_folder,img),image_crop)
                    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    
                    img_crop = img_gray[top:top+height, left+50:left+width-50] # remove the left and right 50 pixels
                    img_crop_v = np.mean(img_crop,axis=1)
                    
                    # Use np.convolve to calculate the moving average
                    moving_averages = np.convolve(img_crop_v, np.ones(window_size)/window_size, mode='valid')
                    ind = np.argmin(moving_averages[150:-50]) + top + int(window_size/2) + 150 # filter out the first 150 rows and last 50 rows
                    
                    if frame==str(1):
                        # Draw the horizontal line seperating the two gel layers
                        cv2.line(image, (0, ind), (img_gray.shape[1], ind), line_color, line_thickness)
                        
                        # Draw two boundary lines next to the seperating line
                        cv2.line(image, (0, ind - layer_boundary_threshold), (img_gray.shape[1], ind - layer_boundary_threshold), (0,128,128), line_thickness)
                        cv2.line(image, (0, ind + layer_boundary_threshold), (img_gray.shape[1], ind + layer_boundary_threshold), (0,128,128), line_thickness)
                        
                        # draw the bbox of cropping
                        start_point = (left, top)
                        end_point = (left+width, top+height)
                        cv2.rectangle(image, start_point, end_point, (128,0,0), line_thickness)
                        cv2.imwrite(os.path.join(demo_path,wave+'_'+scanner+'_'+plant+'_'+img),image)
                    
                    
                    data = np.reshape(np.array([wave, scanner, plant, frame, ind]),(1,5))
                    df_new = pd.DataFrame(data, columns=title)
                    dataframe = pd.concat([dataframe, df_new], ignore_index=True)
                
    ind_data = dataframe[1:]
    ind_data.to_csv(os.path.join(save_path,'output_layer_index.csv'),index=False)


def main():
    parser = argparse.ArgumentParser(description="Crop image and get boundary")
    parser.add_argument("--image_path", required=True, help="Original image path")
    parser.add_argument("--save_path", required=True, help="Cropped image path")
    parser.add_argument("--demo_path", required=True, help="save demo images with boundary line")
    args = parser.parse_args()
    
    image_path = args.image_path
    save_path = args.save_path
    demo_path = args.demo_path

    seperate_layers_cylinder_penetrate(image_path,save_path,demo_path)


if __name__ == "__main__":
    main()