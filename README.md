# SALK_projects

## Purpose and background of this repo
**Purpose**: extracting traits from cylinder segmentation results for Arabidopsis.

**Background**: we generated the segmentation from original Arabidopsis cylinder images. 
We'd like to extract a few traits that related to root penetration.
![Root traits illustration](https://github.com/linwang9926/SALK_projects/blob/main/cylinder_arab_penetration/Root%20growth%20traits_Elohim.jpg)


## Installation

1. **Clone the repository**:  
   Clone the repository to the local drive.
   ```
   git clone https://github.com/linwang9926/SALK_projects.git
   ```
2. **Navigate to the cloned directory**:  
   
   ```
   cd SALK_projects/cylinder_arab_penetration
   ```
3. **Create a new conda environment**:
   ```
   conda env create -f environment.yml
   ```
4. **Activate conda environment**
    ```
    conda activate cylinder-penetration-arab-analysis
    ```
5. **Install jupyter notebook kernal**:
    ```
    ipython kernel install --user --name=cylinder-penetration-arab-analysis
    ```
