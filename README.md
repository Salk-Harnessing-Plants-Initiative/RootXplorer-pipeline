# RootXplorer

## Purpose and background
**Purpose**: Segment cylinder images and extract traits from segmentation masks for Arabidopsis, rice, soybean, and sorghum.

**Background**: We developed RootXplorer, a computer vision-based 3D phenotyping platform for high-throughput quantification of root penetration-related phenotypes across dicot and monocot species non-invasively. RootXplorer integrates a novel phytagel-based cylinder system, a 3D imaging unit, and an automated Python pipeline to extract root penetration-related traits with high precision. 

## Installation

1. **Clone the repository**:  
   Clone the repository to the local drive.
   ```
   git clone https://github.com/Salk-Harnessing-Plants-Initiative/RootXplorer.git
   ```

2. **Navigate to the cloned directory**:  
   
   ```
   cd RootXplorer
   ```

## Organize the pipeline and your images
Models can be downloaded from [Box](https://salkinstitute.box.com/s/cqgv1dwm1hkf84eid72hdjqg47nwbpo5).

Please make sure to organize the downloaded pipeline, model, and your own images in the following architecture:

```
RootXplorer/
├── images/
│   ├── experimental design (e.g., genetic_diversity)/
│   │   ├── species (e.g., Arabidopsis)/
│   │   │   ├── plant name (e.g., ZHOKUWVOIZ)/
│   │   │   │   ├── frame image (e.g., 1.png)
│   │   │   ├── plant and experiment mapping (e.g., acc_barcodes_cylinders.csv)
├── src/
│   ├── pipeline_analysis_v2.py
│   ├── pipeline_crop_segment_v2.py
│   ├── RootXplorer_pipeline.sh
├── model/
│   ├── arabidopsis_model.pth
│   ├── rice_seminal_model.pth
│   ├── soybean_sorghum_model.pth
│   ├── label_class_dict_lr.csv
├── Dockerfile
├── environment.yml
├── LICENSE
├── README.md
├── requirements.txt
```

## Running the pipeline with a shell file (RootXplorer_pipeline.sh)
1. **create the environment**:
   In terminal, navigate to your root folder and type:
   ```
   conda env create -f environment.yml
   ```
   or
   ```
   mamba env create -f environment.yml
   ```

2. **activate the environment**:
   ```
   conda activate segmentation-analysis
   ```

3. **run the shell file**:
   ```
   sed -i 's/\r$//' src/RootXplorer_pipeline.sh
   bash src/RootXplorer_pipeline.sh
   ```

## Running the pipeline with docker
Make sure you have `images`, `model`, and `src` subfolders in your root folder.

1. **build the docker**:
   ```
   docker build -t rootxplorer .
   ```

2. **run the docker**:
   ```
   docker run --gpus all rootxplorer
   ```
