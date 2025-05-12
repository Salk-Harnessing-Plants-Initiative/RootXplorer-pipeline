#!/bin/bash

# Start total timer
TOTAL_START=$(date +%s)

# Set base directory paths
IMAGE_DIR="images"

# Loop through each experimental design
for experiment_path in "$IMAGE_DIR"/*; do
  if [[ -d "$experiment_path" ]]; then
    experiment=$(basename "$experiment_path")

    # Loop through each species inside the experiment folder
    for species_path in "$experiment_path"/*; do
      if [[ -d "$species_path" ]]; then
        species=$(basename "$species_path")
        echo "Processing $experiment/$species..."
        SPECIES_START=$SECONDS

        python "src/pipeline_crop_segment_v2.py" --experiment "$experiment/$species" --species "$species"
        python "src/pipeline_analysis_v2.py" --experiment "$experiment/$species"

        SPECIES_TIME=$((SECONDS - SPECIES_START))
        SPECIES_MINUTES=$((SPECIES_TIME / 60))
        echo "Time taken for $experiment/$species: ${SPECIES_MINUTES} minutes"
        echo "-----------------------------------------"
      fi
    done
  fi
done

# Calculate total time
TOTAL_TIME=$(( $(date +%s) - TOTAL_START ))
TOTAL_MINUTES=$((TOTAL_TIME / 60))

echo "Total execution time: ${TOTAL_MINUTES}m"
