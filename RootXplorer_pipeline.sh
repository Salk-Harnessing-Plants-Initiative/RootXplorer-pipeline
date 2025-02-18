#!/bin/bash

# Declare an array of species and their lowercase equivalents
declare -A species_map
species_map=(
  ["Rice"]="Rice"
  ["Arabidopsis"]="Arabidopsis"
  ["Soybean"]="Soybean"
  ["Sorghum"]="Sorghum"
)

# species_map=(
#   ["Rice"]="Rice"
#   ["Arabidopsis"]="Arabidopsis"
# #   ["Soybean"]="Soybean"
# #   ["Sorghum"]="Sorghum"
# )

# Start total timer
TOTAL_START=$SECONDS

# Loop through each species
for species in "${!species_map[@]}"; do
  echo "Processing $species..."
  SPECIES_START=$SECONDS  # Start species timer

  python pipeline_crop_segment_v2.py --experiment "phytagel_concentrations/$species" --species "${species_map[$species]}"
  python pipeline_analysis_v2.py --experiment "phytagel_concentrations/$species"

#   python pipeline_crop_segment_v2.py --experiment "genetic_diversity/$species" --species "${species_map[$species]}"
#   python pipeline_analysis_v2.py --experiment "genetic_diversity/$species"

  SPECIES_TIME=$((SECONDS - SPECIES_START))  # Calculate time for species
  SPECIES_MINUTES=$((SPECIES_TIME / 60))
  echo "Time taken for $species: $SPECIES_MINUTES minutes"
  echo "-----------------------------------------"
done

# Calculate total time
TOTAL_TIME=$(( $(date +%s) - TOTAL_START ))
TOTAL_MINUTES=$((TOTAL_TIME / 60))

echo "Total execution time: ${TOTAL_MINUTES}m"