#!/bin/bash

# Define common directories
DATA_DIR="path/to/data" 
TARGET_DIR="path/to/data_replay"
REFERENCE_DIR="/path/to/reference_folder"

for i in {0..99}
do
    # Start timing for this episode
    episode_start_time=$(date +%s)
    
    echo "Processing episode $i"
    echo "Started at: $(date)"
    
    # Run interpolation with data and target directories
    echo "Running: python 0_interpolation.py -e $i --data-dir $DATA_DIR --target-dir $TARGET_DIR --enable-fsr"
    python 0_interpolation.py -e "$i" --data-dir "$DATA_DIR" --target-dir "$TARGET_DIR" --downsample-rate 3  --enable-fsr
    
    # Run replay hand with data and save directories
    # Add reference_dir only when i=0
    if [ $i -eq 0 ]; then
        echo "Running: python 1_replay_hand.py -e $i --data-dir $DATA_DIR --save-dir $TARGET_DIR --reference-dir $REFERENCE_DIR --fps 10"
        python 1_replay_hand.py -e "$i" --data-dir "$DATA_DIR" --save-dir "$TARGET_DIR" --reference-dir "$REFERENCE_DIR" --fps 10 --hand-type xhand 
    else
        echo "Running: python 1_replay_hand.py -e $i --data-dir $DATA_DIR --save-dir $TARGET_DIR --fps 10"
        python 1_replay_hand.py -e "$i" --data-dir "$DATA_DIR" --save-dir "$TARGET_DIR" --fps 10 --hand-type xhand --verbose
    fi
    
    # Calculate and display time taken for this episode
    episode_end_time=$(date +%s)
    episode_duration=$((episode_end_time - episode_start_time))
    
    # Convert seconds to hours, minutes, and seconds
    hours=$((episode_duration / 3600))
    minutes=$(( (episode_duration % 3600) / 60 ))
    seconds=$((episode_duration % 60))
    
    echo "----------------------------------------"
    echo "Episode $i completed"
    echo "Time taken: ${hours}h ${minutes}m ${seconds}s"
    echo "Finished at: $(date)"
    echo "----------------------------------------"
done

echo "All iterations completed"

