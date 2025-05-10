#!/bin/bash

# Script to rsync specific folders from a remote server

REMOTE_HOST="dgx07_ext"
REMOTE_BASE_PATH="personal/flipper_training/runs/ppo"
# Using "." as destination means the folders will be copied into the current directory
# where the script is executed.
DESTINATION_BASE_PATH="runs/ppo"

# List of folder names to rsync
# Note: The bullet points (•) have been removed from the folder names.
FOLDERS_TO_SYNC=(
    final_barrier_thesis_666_2025-05-10_11-46-40
    final_barrier_thesis_1_2025-05-10_11-46-28
    final_barrier_thesis_98_2025-05-10_11-46-24
    final_barrier_thesis_26_2025-05-10_11-46-15
    final_barrier_thesis_13_2025-05-10_10-52-39
    final_barrier_thesis_420_2025-05-10_10-52-28
    final_barrier_thesis_64_2025-05-10_10-51-44
    final_barrier_thesis_32_2025-05-10_10-50-02
    final_barrier_thesis_16_2025-05-10_10-48-24
    final_barrier_thesis_42_2025-05-10_10-46-54
)

# Rsync options
# -a: archive mode (preserves permissions, timestamps, etc., and is recursive)
# -v: verbose (shows what's being transferred)
# -z: compress data during transfer
# -r: recursive (though -a implies -r, it's often included for clarity)
# --progress: show progress during transfer (optional, but nice to have for large files)
RSYNC_OPTIONS="-avzr --progress"

# Check if destination base path exists, if not, create it
# For DESTINATION_BASE_PATH=".", this check is not strictly necessary but good practice
# if you were to specify a subdirectory.
if [ "$DESTINATION_BASE_PATH" != "." ] && [ ! -d "$DESTINATION_BASE_PATH" ]; then
    echo "Destination base path '$DESTINATION_BASE_PATH' does not exist. Creating it..."
    mkdir -p "$DESTINATION_BASE_PATH"
    if [ $? -ne 0 ]; then
        echo "Failed to create destination directory '$DESTINATION_BASE_PATH'. Exiting."
        exit 1
    fi
fi

echo "Starting rsync process..."
echo "Remote host: $REMOTE_HOST"
echo "Remote base path: $REMOTE_BASE_PATH"
echo "Local destination base path: $(realpath "$DESTINATION_BASE_PATH")"
echo ""

# Loop through the folders and rsync each one
for folder_name in "${FOLDERS_TO_SYNC[@]}"; do
    REMOTE_SOURCE_PATH="${REMOTE_HOST}:${REMOTE_BASE_PATH}/${folder_name}"
    # The source path does NOT end with a '/' to ensure the folder itself is copied,
    # not just its contents.
    # The destination is DESTINATION_BASE_PATH, so rsync will create folder_name inside it.

    echo "----------------------------------------------------------------------"
    echo "Syncing folder: $folder_name"
    echo "From: $REMOTE_SOURCE_PATH"
    echo "To:   $(realpath "$DESTINATION_BASE_PATH")/$folder_name"
    echo "----------------------------------------------------------------------"

    rsync $RSYNC_OPTIONS "$REMOTE_SOURCE_PATH" "$DESTINATION_BASE_PATH/"

    if [ $? -eq 0 ]; then
        echo "Successfully synced $folder_name."
    else
        echo "ERROR: Failed to sync $folder_name. Rsync exit code: $?"
        # Decide if you want to continue with other folders or exit on error
        # exit 1 # Uncomment to exit on first error
    fi
    echo ""
done

echo "----------------------------------------------------------------------"
echo "All specified rsync operations complete."
echo "----------------------------------------------------------------------"
