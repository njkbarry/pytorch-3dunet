#!/bin/bash

# Bash script to automatically run a set of experiments

source venv/bin/activate

# Define the directory containing the config files
config_dir="local_configs/xct_pore_hdf5/experiment-4-5-24"

# Define the Python script to execute
python_script="pytorch3dunet/train.py"

# Define the argument of which to pass the config file
script_arg="--config" 

# Check if the directory exists
if [ ! -d "$config_dir" ]; then
    echo "Directory $config_dir does not exist."
    exit 1
fi

# Check if the Python script exists
if [ ! -f "$python_script" ]; then
    echo "Python script $python_script not found."
    exit 1
fi

# Iterate through each config file
for config_file in "$config_dir"/*.yaml; do
    if [ -f "$config_file" ]; then
        echo "Processing $config_file..."
        # Call the Python script and pass the config file as an argument
        # python3 "$python_script" "$script_arg" "$config_file"
        train3dunet "$script_arg" "$config_file"
    fi
done

echo "All config files processed."
