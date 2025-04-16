#!/bin/bash

# Function to display help message
function display_help() {
    echo "Usage: $0 <directory_path>"
    echo
    echo "This script searches for all .ipynb files recursively in the specified"
    echo "directory and its subdirectories, and converts them to .html files using"
    echo "Jupyter nbconvert. The output .html files will retain the same name and"
    echo "location as the input .ipynb files."
    echo
    echo "Options:"
    echo "  -h, --help    Display this help message and exit."
    exit 0
}

# Check for help option
if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    display_help
fi

# Check if the directory path is provided
if [ -z "$1" ]; then
    echo "Error: Directory path is required."
    display_help
fi

# The folder to search in, provided as a command-line argument.
SEARCH_DIR="$1"

# Check if the provided path is a valid directory
if [ ! -d "$SEARCH_DIR" ]; then
    echo "Error: Directory '$SEARCH_DIR' does not exist."
    exit 1
fi

# Find all .ipynb files recursively in the specified directory.
find "$SEARCH_DIR" -type f -name "*.ipynb" | while read notebook; do
    # Convert the notebook to an HTML file in the same directory with the same name.
    jupyter nbconvert --to html "$notebook"
done

