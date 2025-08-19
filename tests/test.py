import argparse
import os
from unittest.mock import patch
from model import process_images
import pytest
def create_dummy_file(filename):
    with open(filename, "w") as f:
        f.write("dummy content")

# Test still have old version of arguments
# Test cases
def test_process_images_invalid_directory():
    """Test that process_images() raises OSError when the directory does not exist."""
    invalid_dir = "non_existent_directory"
    valid_json = "valid.json"

    # Mock the command-line arguments
    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        mock_args.return_value = argparse.Namespace(
            directory=invalid_dir, json_file=valid_json
        )

        # We expect an OSError to be raised with a specific message
        with pytest.raises(OSError, match=f"Directory does not exist: {invalid_dir}"):
            process_images(invalid_dir, valid_json)


def test_process_images_invalid_json_file():
    """Test that process_images() raises OSError when the JSON file does not exist."""
    valid_dir = "valid_directory"
    invalid_json = "non_existent.json"

    # Create a dummy directory for this test case
    os.makedirs(valid_dir, exist_ok=True)

    # Mock the command-line arguments
    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        mock_args.return_value = argparse.Namespace(
            directory=valid_dir, json_file=invalid_json
        )

        # We expect an OSError to be raised with a specific message
        with pytest.raises(OSError, match=f"File does not exist: {invalid_json}"):
            process_images(valid_dir, invalid_json)

    # Clean up the dummy directory after the test
    os.rmdir(valid_dir)
