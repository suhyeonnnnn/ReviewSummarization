from PIL import Image
import numpy as np
import os

"""
이미지 내에서 entropy 계산

#TODO
- json 형태로 저장할 수 있도록
    

"""

def shannon_entropy(image):
    """Calculate the Shannon entropy of an image."""
    histogram = image.histogram()
    histogram_length = sum(histogram)
    probabilities = [float(h) / histogram_length for h in histogram if h != 0]
    entropy = -sum([p * np.log2(p) for p in probabilities])
    return entropy

def entropy_rate(image, kernel_size=3):
    """Calculate the entropy rate of an image using a sliding window approach."""
    if image.mode != 'L':
        image = image.convert('L')
    data = np.array(image)
    rows, cols = data.shape
    entropy_rate_value = 0
    count = 0

    for i in range(rows - kernel_size + 1):
        for j in range(cols - kernel_size + 1):
            sub_block = data[i:i+kernel_size, j:j+kernel_size].flatten()
            _, counts = np.unique(sub_block, return_counts=True)
            probabilities = counts / counts.sum()
            local_entropy = -np.sum(probabilities * np.log2(probabilities))
            entropy_rate_value += local_entropy
            count += 1

    return entropy_rate_value / count

def excess_entropy(image, max_lag=5):
    """Approximate the excess entropy by considering the mutual information across different lags."""
    if image.mode != 'L':
        image = image.convert('L')
    data = np.array(image)
    total_entropy = 0

    for lag in range(1, max_lag + 1):
        joint_histogram = np.histogram2d(data[:, :-lag].flatten(), data[:, lag:].flatten(), bins=256)[0]
        joint_prob = joint_histogram / joint_histogram.sum()
        marginal_x = joint_prob.sum(axis=1)
        marginal_y = joint_prob.sum(axis=0)
        mutual_info = np.nansum(joint_prob * np.log2(joint_prob / (marginal_x[:, None] * marginal_y)))
        total_entropy += mutual_info

    return total_entropy

def erasure_entropy(image, erase_probability=0.5):
    """Calculate the erasure entropy of an image."""
    entropy = shannon_entropy(image)
    unique_vals = len(set(image.getdata()))
    erasure_entropy_value = (1 - erase_probability) * entropy + erase_probability * np.log2(unique_vals)
    return erasure_entropy_value

# Directory containing the images
image_directory = 'sample/B07BWS4CSM/cgi'

image_files = os.listdir(image_directory)#[f'image{i}.jpg' for i in range(2, 14)]  # Generates list of filenames from image1.jpg to image13.jpg

# Dictionary to store results
results = {name: {} for name in image_files}

# Process each image
for filename in image_files:
    file_path = os.path.join(image_directory, filename)
    if os.path.exists(file_path):
        image = Image.open(file_path)
        results[filename]['Shannon Entropy'] = shannon_entropy(image)
        results[filename]['Entropy Rate'] = entropy_rate(image)
        results[filename]['Excess Entropy'] = excess_entropy(image)
        results[filename]['Erasure Entropy'] = erasure_entropy(image)
    else:
        print(f"File not found: {file_path}")

# Print results for each image
for filename, metrics in results.items():
    print(f"{filename}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    print()  # Newline for better readability

import json

# File path to save the JSON file
#json_file_path = f"sample/{}entropy.json"
#
## Save the results dictionary as a JSON file
#with open(json_file_path, "w") as json_file:
#    json.dump(results, json_file, indent=4)
#
#print(f"Results saved to {json_file_path}")