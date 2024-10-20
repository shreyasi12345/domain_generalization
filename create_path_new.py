import os
import random
from glob import glob
from pathlib import Path

# Function to create the train, test, val split
def create_splits(root_dir, domains, split_ratio=(0.6, 0.2, 0.2)):
    supported_formats = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']  # Add more extensions if needed
    
    for domain in domains:
        domain_path = os.path.join(root_dir, domain)
        
        if not os.path.isdir(domain_path):
            print(f"Skipping {domain}: Not a valid directory.")
            continue
        
        class_folders = [f for f in os.listdir(domain_path) if os.path.isdir(os.path.join(domain_path, f))]
        
        # Mapping class names to numeric labels starting from 1
        class_to_label = {class_name: idx + 1 for idx, class_name in enumerate(class_folders)}
        
        # Open files with domain name as prefix and save them in the root folder
        train_file = open(os.path.join(root_dir, f'{domain}_train.txt'), 'w')
        val_file = open(os.path.join(root_dir, f'{domain}_val.txt'), 'w')
        test_file = open(os.path.join(root_dir, f'{domain}_test.txt'), 'w')
        
        for class_name, label in class_to_label.items():
            class_folder = os.path.join(domain_path, class_name)
            
            # Gather all images with supported extensions in the class folder
            images = []
            for ext in supported_formats:
                images.extend(glob(os.path.join(class_folder, ext)))
            
            # Shuffle images to randomize the dataset split
            random.shuffle(images)
            
            # Calculate split sizes
            total_images = len(images)
            train_split = int(total_images * split_ratio[0])
            val_split = int(total_images * split_ratio[1])
            test_split = total_images - train_split - val_split
            
            train_images = images[:train_split]
            val_images = images[train_split:train_split + val_split]
            test_images = images[train_split + val_split:]
            
            # Write the image paths and labels to respective files (relative paths)
            for img in train_images:
                relative_path = os.path.relpath(img, root_dir)
                train_file.write(f"{relative_path} {label}\n")
            for img in val_images:
                relative_path = os.path.relpath(img, root_dir)
                val_file.write(f"{relative_path} {label}\n")
            for img in test_images:
                relative_path = os.path.relpath(img, root_dir)
                test_file.write(f"{relative_path} {label}\n")
        
        train_file.close()
        val_file.close()
        test_file.close()
        print(f"Created train, val, and test files for domain: {domain} in {root_dir}")

# Define the root directory containing the domains
root_directory = '/scratch/ghoshs/large_files/FACT_LRDG/Dataset/Data'

# List of domains you want to create the splits for
domains = ['Photos', 'APR', 'Multispectral']  # Example domain names

# Create the splits
create_splits(root_directory, domains)
