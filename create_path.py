
import os
import random

# Specify the directory where the class folders are stored
root_dir = '/scratch/ghoshs/large_files/FACT_LRDG/Dataset/Data/Multispectral'
output_files = ['/scratch/ghoshs/large_files/FACT_LRDG/Dataset/Labels/Multispectral_train.txt', '/scratch/ghoshs/large_files/FACT_LRDG/Dataset/Labels/Multispectral_test.txt', '/scratch/ghoshs/large_files/FACT_LRDG/Dataset/Labels/Multispectral_val.txt']

# Lists to hold the image paths and their corresponding class labels
all_images = []

# Collect all image paths and class labels
class_label = 1
for class_name in sorted(os.listdir(root_dir)):
    class_path = os.path.join(root_dir, class_name)
    
    if os.path.isdir(class_path):
        for img_file in os.listdir(class_path):
            img_path = os.path.abspath(os.path.join(class_path, img_file))
            all_images.append((img_path, class_label))
        
        class_label += 1

# Shuffle the list of images to ensure randomness
random.shuffle(all_images)

# Calculate the number of images for each split
total_images = len(all_images)
train_split = int(0.6 * total_images)  # 60% for training
val_split = int(0.2 * total_images)    # 20% for validation (next 20%)
test_split = total_images - train_split - val_split  # Remaining 20% for testing

# Split the images into 60%, 20%, 20% portions
train_images = all_images[:train_split]
val_images = all_images[train_split:train_split + val_split]
test_images = all_images[train_split + val_split:]

# Function to write image paths and labels to a file
def write_to_file(filename, images):
    with open(filename, 'w') as file:
        for img_path, class_label in images:
            file.write(f"{img_path} {class_label}\n")

# Write to the respective files
write_to_file(output_files[0], train_images)  # 60% images in train_60.txt
write_to_file(output_files[1], val_images)    # 20% images in val_20.txt
write_to_file(output_files[2], test_images)   # 20% images in test_20.txt

print(f"Image paths and labels have been saved to {output_files}.")

