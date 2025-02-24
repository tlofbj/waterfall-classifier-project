import cv2
import os


def modify_image(image, crop_coords=(67, 11, 686, 1550), target_size=(256, 256)):
    x1, y1, x2, y2 = crop_coords
    modified_image = image[y1:y2, x1:x2]  # Crop
    modified_image = cv2.resize(modified_image, target_size) # Resize
    modified_image = cv2.cvtColor(modified_image, cv2.COLOR_BGR2GRAY) # Gray
    return modified_image


def process_image(image_path, output_path, crop_coords=(67, 11, 686, 1550), target_size=(256, 256)):
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Image {image_path} is None!")
        return
    
    modified_image = modify_image(image, crop_coords, target_size)

    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cv2.imwrite(output_path, modified_image)
    print(f"Processed and saved: {output_path}")
    

def preprocess_dataset(input_dir, output_dir, crop_coords=(67, 11, 686, 1550), target_size=(256, 256)):
    
    for filename in os.listdir(input_dir): # Loop every image in input folder
        if filename.endswith(".png"):
            image_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            process_image(image_path, output_path, crop_coords, target_size)
    

if __name__ == "__main__":
    preprocess_dataset("data/train/raw/signal", "data/train/processed/signal")
    preprocess_dataset("data/train/raw/no_signal", "data/train/processed/no_signal")
    preprocess_dataset("data/test/raw/signal", "data/test/processed/signal")
    preprocess_dataset("data/test/raw/no_signal", "data/test/processed/no_signal")
