import os
from PIL import Image

def invert_mask(folder_path):
    # Create output folder if it doesn't exist
    output_folder = os.path.join(folder_path, "inverted")
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            file_path = os.path.join(folder_path, filename)
            
            # Open the image
            with Image.open(file_path) as img:
                # Convert image to grayscale
                grayscale_img = img.convert("L")
                
                # Invert the image
                inverted_img = Image.eval(grayscale_img, lambda x: 255 - x)
                
                # Save the inverted image
                inverted_img.save(os.path.join(output_folder, filename))

    print(f"Inverted mask images have been saved to {output_folder}")

# Specify the folder path here
folder_path = "./dataset/validation/masks"
invert_mask(folder_path)