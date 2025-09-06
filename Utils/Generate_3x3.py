from PIL import Image
import os

def create_image_grid(image_paths, output_path='image_grid.jpg', grid_size=(3, 3), cell_size=(300, 300), padding=10):
    """
    Create a 3x3 grid from 9 images and save it as a single image.
    
    Args:
        image_paths (list): List of paths to 9 images
        output_path (str): Path where the grid image will be saved
        grid_size (tuple): Grid dimensions (rows, cols) - default is (3, 3)
        cell_size (tuple): Size of each image cell (width, height)
        padding (int): Padding between images in pixels
    """
    
    if len(image_paths) != 9:
        raise ValueError("Please provide exactly 9 image paths for a 3x3 grid")
    
    rows, cols = grid_size
    cell_width, cell_height = cell_size
    
    # Calculate total grid dimensions
    total_width = cols * cell_width + (cols - 1) * padding
    total_height = rows * cell_height + (rows - 1) * padding
    
    # Create a new blank image with white background
    grid_image = Image.new('RGB', (total_width, total_height), 'white')
    
    # Process each image
    for i, image_path in enumerate(image_paths):
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
            
        try:
            # Open and resize image
            img = Image.open(image_path)
            img = img.resize(cell_size, Image.Resampling.LANCZOS)
            
            # Calculate position in grid
            row = i // cols
            col = i % cols
            
            # Calculate pixel position
            x = col * (cell_width + padding)
            y = row * (cell_height + padding)
            
            # Paste image into grid
            grid_image.paste(img, (x, y))
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    # Save the grid image
    grid_image.save(output_path, quality=95)
    print(f"Grid image saved as: {output_path}")
    
    return grid_image

# Example usage
if __name__ == "__main__":
    # List your 9 image paths here
    image_list = [
        'testpic1.jpg',
        'testpic2.jpg', 
        'testpic3.jpg',
        'testpic4.jpg',
        'testpic5.jpg',
        'testpic6.jpg',
        'testpic7.jpg',
        'testpic8.jpg',
        'testpic9.jpg'
    ]
    
    # Create the grid
    try:
        grid = create_image_grid(
            image_paths=image_list,
            output_path='my_3x3_grid.jpg',
            cell_size=(400, 400),  # Larger cells for better quality
            padding=15  # More padding between images
        )
        print("Grid created successfully!")
        
    except Exception as e:
        print(f"Error creating grid: {e}")

# Alternative function for creating grid from a folder of images
def create_grid_from_folder(folder_path, output_path='folder_grid.jpg', max_images=9):
    """
    Create a grid from the first 9 images found in a folder.
    """
    import glob
    
    # Supported image extensions
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
    
    if len(image_files) < 9:
        print(f"Warning: Only found {len(image_files)} images, need 9 for 3x3 grid")
        return None
    
    # Take first 9 images
    selected_images = image_files[:max_images]
    
    return create_image_grid(selected_images, output_path)