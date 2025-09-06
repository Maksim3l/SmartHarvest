import os
from PIL import Image
from pathlib import Path

def rotate_images_in_folder(input_dir="ResizedExtendingPictures", rotation_angle=-90):
    """
    Rotate all images in the folder structure by the specified angle.
    -90 degrees = 90 degrees clockwise (to the right)
    90 degrees = 90 degrees counter-clockwise (to the left)
    """
    print(f"Starting to rotate images in '{input_dir}' by {abs(rotation_angle)} degrees {'clockwise' if rotation_angle < 0 else 'counter-clockwise'}...")
    print("-" * 60)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    total_processed = 0
    total_errors = 0
    
    for root, dirs, files in os.walk(input_dir):
        rel_path = os.path.relpath(root, input_dir)
        
        image_files = [f for f in files if Path(f).suffix.lower() in image_extensions]
        
        if image_files:
            if rel_path != '.':
                print(f"\nProcessing subfolder: {rel_path}")
            else:
                print(f"\nProcessing root directory")
            print(f"Found {len(image_files)} images to rotate")
            
            for img_file in image_files:
                file_path = os.path.join(root, img_file)
                
                try:
                    with Image.open(file_path) as img:
                        rotated_img = img.rotate(rotation_angle, expand=True)
                        
                        if Path(img_file).suffix.lower() in ['.jpg', '.jpeg']:
                            rotated_img.save(file_path, 'JPEG', quality=95, optimize=True)
                        else:
                            rotated_img.save(file_path)
                        
                        print(f"  ✓ Rotated: {img_file}")
                        total_processed += 1
                        
                except Exception as e:
                    print(f"  ✗ Error rotating {img_file}: {str(e)}")
                    total_errors += 1

    print("\n" + "=" * 60)
    print("ROTATION COMPLETE!")
    print(f"Total images rotated: {total_processed}")
    if total_errors > 0:
        print(f"Total errors: {total_errors}")
    print("=" * 60)

def main():
    """
    Main function to run the rotation process
    """
    target_directory = "ResizedExtendingPictures"

    if not os.path.exists(target_directory):
        print(f"Error: Directory '{target_directory}' not found!")
        print("Please make sure the 'ResizedExtendingPictures' folder exists.")
        return

    print("This script will rotate all images in 'ResizedExtendingPictures' 90 degrees clockwise.")
    print("The original files will be overwritten with the rotated versions.")
    response = input("\nDo you want to continue? (yes/no): ").lower().strip()
    
    if response in ['yes', 'y']:
        rotate_images_in_folder(
            input_dir=target_directory,
            rotation_angle=-90
        )
    else:
        print("Operation cancelled.")

if __name__ == "__main__":
    main()