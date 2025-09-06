import os
from PIL import Image
from pathlib import Path

def smart_resize_image(image, target_size=(1200, 1200), fill_color=(0, 0, 0)):
    """
    Resize image while preserving aspect ratio using center crop or padding.
    This maintains image quality and focuses on the center portion.
    """
    original_width, original_height = image.size
    target_width, target_height = target_size
    original_ratio = original_width / original_height
    target_ratio = target_width / target_height

    if original_ratio > target_ratio:
        scale_factor = target_height / original_height
        new_width = int(original_width * scale_factor)
        new_height = target_height
        resized = image.resize((new_width, new_height), Image.LANCZOS)
        if new_width > target_width:
            left = (new_width - target_width) // 2
            resized = resized.crop((left, 0, left + target_width, target_height))
        else:
            result = Image.new(image.mode, target_size, fill_color)
            paste_x = (target_width - new_width) // 2
            result.paste(resized, (paste_x, 0))
            resized = result
    else:
        scale_factor = target_width / original_width
        new_width = target_width
        new_height = int(original_height * scale_factor)
        resized = image.resize((new_width, new_height), Image.LANCZOS)
        if new_height > target_height:
            top = (new_height - target_height) // 2
            resized = resized.crop((0, top, target_width, top + target_height))
        else:
            result = Image.new(image.mode, target_size, fill_color)
            paste_y = (target_height - new_height) // 2
            result.paste(resized, (0, paste_y))
            resized = result
    
    return resized

def resize_extending_pictures(input_dir="ExtendingPictures", output_dir="ResizedExtendingPictures", 
                            target_size=(1200, 1200)):
    """
    Process all images in the ExtendingPictures folder structure,
    maintaining the subfolder organization (Apple, Pear, etc.)
    """
    print(f"Starting to resize images from '{input_dir}' to {target_size[0]}x{target_size[1]}...")
    print(f"Output directory: '{output_dir}'")
    print("-" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    total_processed = 0
    total_errors = 0
    
    for root, dirs, files in os.walk(input_dir):
        rel_path = os.path.relpath(root, input_dir)

        if rel_path == '.':
            subfolder_name = ''
        else:
            subfolder_name = rel_path

        image_files = [f for f in files if Path(f).suffix.lower() in image_extensions]
        
        if image_files:
            if subfolder_name:
                output_subfolder = os.path.join(output_dir, subfolder_name)
                os.makedirs(output_subfolder, exist_ok=True)
                print(f"\nProcessing subfolder: {subfolder_name}")
                print(f"Found {len(image_files)} images")
            else:
                output_subfolder = output_dir
                print(f"\nProcessing root directory")
                print(f"Found {len(image_files)} images")

            for img_file in image_files:
                input_path = os.path.join(root, img_file)
                output_path = os.path.join(output_subfolder, img_file)
                
                try:
                    with Image.open(input_path) as img:
                        original_size = img.size
                        if img.mode not in ('RGB', 'L'):
                            if img.mode == 'RGBA':
                                background = Image.new('RGB', img.size, (255, 255, 255))
                                background.paste(img, mask=img.split()[3])
                                img = background
                            else:
                                img = img.convert('RGB')

                        resized_img = smart_resize_image(img, target_size)

                        if Path(img_file).suffix.lower() in ['.jpg', '.jpeg']:
                            resized_img.save(output_path, 'JPEG', quality=95, optimize=True)
                        else:
                            resized_img.save(output_path)
                        
                        print(f"  ✓ {img_file}: {original_size} → {target_size}")
                        total_processed += 1
                        
                except Exception as e:
                    print(f"  ✗ Error processing {img_file}: {str(e)}")
                    total_errors += 1
    
    print("\n" + "=" * 60)
    print("RESIZING COMPLETE!")
    print(f"Total images processed: {total_processed}")
    if total_errors > 0:
        print(f"Total errors: {total_errors}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

def main():
    """
    Main function to run the resizing process
    """
    input_directory = "ExtendingPictures"
    output_directory = "ResizedExtendingPictures"
    target_resolution = (1200, 1200)

    if not os.path.exists(input_directory):
        print(f"Error: Input directory '{input_directory}' not found!")
        print("Please make sure the 'ExtendingPictures' folder exists in the current directory.")
        return

    resize_extending_pictures(
        input_dir=input_directory,
        output_dir=output_directory,
        target_size=target_resolution
    )

if __name__ == "__main__":
    main()