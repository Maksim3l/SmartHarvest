import json
import numpy as np
from PIL import Image, ImageDraw
import os
from pathlib import Path

def create_masks_from_json(json_file_path, images_dir, output_dir):
    """
    Convert VIA JSON annotations to RGB masks at original image sizes.
    """
    with open(json_file_path, 'r') as f:
        annotations = json.load(f)
    os.makedirs(output_dir, exist_ok=True)

    color_map = {
        'ripe': (255, 0, 0),
        'unripe': (0, 255, 0),
        'spoiled': (0, 0, 255),
        'obscured': (255, 255, 255)
    }

    def find_image_file(filename, base_dir):
        direct_path = os.path.join(base_dir, filename)
        if os.path.exists(direct_path):
            return direct_path
        for root, _, files in os.walk(base_dir):
            if filename in files:
                return os.path.join(root, filename)
        return None

    processed_count = 0
    annotated_count = 0

    for image_data in annotations.values():
        if not isinstance(image_data, dict) or 'filename' not in image_data:
            continue

        filename = image_data['filename']
        plant_type = image_data.get('file_attributes', {}).get('plant', 'unknown')
        regions = image_data.get('regions', [])

        print(f"Processing {filename} ({plant_type}) with {len(regions)} regions")
        image_path = find_image_file(filename, images_dir)
        if not image_path:
            print(f"Warning: Image {filename} not found in {images_dir}")
            continue

        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"Error opening {filename}: {e}")
            continue

        plant_output_dir = os.path.join(output_dir, plant_type)
        os.makedirs(plant_output_dir, exist_ok=True)
        mask = Image.new('RGB', (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(mask)

        regions_drawn = 0
        for region in regions:
            if not region:
                continue
            shape_attrs = region.get('shape_attributes', {})
            region_attrs = region.get('region_attributes', {})
            if shape_attrs.get('name') != 'polygon':
                continue
            x_points = shape_attrs.get('all_points_x', [])
            y_points = shape_attrs.get('all_points_y', [])
            if len(x_points) != len(y_points) or len(x_points) < 3:
                continue
            polygon_coords = list(zip(x_points, y_points))
            ripeness = region_attrs.get('ripeness_factor', 'obscured')
            color = color_map.get(ripeness, (255, 255, 255))
            draw.polygon(polygon_coords, fill=color)
            regions_drawn += 1

        mask_filename = f"{Path(filename).stem}_mask.png"
        mask_path = os.path.join(plant_output_dir, mask_filename)
        mask.save(mask_path)
        processed_count += 1
        if regions_drawn > 0:
            annotated_count += 1
        print(f"Saved mask: {mask_path} ({regions_drawn} regions)")

    print(f"\nSummary:")
    print(f"Processed: {processed_count} images")
    print(f"With annotations: {annotated_count} images")
    return processed_count, annotated_count

def create_class_id_masks(json_file_path, images_dir, output_dir):
    """
    Create masks with class IDs at original image sizes.
    """
    with open(json_file_path, 'r') as f:
        annotations = json.load(f)
    os.makedirs(output_dir, exist_ok=True)

    def find_image_file(filename, base_dir):
        direct_path = os.path.join(base_dir, filename)
        if os.path.exists(direct_path):
            return direct_path
        for root, _, files in os.walk(base_dir):
            if filename in files:
                return os.path.join(root, filename)
        return None

    plant_types = sorted({
        image_data.get('file_attributes', {}).get('plant', '').lower()
        for image_data in annotations.values()
        if isinstance(image_data, dict) and 'file_attributes' in image_data
        and image_data.get('file_attributes', {}).get('plant', '').lower() not in ('', 'unknown')
    })

    def get_class_id(plant_type, ripeness):
        ripeness_map = {'ripe': 0, 'unripe': 1, 'spoiled': 2}
        if plant_type.lower() not in plant_types:
            return 0
        if ripeness not in ripeness_map:
            return len(plant_types) * 3 + 1
        plant_idx = plant_types.index(plant_type.lower())
        return plant_idx * 3 + ripeness_map[ripeness] + 1

    for image_data in annotations.values():
        if not isinstance(image_data, dict) or 'filename' not in image_data:
            continue
        filename = image_data['filename']
        plant_type = image_data.get('file_attributes', {}).get('plant', 'unknown')
        regions = image_data.get('regions', [])
        image_path = find_image_file(filename, images_dir)
        if not image_path:
            continue
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"Error opening {filename}: {e}")
            continue

        plant_output_dir = os.path.join(output_dir, plant_type)
        os.makedirs(plant_output_dir, exist_ok=True)
        mask = np.zeros((height, width), dtype=np.uint8)

        for region in regions:
            if not region:
                continue
            shape_attrs = region.get('shape_attributes', {})
            region_attrs = region.get('region_attributes', {})
            if shape_attrs.get('name') != 'polygon':
                continue
            x_points = shape_attrs.get('all_points_x', [])
            y_points = shape_attrs.get('all_points_y', [])
            if len(x_points) != len(y_points) or len(x_points) < 3:
                continue
            polygon_coords = list(zip(x_points, y_points))
            temp_img = Image.new('L', (width, height), 0)
            ImageDraw.Draw(temp_img).polygon(polygon_coords, fill=255)
            temp_mask = np.array(temp_img) > 0
            ripeness = region_attrs.get('ripeness_factor', 'obscured')
            class_id = get_class_id(plant_type, ripeness)
            mask[temp_mask] = class_id

        mask_filename = f"{Path(filename).stem}_class_mask.png"
        mask_path = os.path.join(plant_output_dir, mask_filename)
        Image.fromarray(mask).save(mask_path)

    print(f"\nFound plant types: {plant_types}")
    print("Class ID Mapping:")
    print("0: Background")
    for i, plant in enumerate(plant_types):
        base_id = i * 3 + 1
        print(f"{base_id}: {plant.title()}-ripe, {base_id+1}: {plant.title()}-unripe, {base_id+2}: {plant.title()}-spoiled")
    if plant_types:
        print(f"{len(plant_types) * 3 + 1}: Obscured")

def smart_resize_image(image, target_size=(1200, 1200), fill_color=(0, 0, 0)):
    """
    Resize image while preserving aspect ratio using center crop or padding.
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

def smart_resize_mask(mask, target_size=(1200, 1200), is_class_mask=False):
    """
    Resize mask while preserving aspect ratio (NEAREST interpolation).
    """
    original_width, original_height = mask.size
    target_width, target_height = target_size
    fill_color = 0 if is_class_mask else (0, 0, 0)
    original_ratio = original_width / original_height
    target_ratio = target_width / target_height

    if original_ratio > target_ratio:
        scale_factor = target_height / original_height
        new_width = int(original_width * scale_factor)
        new_height = target_height
        resized = mask.resize((new_width, new_height), Image.NEAREST)
        if new_width > target_width:
            left = (new_width - target_width) // 2
            resized = resized.crop((left, 0, left + target_width, target_height))
        else:
            result = Image.new(mask.mode, target_size, fill_color)
            paste_x = (target_width - new_width) // 2
            result.paste(resized, (paste_x, 0))
            resized = result
    else:
        scale_factor = target_width / original_width
        new_width = target_width
        new_height = int(original_height * scale_factor)
        resized = mask.resize((new_width, new_height), Image.NEAREST)
        if new_height > target_height:
            top = (new_height - target_height) // 2
            resized = resized.crop((0, top, target_width, top + target_height))
        else:
            result = Image.new(mask.mode, target_size, fill_color)
            paste_y = (target_height - new_height) // 2
            result.paste(resized, (0, paste_y))
            resized = result
    return resized

def smart_upsize_images_and_masks(original_images_dir, masks_dir, class_masks_dir,
                                  resized_images_dir, resized_masks_dir, resized_class_masks_dir,
                                  target_size=(1200, 1200)):
    """
    Upsize images and masks to target_size, preserving aspect ratio.
    """
    print(f"\nSmart upsizing images and masks to {target_size} (preserving aspect ratio)...")
    for output_dir in [resized_images_dir, resized_masks_dir, resized_class_masks_dir]:
        os.makedirs(output_dir, exist_ok=True)

    def find_image_file(filename, base_dir):
        direct_path = os.path.join(base_dir, filename)
        if os.path.exists(direct_path):
            return direct_path
        for root, _, files in os.walk(base_dir):
            if filename in files:
                return os.path.join(root, filename)
        return None

    processed = 0
    for root, _, files in os.walk(masks_dir):
        for filename in files:
            if filename.endswith('_mask.png'):
                orig_filename = filename.replace('_mask.png', '')
                orig_image_path = None
                for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                    potential_path = find_image_file(orig_filename + ext, original_images_dir)
                    if potential_path:
                        orig_image_path = potential_path
                        orig_filename_with_ext = orig_filename + ext
                        break
                if not orig_image_path:
                    print(f"Warning: Original image not found for {filename}")
                    continue
                rel_path = os.path.relpath(root, masks_dir)
                plant_type = rel_path if rel_path != '.' else 'unknown'
                resized_img_plant_dir = os.path.join(resized_images_dir, plant_type)
                resized_mask_plant_dir = os.path.join(resized_masks_dir, plant_type)
                resized_class_mask_plant_dir = os.path.join(resized_class_masks_dir, plant_type)
                for plant_dir in [resized_img_plant_dir, resized_mask_plant_dir, resized_class_mask_plant_dir]:
                    os.makedirs(plant_dir, exist_ok=True)
                try:
                    with Image.open(orig_image_path) as img:
                        print(f"Original size: {img.size}")
                        img_resized = smart_resize_image(img, target_size)
                        img_resized.save(os.path.join(resized_img_plant_dir, orig_filename_with_ext))
                    mask_path = os.path.join(root, filename)
                    with Image.open(mask_path) as mask:
                        mask_resized = smart_resize_mask(mask, target_size, is_class_mask=False)
                        mask_resized.save(os.path.join(resized_mask_plant_dir, filename))
                    class_mask_filename = filename.replace('_mask.png', '_class_mask.png')
                    class_mask_path = os.path.join(class_masks_dir, plant_type, class_mask_filename)
                    if os.path.exists(class_mask_path):
                        with Image.open(class_mask_path) as class_mask:
                            class_mask_resized = smart_resize_mask(class_mask, target_size, is_class_mask=True)
                            class_mask_resized.save(os.path.join(resized_class_mask_plant_dir, class_mask_filename))
                    processed += 1
                    print(f"Smart resized: {orig_filename_with_ext} -> {target_size}")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

    print(f"\nSmart upsizing complete! Processed {processed} image/mask pairs")
    print("All images maintain aspect ratio with center crop/padding as needed")

if __name__ == "__main__":
    json_file = "via_export_json(4).json"
    original_images_dir = "Pictures/"
    print("=== STEP 1: Creating masks at original sizes ===")
    rgb_masks_output = "Masks_Original/"
    class_masks_output = "ClassMasks_Original/"
    processed, annotated = create_masks_from_json(json_file, original_images_dir, rgb_masks_output)
    create_class_id_masks(json_file, original_images_dir, class_masks_output)
    print("\n=== STEP 2: Upsizing images and masks to 1200x1200 ===")
    resized_images_dir = "RePictures/"
    resized_masks_dir = "ReMasks/"
    resized_class_masks_dir = "ReClassMasks/"
    target_resolution = (1200, 1200)
    smart_upsize_images_and_masks(
        original_images_dir, rgb_masks_output, class_masks_output,
        resized_images_dir, resized_masks_dir, resized_class_masks_dir,
        target_resolution
    )
    print("\n=== COMPLETE ===")
    print("You now have:")
    print("- Original size masks in Masks_Original/ and ClassMasks_Original/")
    print("- 1200x1200 images in RePictures/")
    print("- 1200x1200 masks in ReMasks/ and ReClassMasks/")