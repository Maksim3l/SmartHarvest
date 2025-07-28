import json
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import os
from pathlib import Path
import random
import cv2
from scipy import ndimage

def create_instance_masks_from_json(json_file_path, images_dir, output_dir, skip_classes=None):
    """
    Convert VIA JSON annotations to instance masks preserving individual fruit instances.
    Each instance gets a unique ID while maintaining class information.
    """
    if skip_classes is None:
        skip_classes = []
    
    skip_classes = [cls.lower() for cls in skip_classes]
    
    with open(json_file_path, 'r') as f:
        annotations = json.load(f)
    
    os.makedirs(output_dir, exist_ok=True)

    semantic_dir = os.path.join(output_dir, 'semantic_masks')
    instance_dir = os.path.join(output_dir, 'instance_masks')
    panoptic_dir = os.path.join(output_dir, 'panoptic_masks')
    os.makedirs(semantic_dir, exist_ok=True)
    os.makedirs(instance_dir, exist_ok=True)
    os.makedirs(panoptic_dir, exist_ok=True)

    class_map = {
        'apple': {'ripe': 1, 'unripe': 2, 'spoiled': 3},
        'cherry': {'ripe': 4, 'unripe': 5, 'spoiled': 6},
        'cucumber': {'ripe': 7, 'unripe': 8, 'spoiled': 9},
        'lettuce': {'ripe': 10, 'unripe': 11, 'spoiled': 12},
        'pear': {'ripe': 13, 'unripe': 14, 'spoiled': 15},
        'pepper': {'ripe': 16, 'unripe': 17, 'spoiled': 18},
        'raspberry': {'ripe': 19, 'unripe': 20, 'spoiled': 21},
        'strawberry': {'ripe': 22, 'unripe': 23, 'spoiled': 24},
        'tomato': {'ripe': 25, 'unripe': 26, 'spoiled': 27},
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
    skipped_count = 0
    total_instances = 0
    
    global_instance_id = 1

    for image_data in annotations.values():
        if not isinstance(image_data, dict) or 'filename' not in image_data:
            continue

        filename = image_data['filename']
        plant_type = image_data.get('file_attributes', {}).get('plant', 'unknown').lower()
        regions = image_data.get('regions', [])

        if plant_type in skip_classes:
            print(f"SKIPPING {filename} ({plant_type}) - class in skip list")
            skipped_count += 1
            continue

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

        plant_semantic_dir = os.path.join(semantic_dir, plant_type)
        plant_instance_dir = os.path.join(instance_dir, plant_type)
        plant_panoptic_dir = os.path.join(panoptic_dir, plant_type)
        
        for dir_path in [plant_semantic_dir, plant_instance_dir, plant_panoptic_dir]:
            os.makedirs(dir_path, exist_ok=True)

        semantic_mask = np.zeros((height, width), dtype=np.uint8)
        instance_mask = np.zeros((height, width), dtype=np.uint16)
        panoptic_mask = np.zeros((height, width), dtype=np.uint32)

        instances_info = []
        image_instance_count = 0

        for region_idx, region in enumerate(regions):
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
            ripeness = region_attrs.get('ripeness_factor', 'ripe')

            if plant_type in class_map and ripeness in class_map[plant_type]:
                semantic_class_id = class_map[plant_type][ripeness]
            else:
                semantic_class_id = 28

            temp_img = Image.new('L', (width, height), 0)
            ImageDraw.Draw(temp_img).polygon(polygon_coords, fill=255)
            instance_pixels = np.array(temp_img) > 0
            
            if np.sum(instance_pixels) < 10:
                continue

            semantic_mask[instance_pixels] = semantic_class_id
            instance_mask[instance_pixels] = global_instance_id
            
            panoptic_id = semantic_class_id * 1000 + global_instance_id
            panoptic_mask[instance_pixels] = panoptic_id

            bbox = get_bbox_from_mask(instance_pixels)
            
            instances_info.append({
                'instance_id': int(global_instance_id),
                'semantic_class_id': int(semantic_class_id),
                'plant_type': plant_type,
                'ripeness': ripeness,
                'area': int(np.sum(instance_pixels)),  
                'bbox': [int(x) for x in bbox]
            })
            
            global_instance_id += 1
            image_instance_count += 1
            total_instances += 1

        base_filename = Path(filename).stem
        semantic_path = os.path.join(plant_semantic_dir, f"{base_filename}_semantic.png")
        Image.fromarray(semantic_mask).save(semantic_path)
        instance_path = os.path.join(plant_instance_dir, f"{base_filename}_instance.png")
        Image.fromarray(instance_mask).save(instance_path)
        panoptic_path = os.path.join(plant_panoptic_dir, f"{base_filename}_panoptic.png")
        Image.fromarray(panoptic_mask).save(panoptic_path)

        info_path = os.path.join(plant_instance_dir, f"{base_filename}_instances.json")
        with open(info_path, 'w') as f:
            json.dump({
                'filename': filename,
                'plant_type': plant_type,
                'image_size': [int(width), int(height)], 
                'num_instances': int(image_instance_count),  
                'instances': instances_info
            }, f, indent=2)

        processed_count += 1
        print(f"Saved masks for {filename}: {image_instance_count} instances")

    print(f"\nInstance Mask Generation Summary:")
    print(f"Processed: {processed_count} images")
    print(f"Skipped: {skipped_count} images ({skip_classes})")
    print(f"Total instances: {total_instances} individual fruits/vegetables")
    if processed_count > 0:
        print(f"Average instances per image: {total_instances/processed_count:.1f}")
    
    return processed_count, total_instances


def get_bbox_from_mask(mask):
    """Get bounding box coordinates from binary mask"""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return [0, 0, 0, 0]
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    return [int(cmin), int(rmin), int(cmax), int(rmax)]  # [x1, y1, x2, y2]

def smart_resize_image(image, target_size=(1200, 1200), fill_color=(0, 0, 0)):
    """
    Resize image while preserving aspect ratio using center crop or padding.
    """
    original_width, original_height = image.size
    target_width, target_height = target_size
    original_ratio = original_width / original_height
    target_ratio = target_width / target_height

    if original_ratio > target_ratio:
        # Image is wider - scale by height and crop width
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
        # Image is taller - scale by width and crop height
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

def smart_resize_mask(mask, target_size=(1200, 1200), is_instance_mask=False):
    """
    Resize mask while preserving aspect ratio (NEAREST interpolation).
    """
    original_width, original_height = mask.size
    target_width, target_height = target_size
    
    # Set appropriate fill color based on mask type
    if is_instance_mask:
        fill_color = 0  # Background for instance masks
    elif mask.mode == 'L':
        fill_color = 0  # Grayscale mask
    else:
        fill_color = (0, 0, 0)  # RGB mask
    
    original_ratio = original_width / original_height
    target_ratio = target_width / target_height

    if original_ratio > target_ratio:
        # Image is wider - scale by height and crop width
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
        # Image is taller - scale by width and crop height
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

def smart_resize_instance_masks(original_images_dir, instance_masks_dir, 
                               resized_images_dir, resized_masks_dir, 
                               target_size=(1200, 1200)):
    """
    Smart resize images and all instance mask types to target_size, preserving aspect ratio.
    """
    print(f"\nSmart resizing images and instance masks to {target_size} (preserving aspect ratio)...")
    
    # Create output directories
    resized_semantic_dir = os.path.join(resized_masks_dir, 'semantic_masks')
    resized_instance_dir = os.path.join(resized_masks_dir, 'instance_masks')
    resized_panoptic_dir = os.path.join(resized_masks_dir, 'panoptic_masks')
    
    for output_dir in [resized_images_dir, resized_semantic_dir, resized_instance_dir, resized_panoptic_dir]:
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
    
    # Process semantic masks as the reference
    semantic_masks_original = os.path.join(instance_masks_dir, 'semantic_masks')
    
    for root, _, files in os.walk(semantic_masks_original):
        for filename in files:
            if filename.endswith('_semantic.png'):
                base_name = filename.replace('_semantic.png', '')
                
                # Get plant type from directory structure
                rel_path = os.path.relpath(root, semantic_masks_original)
                plant_type = rel_path if rel_path != '.' else 'unknown'
                
                # Find original image
                orig_image_path = None
                for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                    potential_path = find_image_file(base_name + ext, original_images_dir)
                    if potential_path:
                        orig_image_path = potential_path
                        orig_filename_with_ext = base_name + ext
                        break
                
                if not orig_image_path:
                    print(f"Warning: Original image not found for {filename}")
                    continue
                
                # Create plant-specific output directories
                resized_img_plant_dir = os.path.join(resized_images_dir, plant_type)
                resized_semantic_plant_dir = os.path.join(resized_semantic_dir, plant_type)
                resized_instance_plant_dir = os.path.join(resized_instance_dir, plant_type)
                resized_panoptic_plant_dir = os.path.join(resized_panoptic_dir, plant_type)
                
                for plant_dir in [resized_img_plant_dir, resized_semantic_plant_dir, 
                                resized_instance_plant_dir, resized_panoptic_plant_dir]:
                    os.makedirs(plant_dir, exist_ok=True)
                
                try:
                    # Resize original image
                    with Image.open(orig_image_path) as img:
                        print(f"Original size: {img.size} -> {target_size}")
                        img_resized = smart_resize_image(img, target_size)
                        img_resized.save(os.path.join(resized_img_plant_dir, orig_filename_with_ext))
                    
                    # Resize semantic mask
                    semantic_path = os.path.join(root, filename)
                    with Image.open(semantic_path) as semantic_mask:
                        semantic_resized = smart_resize_mask(semantic_mask, target_size, is_instance_mask=False)
                        semantic_resized.save(os.path.join(resized_semantic_plant_dir, filename))
                    
                    # Resize instance mask
                    instance_filename = filename.replace('_semantic.png', '_instance.png')
                    instance_path = os.path.join(instance_masks_dir, 'instance_masks', plant_type, instance_filename)
                    if os.path.exists(instance_path):
                        with Image.open(instance_path) as instance_mask:
                            instance_resized = smart_resize_mask(instance_mask, target_size, is_instance_mask=True)
                            instance_resized.save(os.path.join(resized_instance_plant_dir, instance_filename))
                    
                    # Resize panoptic mask
                    panoptic_filename = filename.replace('_semantic.png', '_panoptic.png')
                    panoptic_path = os.path.join(instance_masks_dir, 'panoptic_masks', plant_type, panoptic_filename)
                    if os.path.exists(panoptic_path):
                        with Image.open(panoptic_path) as panoptic_mask:
                            panoptic_resized = smart_resize_mask(panoptic_mask, target_size, is_instance_mask=True)
                            panoptic_resized.save(os.path.join(resized_panoptic_plant_dir, panoptic_filename))
                    
                    # Copy instance info JSON (no resizing needed)
                    info_filename = filename.replace('_semantic.png', '_instances.json')
                    info_path = os.path.join(instance_masks_dir, 'instance_masks', plant_type, info_filename)
                    if os.path.exists(info_path):
                        import shutil
                        shutil.copy2(info_path, os.path.join(resized_instance_plant_dir, info_filename))
                    
                    processed += 1
                    print(f"Smart resized: {orig_filename_with_ext} and all masks -> {target_size}")
                    
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

    print(f"\nSmart resizing complete! Processed {processed} image/mask sets")
    print("All images and masks maintain aspect ratio with center crop/padding as needed")

def augment_image_and_masks(image, semantic_mask, instance_mask, augmentation_params):
    """
    Apply augmentations to image and corresponding masks.
    Returns augmented image, semantic mask, and instance mask.
    """
    # Convert PIL images to numpy arrays for processing
    img_array = np.array(image)
    semantic_array = np.array(semantic_mask)
    instance_array = np.array(instance_mask)
    
    # Store original dtypes
    semantic_dtype = semantic_array.dtype
    instance_dtype = instance_array.dtype
    
    # Random selection of augmentations
    augmentations_applied = []
    
    # 1. Rotation (always apply with some probability)
    if random.random() < augmentation_params.get('rotation_prob', 0.5):
        angle = random.uniform(
            augmentation_params.get('rotation_min', -15),
            augmentation_params.get('rotation_max', 15)
        )
        img_array = ndimage.rotate(img_array, angle, reshape=False, order=1)
        semantic_array = ndimage.rotate(semantic_array, angle, reshape=False, order=0)
        instance_array = ndimage.rotate(instance_array, angle, reshape=False, order=0)
        augmentations_applied.append(f"rotation_{angle:.1f}")
    
    # 2. Horizontal Flip
    if random.random() < augmentation_params.get('flip_prob', 0.5):
        img_array = np.fliplr(img_array)
        semantic_array = np.fliplr(semantic_array)
        instance_array = np.fliplr(instance_array)
        augmentations_applied.append("h_flip")
    
    # 3. Scaling/Zoom
    if random.random() < augmentation_params.get('scale_prob', 0.5):
        scale = random.uniform(
            augmentation_params.get('scale_min', 0.8),
            augmentation_params.get('scale_max', 1.2)
        )
        h, w = img_array.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize
        img_array = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        semantic_array = cv2.resize(semantic_array, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        instance_array = cv2.resize(instance_array, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # Crop or pad to original size
        if scale > 1:
            # Crop center
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            img_array = img_array[start_h:start_h+h, start_w:start_w+w]
            semantic_array = semantic_array[start_h:start_h+h, start_w:start_w+w]
            instance_array = instance_array[start_h:start_h+h, start_w:start_w+w]
        else:
            # Pad
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            img_array = np.pad(img_array, ((pad_h, h-new_h-pad_h), (pad_w, w-new_w-pad_w), (0, 0)), mode='constant')
            semantic_array = np.pad(semantic_array, ((pad_h, h-new_h-pad_h), (pad_w, w-new_w-pad_w)), mode='constant')
            instance_array = np.pad(instance_array, ((pad_h, h-new_h-pad_h), (pad_w, w-new_w-pad_w)), mode='constant')
        
        augmentations_applied.append(f"scale_{scale:.2f}")
    
    # 4. Translation
    if random.random() < augmentation_params.get('translate_prob', 0.3):
        h, w = img_array.shape[:2]
        max_trans = augmentation_params.get('translate_max', 0.1)
        tx = random.uniform(-max_trans * w, max_trans * w)
        ty = random.uniform(-max_trans * h, max_trans * h)
        
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        img_array = cv2.warpAffine(img_array, M, (w, h))
        semantic_array = cv2.warpAffine(semantic_array, M, (w, h), flags=cv2.INTER_NEAREST)
        instance_array = cv2.warpAffine(instance_array, M, (w, h), flags=cv2.INTER_NEAREST)
        augmentations_applied.append(f"translate_{tx:.0f}_{ty:.0f}")
    
    # Convert back to PIL for photometric augmentations
    image = Image.fromarray(img_array)
    semantic_mask = Image.fromarray(semantic_array.astype(semantic_dtype))
    instance_mask = Image.fromarray(instance_array.astype(instance_dtype))
    
    # 5. Brightness
    if random.random() < augmentation_params.get('brightness_prob', 0.5):
        factor = random.uniform(
            augmentation_params.get('brightness_min', 0.7),
            augmentation_params.get('brightness_max', 1.3)
        )
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(factor)
        augmentations_applied.append(f"brightness_{factor:.2f}")
    
    # 6. Contrast
    if random.random() < augmentation_params.get('contrast_prob', 0.5):
        factor = random.uniform(
            augmentation_params.get('contrast_min', 0.7),
            augmentation_params.get('contrast_max', 1.3)
        )
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(factor)
        augmentations_applied.append(f"contrast_{factor:.2f}")
    
    # 7. Saturation
    if random.random() < augmentation_params.get('saturation_prob', 0.5):
        factor = random.uniform(
            augmentation_params.get('saturation_min', 0.5),
            augmentation_params.get('saturation_max', 1.5)
        )
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(factor)
        augmentations_applied.append(f"saturation_{factor:.2f}")
    
    # 8. Hue Shift
    if random.random() < augmentation_params.get('hue_prob', 0.3):
        img_array = np.array(image)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV).astype(np.float32)
        hue_shift = random.uniform(
            augmentation_params.get('hue_min', -15),
            augmentation_params.get('hue_max', 15)
        )
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
        img_array = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        image = Image.fromarray(img_array)
        augmentations_applied.append(f"hue_{hue_shift:.1f}")
    
    # 9. Gaussian Blur
    if random.random() < augmentation_params.get('blur_prob', 0.2):
        radius = random.uniform(
            augmentation_params.get('blur_min', 0.5),
            augmentation_params.get('blur_max', 2.0)
        )
        image = image.filter(ImageFilter.GaussianBlur(radius=radius))
        augmentations_applied.append(f"blur_{radius:.1f}")
    
    # 10. Noise
    if random.random() < augmentation_params.get('noise_prob', 0.2):
        img_array = np.array(image)
        noise_strength = augmentation_params.get('noise_strength', 0.02)
        noise = np.random.normal(0, noise_strength * 255, img_array.shape)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        image = Image.fromarray(img_array)
        augmentations_applied.append(f"noise_{noise_strength:.3f}")
    
    return image, semantic_mask, instance_mask, augmentations_applied


def create_augmented_dataset(original_images_dir, masks_dir, augmented_output_dir, 
                           augmentation_params=None, augmentations_per_image=8):
    """
    Create augmented dataset from original images and masks.
    Fixed version that properly matches images with their corresponding masks.
    """
    if augmentation_params is None:
        augmentation_params = {
            # Geometric augmentations
            'rotation_prob': 0.7,
            'rotation_min': -15,
            'rotation_max': 15,
            'flip_prob': 0.5,
            'scale_prob': 0.5,
            'scale_min': 0.8,
            'scale_max': 1.2,
            'translate_prob': 0.3,
            'translate_max': 0.1,
            
            # Photometric augmentations
            'brightness_prob': 0.6,
            'brightness_min': 0.7,
            'brightness_max': 1.3,
            'contrast_prob': 0.5,
            'contrast_min': 0.7,
            'contrast_max': 1.3,
            'saturation_prob': 0.5,
            'saturation_min': 0.5,
            'saturation_max': 1.5,
            'hue_prob': 0.3,
            'hue_min': -15,
            'hue_max': 15,
            
            # Other augmentations
            'blur_prob': 0.2,
            'blur_min': 0.5,
            'blur_max': 2.0,
            'noise_prob': 0.2,
            'noise_strength': 0.02
        }
    
    # Create main output directory
    os.makedirs(augmented_output_dir, exist_ok=True)
    
    # Create output directories
    aug_images_dir = os.path.join(augmented_output_dir, 'images')
    aug_semantic_dir = os.path.join(augmented_output_dir, 'semantic_masks')
    aug_instance_dir = os.path.join(augmented_output_dir, 'instance_masks')
    aug_panoptic_dir = os.path.join(augmented_output_dir, 'panoptic_masks')
    
    # Create all base directories
    for dir_path in [aug_images_dir, aug_semantic_dir, aug_instance_dir, aug_panoptic_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Find all semantic masks
    semantic_masks_dir = os.path.join(masks_dir, 'semantic_masks')
    instance_masks_dir = os.path.join(masks_dir, 'instance_masks')
    panoptic_masks_dir = os.path.join(masks_dir, 'panoptic_masks')
    
    total_augmented = 0
    augmentation_log = []
    
    def find_image_file_in_plant_dir(base_name, plant_type, images_dir):
        """
        Find image file specifically in the plant type directory
        """
        # First try in plant-specific directory
        plant_img_dir = os.path.join(images_dir, plant_type)
        if os.path.exists(plant_img_dir):
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                img_path = os.path.join(plant_img_dir, f"{base_name}{ext}")
                if os.path.exists(img_path):
                    return img_path, ext
        
        # If not found, search recursively but prioritize matching plant type in path
        for root, _, files in os.walk(images_dir):
            # Check if this directory contains the plant type
            if plant_type.lower() in root.lower() or plant_type in root:
                for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                    filename = f"{base_name}{ext}"
                    if filename in files:
                        return os.path.join(root, filename), ext
        
        # Last resort: search everywhere
        for root, _, files in os.walk(images_dir):
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                filename = f"{base_name}{ext}"
                if filename in files:
                    return os.path.join(root, filename), ext
        
        return None, None
    
    # Debug: Print directory structure
    print(f"Semantic masks directory: {semantic_masks_dir}")
    print(f"Original images directory: {original_images_dir}")
    
    # Get list of plant types that should be processed
    processed_plants = []
    for plant_dir in os.listdir(semantic_masks_dir):
        plant_path = os.path.join(semantic_masks_dir, plant_dir)
        if os.path.isdir(plant_path):
            processed_plants.append(plant_dir)
    
    print(f"Plant types found in masks: {processed_plants}")
    
    for root, dirs, files in os.walk(semantic_masks_dir):
        for filename in files:
            if filename.endswith('_semantic.png'):
                base_name = filename.replace('_semantic.png', '')
                rel_path = os.path.relpath(root, semantic_masks_dir)
                plant_type = rel_path if rel_path != '.' else 'unknown'
                
                print(f"Processing {plant_type}/{base_name}")
                
                # Create plant-specific directories
                for dir_path in [aug_images_dir, aug_semantic_dir, aug_instance_dir, aug_panoptic_dir]:
                    plant_dir = os.path.join(dir_path, plant_type)
                    os.makedirs(plant_dir, exist_ok=True)
                
                # Find original image using improved search
                orig_image_path, orig_ext = find_image_file_in_plant_dir(base_name, plant_type, original_images_dir)
                
                if not orig_image_path:
                    print(f"Warning: Original image not found for {plant_type}/{base_name}")
                    # List available files for debugging
                    plant_img_dir = os.path.join(original_images_dir, plant_type)
                    if os.path.exists(plant_img_dir):
                        available_files = os.listdir(plant_img_dir)[:5]  # Show first 5 files
                        print(f"  Available files in {plant_type}: {available_files}")
                    continue
                
                print(f"Found image: {orig_image_path}")
                
                # Build paths for all mask types
                semantic_path = os.path.join(root, filename)
                instance_path = os.path.join(instance_masks_dir, plant_type, f"{base_name}_instance.png")
                panoptic_path = os.path.join(panoptic_masks_dir, plant_type, f"{base_name}_panoptic.png")
                
                # Check if all required files exist
                if not all(os.path.exists(p) for p in [orig_image_path, semantic_path, instance_path, panoptic_path]):
                    print(f"Warning: Not all files found for {base_name}")
                    print(f"  Image: {os.path.exists(orig_image_path)}")
                    print(f"  Semantic: {os.path.exists(semantic_path)}")
                    print(f"  Instance: {os.path.exists(instance_path)}")
                    print(f"  Panoptic: {os.path.exists(panoptic_path)}")
                    continue
                
                # Load original image and masks
                try:
                    original_image = Image.open(orig_image_path).convert('RGB')
                    semantic_mask = Image.open(semantic_path)
                    instance_mask = Image.open(instance_path)
                    panoptic_mask = Image.open(panoptic_path)
                    
                    # Copy originals with _original suffix
                    original_image.save(os.path.join(aug_images_dir, plant_type, f"{base_name}_original{orig_ext}"))
                    semantic_mask.save(os.path.join(aug_semantic_dir, plant_type, f"{base_name}_original_semantic.png"))
                    instance_mask.save(os.path.join(aug_instance_dir, plant_type, f"{base_name}_original_instance.png"))
                    panoptic_mask.save(os.path.join(aug_panoptic_dir, plant_type, f"{base_name}_original_panoptic.png"))
                    
                    # Generate augmentations
                    for aug_idx in range(augmentations_per_image):
                        aug_image, aug_semantic, aug_instance, augmentations = augment_image_and_masks(
                            original_image, semantic_mask, instance_mask, augmentation_params
                        )
                        
                        # Recreate panoptic mask from augmented semantic and instance masks
                        aug_panoptic = create_panoptic_from_masks(aug_semantic, aug_instance)
                        
                        # Save augmented versions
                        aug_suffix = f"_aug{aug_idx:02d}"
                        aug_image.save(os.path.join(aug_images_dir, plant_type, f"{base_name}{aug_suffix}{orig_ext}"))
                        aug_semantic.save(os.path.join(aug_semantic_dir, plant_type, f"{base_name}{aug_suffix}_semantic.png"))
                        aug_instance.save(os.path.join(aug_instance_dir, plant_type, f"{base_name}{aug_suffix}_instance.png"))
                        aug_panoptic.save(os.path.join(aug_panoptic_dir, plant_type, f"{base_name}{aug_suffix}_panoptic.png"))
                        
                        # Copy instance info JSON with updated filename
                        info_path = os.path.join(instance_masks_dir, plant_type, f"{base_name}_instances.json")
                        if os.path.exists(info_path):
                            with open(info_path, 'r') as f:
                                info_data = json.load(f)
                            info_data['augmentations'] = augmentations
                            info_data['original_filename'] = info_data['filename']
                            info_data['filename'] = f"{base_name}{aug_suffix}{orig_ext}"
                            
                            with open(os.path.join(aug_instance_dir, plant_type, f"{base_name}{aug_suffix}_instances.json"), 'w') as f:
                                json.dump(info_data, f, indent=2)
                        
                        augmentation_log.append({
                            'original': base_name,
                            'augmented': f"{base_name}{aug_suffix}",
                            'augmentations': augmentations,
                            'plant_type': plant_type
                        })
                        
                        total_augmented += 1
                        
                    print(f"Generated {augmentations_per_image} augmentations for {plant_type}/{base_name}")
                    
                except Exception as e:
                    print(f"Error processing {plant_type}/{base_name}: {e}")
                    import traceback
                    traceback.print_exc()
    
    # Save augmentation log
    log_path = os.path.join(augmented_output_dir, 'augmentation_log.json')
    with open(log_path, 'w') as f:
        json.dump(augmentation_log, f, indent=2)
    
    print(f"\nAugmentation complete!")
    print(f"Total augmented images created: {total_augmented}")
    print(f"Augmentation log saved to: {log_path}")
    
    return total_augmented


def create_panoptic_from_masks(semantic_mask, instance_mask):
    """Recreate panoptic mask from semantic and instance masks"""
    semantic_array = np.array(semantic_mask)
    instance_array = np.array(instance_mask)
    
    # Create panoptic mask
    panoptic_array = np.zeros_like(instance_array, dtype=np.uint32)
    
    # For each unique instance
    unique_instances = np.unique(instance_array)
    for instance_id in unique_instances:
        if instance_id == 0:  # Skip background
            continue
        
        # Get the semantic class for this instance
        instance_pixels = instance_array == instance_id
        semantic_values = semantic_array[instance_pixels]
        
        # Use mode (most common) semantic value for this instance
        if len(semantic_values) > 0:
            semantic_class = np.bincount(semantic_values).argmax()
            panoptic_id = semantic_class * 1000 + instance_id
            panoptic_array[instance_pixels] = panoptic_id
    
    return Image.fromarray(panoptic_array)


def validate_augmented_dataset(augmented_dir):
    """Validate the augmented dataset structure and consistency"""
    issues = []
    
    images_dir = os.path.join(augmented_dir, 'images')
    semantic_dir = os.path.join(augmented_dir, 'semantic_masks')
    instance_dir = os.path.join(augmented_dir, 'instance_masks')
    panoptic_dir = os.path.join(augmented_dir, 'panoptic_masks')
    
    # Count files in each directory
    image_files = set()
    semantic_files = set()
    instance_files = set()
    panoptic_files = set()
    
    for root, _, files in os.walk(images_dir):
        for f in files:
            if f.endswith(('.jpg', '.jpeg', '.png')):
                base = os.path.splitext(f)[0]
                image_files.add(base)
    
    for root, _, files in os.walk(semantic_dir):
        for f in files:
            if f.endswith('_semantic.png'):
                base = f.replace('_semantic.png', '')
                semantic_files.add(base)
    
    for root, _, files in os.walk(instance_dir):
        for f in files:
            if f.endswith('_instance.png'):
                base = f.replace('_instance.png', '')
                instance_files.add(base)
    
    for root, _, files in os.walk(panoptic_dir):
        for f in files:
            if f.endswith('_panoptic.png'):
                base = f.replace('_panoptic.png', '')
                panoptic_files.add(base)
    
    # Check consistency
    if image_files != semantic_files:
        issues.append(f"Mismatch between images and semantic masks")
    if image_files != instance_files:
        issues.append(f"Mismatch between images and instance masks")
    if image_files != panoptic_files:
        issues.append(f"Mismatch between images and panoptic masks")
    
    print(f"\nValidation Results:")
    print(f"Images: {len(image_files)}")
    print(f"Semantic masks: {len(semantic_files)}")
    print(f"Instance masks: {len(instance_files)}")
    print(f"Panoptic masks: {len(panoptic_files)}")
    
    if issues:
        print("\nIssues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\nNo issues found! Dataset is consistent.")
    
    return len(issues) == 0


def visualize_instance_masks(semantic_path, instance_path, panoptic_path, save_path=None):
    """Visualize the three types of masks"""
    import matplotlib.pyplot as plt
    
    semantic_mask = np.array(Image.open(semantic_path))
    instance_mask = np.array(Image.open(instance_path))
    panoptic_mask = np.array(Image.open(panoptic_path))
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Semantic mask
    axes[0].imshow(semantic_mask, cmap='tab20')
    axes[0].set_title('Semantic Segmentation\n(Class per pixel)')
    axes[0].axis('off')
    
    # Instance mask
    axes[1].imshow(instance_mask, cmap='nipy_spectral')
    axes[1].set_title('Instance Segmentation\n(Unique ID per fruit)')
    axes[1].axis('off')
    
    # Panoptic mask
    axes[2].imshow(panoptic_mask, cmap='prism')
    axes[2].set_title('Panoptic Segmentation\n(Class + Instance)')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# Enhanced dataset class for instance segmentation training
class InstanceSegmentationDataset:
    """Dataset class that loads instance-aware masks"""
    
    def __init__(self, image_dir, semantic_dir, instance_dir, transform=None):
        self.image_dir = image_dir
        self.semantic_dir = semantic_dir
        self.instance_dir = instance_dir
        self.transform = transform
        
        self.samples = []
        
        # Find all image-mask pairs
        for root, dirs, files in os.walk(semantic_dir):
            for file in files:
                if file.endswith('_semantic.png'):
                    base_name = file.replace('_semantic.png', '')
                    
                    # Find corresponding files
                    semantic_path = os.path.join(root, file)
                    instance_path = os.path.join(root.replace(semantic_dir, instance_dir), 
                                               f"{base_name}_instance.png")
                    info_path = os.path.join(root.replace(semantic_dir, instance_dir), 
                                           f"{base_name}_instances.json")
                    
                    # Find original image
                    for ext in ['.jpg', '.jpeg', '.png']:
                        img_path = self.find_image_file(f"{base_name}{ext}", image_dir)
                        if img_path and os.path.exists(semantic_path) and os.path.exists(instance_path):
                            self.samples.append({
                                'image': img_path,
                                'semantic': semantic_path,
                                'instance': instance_path,
                                'info': info_path if os.path.exists(info_path) else None
                            })
                            break
        
        print(f"Found {len(self.samples)} image-mask pairs for instance training")
    
    def find_image_file(self, filename, base_dir):
        """Find image file in directory structure"""
        direct_path = os.path.join(base_dir, filename)
        if os.path.exists(direct_path):
            return direct_path
        for root, _, files in os.walk(base_dir):
            if filename in files:
                return os.path.join(root, filename)
        return None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image']).convert('RGB')
        
        # Load masks
        semantic_mask = Image.open(sample['semantic'])
        instance_mask = Image.open(sample['instance'])
        
        # Load instance info if available
        instance_info = None
        if sample['info'] and os.path.exists(sample['info']):
            with open(sample['info'], 'r') as f:
                instance_info = json.load(f)
        
        if self.transform:
            # Apply transforms to image and masks
            transformed = self.transform(
                image=np.array(image),
                masks=[np.array(semantic_mask), np.array(instance_mask)]
            )
            image = transformed['image']
            semantic_mask = transformed['masks'][0]
            instance_mask = transformed['masks'][1]
        
        return {
            'image': image,
            'semantic_mask': semantic_mask,
            'instance_mask': instance_mask,
            'instance_info': instance_info
        }


# Main execution
if __name__ == "__main__":
    json_file = "via_export_json(4).json"
    original_images_dir = "Pictures/"
    
    classes_to_skip = ['lettuce', 'raspberry', 'pear', 'pepper']
    
    print("=== STEP 1: Creating Instance-Aware Masks at Original Sizes ===")
    print(f"SKIPPING CLASSES: {classes_to_skip}")
    output_dir = "InstanceMasks/"
    
    processed, total_instances = create_instance_masks_from_json(
        json_file, original_images_dir, output_dir, skip_classes=classes_to_skip
    )
    
    print(f"\n   Successfully created instance masks!")
    print(f"   Output structure:")
    print(f"   {output_dir}/semantic_masks/  - Semantic segmentation masks")
    print(f"   {output_dir}/instance_masks/  - Instance segmentation masks + JSON info")
    print(f"   {output_dir}/panoptic_masks/  - Combined panoptic masks")
    
    print(f"\n   Statistics:")
    print(f"   Images processed: {processed}")
    print(f"   Individual fruits labeled: {total_instances}")
    if processed > 0:
        print(f"   Average fruits per image: {total_instances/processed:.1f}")
    
    print("\n=== STEP 2: Smart Resizing Images and Masks to 1200x1200 ===")
    resized_images_dir = "RePictures/"
    resized_masks_dir = "ReInstanceMasks/"
    target_resolution = (1200, 1200)
    
    smart_resize_instance_masks(
        original_images_dir, output_dir,
        resized_images_dir, resized_masks_dir,
        target_resolution
    )
    
    print("\n=== STEP 3: Creating Augmented Dataset ===")
    
    # Use the resized images and masks for augmentation
    augmented_output_dir = "AugmentedDataset/"
    
    # Create augmented dataset
    augmentations_per_image = 8  # This will give you 9x total (1 original + 8 augmented)
    
    total_augmented = create_augmented_dataset(
        resized_images_dir,
        resized_masks_dir,
        augmented_output_dir,
        augmentations_per_image=augmentations_per_image
    )
    
    print(f"\nCreated {total_augmented} augmented images!")
    print(f"Total dataset size: ~{total_augmented + processed} images")
    
    # Validate the augmented dataset
    print("\n=== STEP 4: Validating Augmented Dataset ===")
    is_valid = validate_augmented_dataset(augmented_output_dir)
    
    print("\n=== COMPLETE ===")
    print("You now have:")
    print("- Original size instance masks in InstanceMasks/")
    print("- 1200x1200 images in RePictures/")
    print("- 1200x1200 instance masks in ReInstanceMasks/")
    print(f"- Augmented dataset in AugmentedDataset/ (~{total_augmented + processed} total images)")
    print(f"EXCLUDED CLASSES: {classes_to_skip}")