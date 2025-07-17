import json
import numpy as np
from PIL import Image, ImageDraw
import os
from pathlib import Path

def create_instance_masks_from_json(json_file_path, images_dir, output_dir):
    """
    Convert VIA JSON annotations to instance masks preserving individual fruit instances.
    Each instance gets a unique ID while maintaining class information.
    """
    with open(json_file_path, 'r') as f:
        annotations = json.load(f)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output subdirectories
    semantic_dir = os.path.join(output_dir, 'semantic_masks')
    instance_dir = os.path.join(output_dir, 'instance_masks')
    panoptic_dir = os.path.join(output_dir, 'panoptic_masks')
    os.makedirs(semantic_dir, exist_ok=True)
    os.makedirs(instance_dir, exist_ok=True)
    os.makedirs(panoptic_dir, exist_ok=True)

    # Class mapping for semantic segmentation
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
    total_instances = 0
    
    # Global instance ID counter (across all images)
    global_instance_id = 1

    for image_data in annotations.values():
        if not isinstance(image_data, dict) or 'filename' not in image_data:
            continue

        filename = image_data['filename']
        plant_type = image_data.get('file_attributes', {}).get('plant', 'unknown').lower()
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

        # Create plant-specific output directories
        plant_semantic_dir = os.path.join(semantic_dir, plant_type)
        plant_instance_dir = os.path.join(instance_dir, plant_type)
        plant_panoptic_dir = os.path.join(panoptic_dir, plant_type)
        
        for dir_path in [plant_semantic_dir, plant_instance_dir, plant_panoptic_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # Initialize masks
        semantic_mask = np.zeros((height, width), dtype=np.uint8)
        instance_mask = np.zeros((height, width), dtype=np.uint16)  # uint16 for more instances
        panoptic_mask = np.zeros((height, width), dtype=np.uint32)  # Combines class + instance
        
        # Store instance information for this image
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
            
            # Get semantic class ID
            if plant_type in class_map and ripeness in class_map[plant_type]:
                semantic_class_id = class_map[plant_type][ripeness]
            else:
                semantic_class_id = 28  # Obscured/unknown
            
            # Create temporary mask for this instance
            temp_img = Image.new('L', (width, height), 0)
            ImageDraw.Draw(temp_img).polygon(polygon_coords, fill=255)
            instance_pixels = np.array(temp_img) > 0
            
            if np.sum(instance_pixels) < 10:  # Skip very small instances
                continue
            
            # Assign IDs to masks
            semantic_mask[instance_pixels] = semantic_class_id
            instance_mask[instance_pixels] = global_instance_id
            
            # Panoptic mask: combine semantic class and instance ID
            # Format: (semantic_class_id * 1000) + instance_id
            panoptic_id = semantic_class_id * 1000 + global_instance_id
            panoptic_mask[instance_pixels] = panoptic_id
            
            # Get bbox and convert to Python ints
            bbox = get_bbox_from_mask(instance_pixels)
            
            # Store instance information - Convert numpy types to Python types
            instances_info.append({
                'instance_id': int(global_instance_id),
                'semantic_class_id': int(semantic_class_id),
                'plant_type': plant_type,
                'ripeness': ripeness,
                'area': int(np.sum(instance_pixels)),  # Convert numpy int64 to Python int
                'bbox': [int(x) for x in bbox]  # Convert all bbox values to Python ints
            })
            
            global_instance_id += 1
            image_instance_count += 1
            total_instances += 1

        # Save masks
        base_filename = Path(filename).stem
        
        # Semantic mask
        semantic_path = os.path.join(plant_semantic_dir, f"{base_filename}_semantic.png")
        Image.fromarray(semantic_mask).save(semantic_path)
        
        # Instance mask
        instance_path = os.path.join(plant_instance_dir, f"{base_filename}_instance.png")
        Image.fromarray(instance_mask).save(instance_path)
        
        # Panoptic mask
        panoptic_path = os.path.join(plant_panoptic_dir, f"{base_filename}_panoptic.png")
        Image.fromarray(panoptic_mask).save(panoptic_path)
        
        # Save instance information as JSON
        info_path = os.path.join(plant_instance_dir, f"{base_filename}_instances.json")
        with open(info_path, 'w') as f:
            json.dump({
                'filename': filename,
                'plant_type': plant_type,
                'image_size': [int(width), int(height)],  # Convert to Python ints
                'num_instances': int(image_instance_count),  # Convert to Python int
                'instances': instances_info
            }, f, indent=2)

        processed_count += 1
        print(f"Saved masks for {filename}: {image_instance_count} instances")

    print(f"\nInstance Mask Generation Summary:")
    print(f"Processed: {processed_count} images")
    print(f"Total instances: {total_instances} individual fruits/vegetables")
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
    
    print("=== STEP 1: Creating Instance-Aware Masks at Original Sizes ===")
    output_dir = "InstanceMasks/"
    
    processed, total_instances = create_instance_masks_from_json(
        json_file, original_images_dir, output_dir
    )
    
    print(f"\n   Successfully created instance masks!")
    print(f"   Output structure:")
    print(f"   {output_dir}/semantic_masks/  - Semantic segmentation masks")
    print(f"   {output_dir}/instance_masks/  - Instance segmentation masks + JSON info")
    print(f"   {output_dir}/panoptic_masks/  - Combined panoptic masks")
    
    print(f"\n   Statistics:")
    print(f"   Images processed: {processed}")
    print(f"   Individual fruits labeled: {total_instances}")
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
    
    print("\n=== COMPLETE ===")
    print("You now have:")
    print("- Original size instance masks in InstanceMasks/")
    print("- 1200x1200 images in RePictures/")
    print("- 1200x1200 instance masks in ReInstanceMasks/")
    print("  - ReInstanceMasks/semantic_masks/")
    print("  - ReInstanceMasks/instance_masks/")
    print("  - ReInstanceMasks/panoptic_masks/")