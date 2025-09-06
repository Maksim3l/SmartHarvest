import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

def analyze_via_annotations(json_file_path, skip_classes=None):
    """
    Comprehensive analysis of VIA JSON annotations for dataset statistics.
    """
    if skip_classes is None:
        skip_classes = []
    
    skip_classes = [cls.lower() for cls in skip_classes]
    
    # Load annotations
    with open(json_file_path, 'r') as f:
        annotations = json.load(f)
    
    # Initialize statistics collectors
    stats = {
        'total_images': 0,
        'total_annotations': 0,
        'images_per_plant': defaultdict(int),
        'annotations_per_plant': defaultdict(int),
        'ripeness_distribution': defaultdict(lambda: defaultdict(int)),
        'overall_ripeness': defaultdict(int),
        'avg_annotations_per_image': defaultdict(list),
        'polygon_complexity': [],
        'occlusion_stats': defaultdict(int),
        'image_sizes': [],
        'skipped_images': 0,
        'null_regions': 0,
        'bbox_sizes': [],
        'annotation_areas': []
    }
    
    # Process each image
    for image_id, image_data in annotations.items():
        if not isinstance(image_data, dict) or 'filename' not in image_data:
            continue
        
        filename = image_data['filename']
        plant_type = image_data.get('file_attributes', {}).get('plant', 'unknown').lower()
        regions = image_data.get('regions', [])
        
        # Skip if plant type in skip list
        if plant_type in skip_classes:
            stats['skipped_images'] += 1
            continue
        
        # Count image
        stats['total_images'] += 1
        stats['images_per_plant'][plant_type] += 1
        
        # Track image size if available
        if 'size' in image_data:
            stats['image_sizes'].append(image_data['size'])
        
        # Count valid annotations for this image
        valid_annotations = 0
        
        # Process regions
        for region in regions:
            if region is None:
                stats['null_regions'] += 1
                continue
            
            shape_attrs = region.get('shape_attributes', {})
            region_attrs = region.get('region_attributes', {})
            
            # Only process polygon annotations
            if shape_attrs.get('name') != 'polygon':
                continue
            
            x_points = shape_attrs.get('all_points_x', [])
            y_points = shape_attrs.get('all_points_y', [])
            
            # Validate polygon
            if len(x_points) < 3 or len(x_points) != len(y_points):
                continue
            
            # Count annotation
            stats['total_annotations'] += 1
            valid_annotations += 1
            stats['annotations_per_plant'][plant_type] += 1
            
            # Track polygon complexity
            stats['polygon_complexity'].append(len(x_points))
            
            # Get ripeness
            ripeness = region_attrs.get('ripeness_factor', 'unknown')
            stats['ripeness_distribution'][plant_type][ripeness] += 1
            stats['overall_ripeness'][ripeness] += 1
            
            # Calculate bounding box and area
            if x_points and y_points:
                min_x, max_x = min(x_points), max(x_points)
                min_y, max_y = min(y_points), max(y_points)
                bbox_width = max_x - min_x
                bbox_height = max_y - min_y
                stats['bbox_sizes'].append((bbox_width, bbox_height))
                
                # Approximate area using shoelace formula
                area = calculate_polygon_area(x_points, y_points)
                stats['annotation_areas'].append(area)
            
            # Check for occlusion indicators (this is a heuristic)
            # You might need to adjust based on your annotation scheme
            if 'occluded' in region_attrs:
                if region_attrs['occluded']:
                    stats['occlusion_stats']['occluded'] += 1
                else:
                    stats['occlusion_stats']['visible'] += 1
        
        # Track annotations per image
        if valid_annotations > 0:
            stats['avg_annotations_per_image'][plant_type].append(valid_annotations)
    
    # Calculate averages
    results = {
        'Total Images': stats['total_images'],
        'Total Annotations': stats['total_annotations'],
        'Skipped Images': stats['skipped_images'],
        'Null Regions': stats['null_regions'],
        'Images per Plant Type': dict(stats['images_per_plant']),
        'Annotations per Plant Type': dict(stats['annotations_per_plant']),
        'Ripeness Distribution': dict(stats['ripeness_distribution']),
        'Overall Ripeness': dict(stats['overall_ripeness']),
        'Average Annotations per Image': {},
        'Polygon Complexity': {},
        'Bounding Box Stats': {},
        'Area Stats': {}
    }
    
    # Calculate averages per plant type
    for plant, counts in stats['avg_annotations_per_image'].items():
        if counts:
            results['Average Annotations per Image'][plant] = {
                'mean': np.mean(counts),
                'std': np.std(counts),
                'min': min(counts),
                'max': max(counts)
            }
    
    # Polygon complexity stats
    if stats['polygon_complexity']:
        results['Polygon Complexity'] = {
            'mean_vertices': np.mean(stats['polygon_complexity']),
            'std_vertices': np.std(stats['polygon_complexity']),
            'min_vertices': min(stats['polygon_complexity']),
            'max_vertices': max(stats['polygon_complexity'])
        }
    
    # Bounding box stats
    if stats['bbox_sizes']:
        widths = [w for w, h in stats['bbox_sizes']]
        heights = [h for w, h in stats['bbox_sizes']]
        results['Bounding Box Stats'] = {
            'mean_width': np.mean(widths),
            'mean_height': np.mean(heights),
            'std_width': np.std(widths),
            'std_height': np.std(heights)
        }
    
    # Area stats
    if stats['annotation_areas']:
        results['Area Stats'] = {
            'mean_area': np.mean(stats['annotation_areas']),
            'std_area': np.std(stats['annotation_areas']),
            'min_area': min(stats['annotation_areas']),
            'max_area': max(stats['annotation_areas'])
        }
    
    return results

def calculate_polygon_area(x_points, y_points):
    """Calculate polygon area using shoelace formula."""
    n = len(x_points)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += x_points[i] * y_points[j]
        area -= x_points[j] * y_points[i]
    return abs(area) / 2.0

def print_detailed_statistics(stats):
    """Print formatted statistics report."""
    print("\n" + "="*60)
    print("DATASET STATISTICS REPORT")
    print("="*60)
    
    print(f"\n📊 OVERVIEW")
    print(f"  Total Images: {stats['Total Images']}")
    print(f"  Total Annotations: {stats['Total Annotations']}")
    if stats['Total Images'] > 0:
        print(f"  Average Annotations per Image: {stats['Total Annotations']/stats['Total Images']:.2f}")
    print(f"  Skipped Images: {stats['Skipped Images']}")
    print(f"  Null Regions Found: {stats['Null Regions']}")
    
    print(f"\n🌱 PLANT TYPE DISTRIBUTION")
    for plant, count in stats['Images per Plant Type'].items():
        annot_count = stats['Annotations per Plant Type'].get(plant, 0)
        avg_annot = stats['Average Annotations per Image'].get(plant, {})
        print(f"\n  {plant.upper()}:")
        print(f"    Images: {count}")
        print(f"    Total Annotations: {annot_count}")
        if avg_annot:
            print(f"    Avg Annotations/Image: {avg_annot['mean']:.2f} ± {avg_annot['std']:.2f}")
            print(f"    Range: [{avg_annot['min']}, {avg_annot['max']}]")
    
    print(f"\n🍎 RIPENESS DISTRIBUTION")
    print(f"\n  Overall:")
    for ripeness, count in stats['Overall Ripeness'].items():
        percentage = (count / stats['Total Annotations'] * 100) if stats['Total Annotations'] > 0 else 0
        print(f"    {ripeness}: {count} ({percentage:.1f}%)")
    
    print(f"\n  By Plant Type:")
    for plant, ripeness_dict in stats['Ripeness Distribution'].items():
        print(f"\n    {plant.upper()}:")
        total_plant = sum(ripeness_dict.values())
        for ripeness, count in ripeness_dict.items():
            percentage = (count / total_plant * 100) if total_plant > 0 else 0
            print(f"      {ripeness}: {count} ({percentage:.1f}%)")
    
    if stats['Polygon Complexity']:
        print(f"\n📐 ANNOTATION COMPLEXITY")
        print(f"  Average Vertices per Polygon: {stats['Polygon Complexity']['mean_vertices']:.1f} ± {stats['Polygon Complexity']['std_vertices']:.1f}")
        print(f"  Range: [{stats['Polygon Complexity']['min_vertices']}, {stats['Polygon Complexity']['max_vertices']}]")
    
    if stats['Bounding Box Stats']:
        print(f"\n📦 BOUNDING BOX STATISTICS")
        print(f"  Mean Width: {stats['Bounding Box Stats']['mean_width']:.1f} ± {stats['Bounding Box Stats']['std_width']:.1f}")
        print(f"  Mean Height: {stats['Bounding Box Stats']['mean_height']:.1f} ± {stats['Bounding Box Stats']['std_height']:.1f}")
    
    if stats['Area Stats']:
        print(f"\n📏 ANNOTATION AREA STATISTICS")
        print(f"  Mean Area: {stats['Area Stats']['mean_area']:.1f} ± {stats['Area Stats']['std_area']:.1f} pixels²")
        print(f"  Range: [{stats['Area Stats']['min_area']:.1f}, {stats['Area Stats']['max_area']:.1f}] pixels²")
    
    print("\n" + "="*60 + "\n")

def generate_latex_table(stats):
    """Generate LaTeX table code for the statistics."""
    print("\n% LaTeX Table for Dataset Statistics")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Dataset Composition and Annotation Statistics}")
    print("\\begin{tabular}{|l|c|c|c|}")
    print("\\hline")
    print("\\textbf{Plant Type} & \\textbf{Images} & \\textbf{Annotations} & \\textbf{Avg/Image} \\\\")
    print("\\hline")
    
    for plant, count in stats['Images per Plant Type'].items():
        annot_count = stats['Annotations per Plant Type'].get(plant, 0)
        avg_annot = stats['Average Annotations per Image'].get(plant, {})
        avg_val = avg_annot.get('mean', 0) if avg_annot else 0
        print(f"{plant.capitalize()} & {count} & {annot_count} & {avg_val:.1f} \\\\")
    
    print("\\hline")
    print(f"\\textbf{{Total}} & \\textbf{{{stats['Total Images']}}} & \\textbf{{{stats['Total Annotations']}}} & \\textbf{{{stats['Total Annotations']/stats['Total Images']:.1f}}} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\label{tab:dataset_stats}")
    print("\\end{table}")
    
    # Generate ripeness distribution table
    print("\n% LaTeX Table for Ripeness Distribution")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Ripeness Distribution Across Plant Types}")
    print("\\begin{tabular}{|l|c|c|c|c|}")
    print("\\hline")
    print("\\textbf{Plant Type} & \\textbf{Ripe} & \\textbf{Unripe} & \\textbf{Spoiled} & \\textbf{Total} \\\\")
    print("\\hline")
    
    for plant, ripeness_dict in stats['Ripeness Distribution'].items():
        ripe = ripeness_dict.get('ripe', 0)
        unripe = ripeness_dict.get('unripe', 0)
        spoiled = ripeness_dict.get('spoiled', 0)
        total = ripe + unripe + spoiled
        print(f"{plant.capitalize()} & {ripe} & {unripe} & {spoiled} & {total} \\\\")
    
    print("\\hline")
    overall = stats['Overall Ripeness']
    print(f"\\textbf{{Total}} & \\textbf{{{overall.get('ripe', 0)}}} & \\textbf{{{overall.get('unripe', 0)}}} & \\textbf{{{overall.get('spoiled', 0)}}} & \\textbf{{{stats['Total Annotations']}}} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\label{tab:ripeness_dist}")
    print("\\end{table}")

def visualize_statistics(stats):
    """Create visualization plots for the statistics."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plant type distribution
    plants = list(stats['Images per Plant Type'].keys())
    image_counts = list(stats['Images per Plant Type'].values())
    
    ax = axes[0, 0]
    ax.bar(plants, image_counts, color='green', alpha=0.7)
    ax.set_xlabel('Plant Type')
    ax.set_ylabel('Number of Images')
    ax.set_title('Images per Plant Type')
    ax.tick_params(axis='x', rotation=45)
    
    # Ripeness distribution
    ax = axes[0, 1]
    ripeness_types = list(stats['Overall Ripeness'].keys())
    ripeness_counts = list(stats['Overall Ripeness'].values())
    colors = {'ripe': 'green', 'unripe': 'yellow', 'spoiled': 'red'}
    bar_colors = [colors.get(r, 'gray') for r in ripeness_types]
    ax.bar(ripeness_types, ripeness_counts, color=bar_colors, alpha=0.7)
    ax.set_xlabel('Ripeness State')
    ax.set_ylabel('Number of Annotations')
    ax.set_title('Overall Ripeness Distribution')
    
    # Annotations per image distribution
    ax = axes[1, 0]
    for plant, avg_data in stats['Average Annotations per Image'].items():
        ax.bar(plant, avg_data['mean'], yerr=avg_data['std'], capsize=5, alpha=0.7)
    ax.set_xlabel('Plant Type')
    ax.set_ylabel('Average Annotations per Image')
    ax.set_title('Average Annotation Density')
    ax.tick_params(axis='x', rotation=45)
    
    # Ripeness by plant type (stacked bar)
    ax = axes[1, 1]
    plants = list(stats['Ripeness Distribution'].keys())
    ripe_counts = [stats['Ripeness Distribution'][p].get('ripe', 0) for p in plants]
    unripe_counts = [stats['Ripeness Distribution'][p].get('unripe', 0) for p in plants]
    spoiled_counts = [stats['Ripeness Distribution'][p].get('spoiled', 0) for p in plants]
    
    x = np.arange(len(plants))
    width = 0.6
    
    ax.bar(x, ripe_counts, width, label='Ripe', color='green', alpha=0.7)
    ax.bar(x, unripe_counts, width, bottom=ripe_counts, label='Unripe', color='yellow', alpha=0.7)
    ax.bar(x, spoiled_counts, width, bottom=np.array(ripe_counts)+np.array(unripe_counts), 
           label='Spoiled', color='red', alpha=0.7)
    
    ax.set_xlabel('Plant Type')
    ax.set_ylabel('Number of Annotations')
    ax.set_title('Ripeness Distribution by Plant Type')
    ax.set_xticks(x)
    ax.set_xticklabels(plants, rotation=45)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('dataset_statistics.png', dpi=150, bbox_inches='tight')
    plt.show()

# Main execution
if __name__ == "__main__":
    # Configuration
    json_file = "via_export_json(4).json"  # Your VIA JSON file
    skip_classes = []  # Add any classes to skip
    
    # Analyze annotations
    print("Analyzing VIA annotations...")
    stats = analyze_via_annotations(json_file, skip_classes)
    
    # Print detailed statistics
    print_detailed_statistics(stats)
    
    # Generate LaTeX tables
    generate_latex_table(stats)
    
    # Create visualizations
    print("\nGenerating visualization plots...")
    visualize_statistics(stats)
    
    # Save statistics to JSON for later use
    with open('dataset_statistics.json', 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj
        
        json.dump(convert_numpy(stats), f, indent=2)
    
    print("\nStatistics saved to 'dataset_statistics.json'")
    print("Visualization saved to 'dataset_statistics.png'")