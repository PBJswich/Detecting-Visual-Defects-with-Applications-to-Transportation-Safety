#!/usr/bin/env python3
"""
Convert XML annotations (PASCAL VOC format) to YOLO format (.txt files)

YOLO format: 
- One .txt file per image
- Each line: class_id x_center y_center width height
- All values are normalized to [0, 1]
- x_center, y_center are center coordinates (not top-left corner)
"""

import xml.etree.ElementTree as ET
import os
from pathlib import Path
import argparse

def convert_bbox_to_yolo(xmin, ymin, xmax, ymax, img_width, img_height):
    """
    Convert PASCAL VOC bbox format to YOLO format
    
    PASCAL VOC: [xmin, ymin, xmax, ymax] in pixel coordinates
    YOLO: [x_center, y_center, width, height] in normalized [0, 1] coordinates
    
    Returns: (x_center, y_center, width, height) normalized
    """
    # Calculate center, width, and height
    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    width = xmax - xmin
    height = ymax - ymin
    
    # Normalize coordinates
    x_center = x_center / img_width
    y_center = y_center / img_height
    width = width / img_width
    height = height / img_height
    
    return x_center, y_center, width, height

def parse_xml_file(xml_path):
    """
    Parse XML file and extract bounding box information
    
    Returns:
        - img_width: image width
        - img_height: image height
        - bboxes: list of (class_id, x_center, y_center, width, height) normalized
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Get image dimensions
    size = root.find('size')
    img_width = int(size.find('width').text)
    img_height = int(size.find('height').text)
    
    # Parse all objects (cracks)
    bboxes = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        
        # Convert to YOLO format
        x_center, y_center, width, height = convert_bbox_to_yolo(
            xmin, ymin, xmax, ymax, img_width, img_height
        )
        
        # Class ID for crack (assuming single class = 0)
        class_id = 0
        
        bboxes.append((class_id, x_center, y_center, width, height))
    
    return img_width, img_height, bboxes

def convert_xml_to_yolo(xml_dir, output_dir):
    """
    Convert all XML files in xml_dir to YOLO format .txt files in output_dir
    """
    xml_dir = Path(xml_dir)
    output_dir = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all XML files
    xml_files = sorted(xml_dir.glob('*.xml'))
    
    print(f"Found {len(xml_files)} XML files in {xml_dir}")
    print(f"Output directory: {output_dir}\n")
    
    conversion_stats = {
        'total': 0,
        'success': 0,
        'failed': 0,
        'total_bboxes': 0,
        'images_with_bboxes': 0
    }
    
    for xml_file in xml_files:
        try:
            conversion_stats['total'] += 1
            
            # Parse XML file
            img_width, img_height, bboxes = parse_xml_file(xml_file)
            
            if len(bboxes) == 0:
                print(f"Warning: No bounding boxes found in {xml_file.name}")
                conversion_stats['success'] += 1
                # Create empty output file
                output_file = output_dir / f"{xml_file.stem}.txt"
                output_file.write_text("")
                continue
            
            # Create output file with same name but .txt extension
            output_file = output_dir / f"{xml_file.stem}.txt"
            
            # Write YOLO format annotations
            with open(output_file, 'w') as f:
                for class_id, x_center, y_center, width, height in bboxes:
                    # Format: class_id x_center y_center width height
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            conversion_stats['success'] += 1
            conversion_stats['total_bboxes'] += len(bboxes)
            if len(bboxes) > 0:
                conversion_stats['images_with_bboxes'] += 1
            
            print(f"✓ Converted {xml_file.name} -> {output_file.name} ({len(bboxes)} bboxes)")
            
        except Exception as e:
            conversion_stats['failed'] += 1
            print(f"✗ Error converting {xml_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Conversion Summary:")
    print("="*60)
    print(f"Total XML files processed: {conversion_stats['total']}")
    print(f"Successfully converted: {conversion_stats['success']}")
    print(f"Failed conversions: {conversion_stats['failed']}")
    print(f"Total bounding boxes: {conversion_stats['total_bboxes']}")
    print(f"Images with bounding boxes: {conversion_stats['images_with_bboxes']}")
    print(f"Average bboxes per image: {conversion_stats['total_bboxes'] / max(conversion_stats['success'], 1):.2f}")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Convert XML annotations to YOLO format')
    parser.add_argument(
        '--xml-dir',
        type=str,
        default='SUT-Crack/2-Object Detection',
        help='Directory containing XML annotation files (default: SUT-Crack/2-Object Detection)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='SUT-Crack/yolo_annotations',
        help='Output directory for YOLO format .txt files (default: SUT-Crack/yolo_annotations)'
    )
    
    args = parser.parse_args()
    
    print("XML to YOLO Annotation Converter")
    print("="*60)
    print(f"Input directory: {args.xml_dir}")
    print(f"Output directory: {args.output_dir}")
    print("="*60 + "\n")
    
    convert_xml_to_yolo(args.xml_dir, args.output_dir)
    
    print("\nConversion complete!")
    print(f"\nYOLO format annotations saved to: {args.output_dir}")
    print("\nYOLO format format:")
    print("- One .txt file per image")
    print("- Each line: class_id x_center y_center width height")
    print("- All values are normalized to [0, 1]")
    print("- class_id: 0 for 'Crack'")

if __name__ == "__main__":
    main()
