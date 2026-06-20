import os
import argparse
import pickle
from pathlib import Path
import numpy as np
import scipy.io as sio
from PIL import Image, ImageDraw
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# Room types color mapping
# 0: LivingRoom
# 1: MasterRoom
# 2: Kitchen (Pink as requested!)
# 3: Bathroom
# 4: DiningRoom
# 5: ChildRoom
# 6: StudyRoom
# 7: SecondRoom
# 8: GuestRoom
# 9: Balcony
# 10: Entrance
# 11: Storage
# 12: Wall-in / Walk-in closet
ROOM_COLORS = {
    0: (255, 235, 156),   # LivingRoom - Light Warm Yellow
    1: (255, 179, 138),   # MasterRoom - Soft Coral/Orange
    2: (255, 192, 203),   # Kitchen - Pink
    3: (160, 210, 255),   # Bathroom - Light Sky Blue
    4: (220, 200, 170),   # DiningRoom - Warm Tan
    5: (230, 210, 250),   # ChildRoom - Lavender
    6: (170, 240, 190),   # StudyRoom - Mint Green
    7: (255, 218, 185),   # SecondRoom - Peach
    8: (210, 245, 210),   # GuestRoom - Light Mint
    9: (220, 220, 220),   # Balcony - Muted Grey
    10: (175, 235, 235),  # Entrance - Light Teal
    11: (240, 230, 160),  # Storage - Soft Mustard/Gold
    12: (240, 200, 240),  # Wall-in / Walk-in - Light Lilac/Orchid
}

# Default color for unknown room types
DEFAULT_COLOR = (200, 200, 200)

def extract_record_data(record):
    """Safely extracts boundary and room information from a matlab record."""
    boundary = np.asarray(record.boundary)
    
    # Extract rooms
    r_bounds = getattr(record, 'rBoundary', None)
    if r_bounds is None:
        r_bounds = []
    elif isinstance(r_bounds, np.ndarray) and r_bounds.dtype == object:
        r_bounds = list(r_bounds)
    elif isinstance(r_bounds, np.ndarray):
        r_bounds = [r_bounds]
    elif not isinstance(r_bounds, (list, tuple)):
        r_bounds = [r_bounds]

    r_types = getattr(record, 'rType', [])
    if isinstance(r_types, (int, np.integer)):
        r_types = [r_types]
    else:
        r_types = np.atleast_1d(r_types).tolist()
        
    return boundary, r_bounds, r_types

def render_layout(boundary, r_bounds, r_types, draw_rooms=False):
    """Renders a single layout using PIL and returns the Image object."""
    # Create white canvas (256x256)
    img = Image.new("RGB", (256, 256), "white")
    draw = ImageDraw.Draw(img)
    
    # 1. Fill room interiors if requested
    if draw_rooms and len(r_bounds) > 0:
        for idx, room in enumerate(r_bounds):
            room_pts = np.asarray(room)
            if len(room_pts) < 3:
                continue
            
            # Get room type and color
            rtype = r_types[idx] if idx < len(r_types) else -1
            color = ROOM_COLORS.get(rtype, DEFAULT_COLOR)
            
            # PIL polygon drawing
            pts_tuples = [(float(p[0]), float(p[1])) for p in room_pts]
            draw.polygon(pts_tuples, fill=color)
            
        # Draw room borders (internal walls)
        for room in r_bounds:
            room_pts = np.asarray(room)
            if len(room_pts) < 2:
                continue
            pts_tuples = [(float(p[0]), float(p[1])) for p in room_pts]
            pts_tuples.append(pts_tuples[0]) # Close loop
            draw.line(pts_tuples, fill="black", width=2)
            
    # 2. Draw external walls (from boundary)
    boundary_pts = boundary[:, :2]
    pts_tuples = [(float(p[0]), float(p[1])) for p in boundary_pts]
    pts_tuples.append(pts_tuples[0]) # Close loop
    draw.line(pts_tuples, fill="black", width=3)
    
    # 3. Draw entrance door (first segment of boundary)
    door_start = (float(boundary[0, 0]), float(boundary[0, 1]))
    door_end = (float(boundary[1, 0]), float(boundary[1, 1]))
    draw.line([door_start, door_end], fill="red", width=3)
    
    return img

def process_single_record(args):
    """Worker function for multiprocessing."""
    record_idx, record_data, save_boundary_dir, save_full_dir, is_train = args
    try:
        boundary, r_bounds, r_types = record_data
        
        # Render and save boundary-only image
        if save_boundary_dir:
            img_b = render_layout(boundary, r_bounds, r_types, draw_rooms=False)
            img_b.save(Path(save_boundary_dir) / f"{record_idx}.png")
            
        # Render and save full image with rooms
        if is_train and save_full_dir:
            img_f = render_layout(boundary, r_bounds, r_types, draw_rooms=True)
            img_f.save(Path(save_full_dir) / f"{record_idx}.png")
            
        return True
    except Exception as e:
        print(f"Error processing record {record_idx}: {e}")
        return False

def load_mat_file(file_path):
    print(f"Loading {file_path.name}...")
    mat = sio.loadmat(str(file_path), struct_as_record=False, squeeze_me=True)
    return mat['data']

def main():
    parser = argparse.ArgumentParser(description="Generate RPLAN layout images.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of records processed per dataset.")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers.")
    args = parser.parse_args()

    base_dir = Path(r"c:\Users\hanbi\Downloads\RPLAN dataset")
    data_dir = base_dir / "Network" / "data"
    out_dir = base_dir / "Network" / "generated_images"
    
    # Create output directories
    paths = {
        'train_boundary': out_dir / 'train_boundary',
        'train_full': out_dir / 'train_full',
        'valid_boundary': out_dir / 'valid_boundary',
        'test_boundary': out_dir / 'test_boundary'
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
        
    datasets = [
        ('data_train.mat', paths['train_boundary'], paths['train_full'], True),
        ('data_valid.mat', paths['valid_boundary'], None, False),
        ('data_test.mat', paths['test_boundary'], None, False),
    ]
    
    for filename, boundary_dir, full_dir, is_train in datasets:
        file_path = data_dir / filename
        if not file_path.exists():
            print(f"File not found: {file_path}")
            continue
            
        records = load_mat_file(file_path)
        count = len(records)
        if args.limit:
            count = min(args.limit, count)
            records = records[:count]
            
        print(f"Preparing data for {filename} ({count} records)...")
        tasks = []
        for idx in range(count):
            record = records[idx]
            if isinstance(record, np.ndarray):
                record = record.flat[0]
            # Safely extract values in main thread to avoid pickling Matlab objects
            boundary, r_bounds, r_types = extract_record_data(record)
            tasks.append((idx, (boundary, r_bounds, r_types), boundary_dir, full_dir, is_train))
            
        print(f"Processing and rendering images for {filename} using multiprocessing...")
        success_count = 0
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = list(tqdm(executor.map(process_single_record, tasks), total=len(tasks)))
            success_count = sum(1 for r in results if r)
            
        print(f"Completed {filename}: {success_count}/{count} successfully saved.\n")

if __name__ == "__main__":
    main()
