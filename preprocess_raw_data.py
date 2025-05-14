import cv2
import numpy as np
from scipy.io import loadmat
import json
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
import src.utils.helpers as helpers

def prepare_full_video(video_path, label_path, output_dir_split, frame_size, subsample_factor):
    try:
        tlabs = loadmat(label_path)["tlabs"].ravel() #mở file label (ravel là làm dẹt thành mảng 1D)
    except FileNotFoundError:
        print(f"Label file not found: {label_path}")
        return 0, 0, False
    except Exception as e:
        print(f"loading label file {label_path}: {e}")
        return 0, 0, False
        
    video_id = Path(video_path).stem.replace("_crop", "") #lấy đúng số của video (1)

    video_folder = output_dir_split / "frames" #đưa file frames vào đúng đường dẫn bên trong output_dir_split
    annotation_folder = output_dir_split / "annotations" #Path(data/full_videos/train/annotations)
    video_folder.mkdir(parents=True, exist_ok=True) #tạo file frames nếu chưa có
    annotation_folder.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path)) #Dùng cv2 để đọc video
    if not cap.isOpened(): #nếu ko có
        print(f"Cannot open video file: {video_path}") #ghi đường dẫn 
        return 0, 0, False
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # tổng số video đã được đọc
    fps = float(cap.get(cv2.CAP_PROP_FPS)) # lấy fps của video

    frames = []
    frame_indices = []
    idx = 0
    while True:
        ret, frame = cap.read() #bắt đầu đọc (ret là T/F để biết có đang đọc đc ko)(cap sẽ chạy +1 frame nữa khi qua 1 vòng while -> chạy tất cả frame)
        if not ret: #khi hết video để đọc
            break 

        if idx % subsample_factor == 0:
            try:
                 resized_frame = cv2.resize(frame, (frame_size, frame_size)) #dùng cv2 để resize 224x224 #224x224x3-shape
                 frames.append(resized_frame) #thêm frame đã resize vào frame[]
                 frame_indices.append(int(idx))
            except Exception as e: #resize thất bại (Exception là các tất cả các lỗi)
                 print(f" resizing frame {idx} for video {video_id}: {e}") #e là chỉ ra lỗi để debug
                 pass 

        idx += 1
    cap.release() #Đóng đối tượng VideoCapture, giải phóng bộ nhớ cho đỡ nặng 
    
    if not frames: #ko có frame trong video
         print(f"No frames extracted for video {video_id} Skip")
         return 0, 0, False

    try:
        video_array = np.stack(frames) #chuyển từ dạng (H,W,C) -> 4D (N,H,W,C) N là số frame 
    except ValueError as e: #Các frame không có shape đồng nhất
         print(f" stacking frames for {video_id} (likely inconsistent shapes): {e}")
         return 0, 0, False

    npz_path = video_folder / f"{video_id}_frames.npz" #frames/1_frames.npz
    try:
        np.savez_compressed(npz_path, frames=video_array) #lưu video_array vào file npz_path -> đặt tên là frames
    except Exception as e: #lỗi
        print(f" saving frames NPZ for {video_id}: {e}")
        return 0, 0, False

    action_annotations = [] # list chứa annotations
    try:
        for action_idx, segments in enumerate(tlabs, start=1): #duyệt qua mỗi segments có trong tlabs, bắt đầu action_idx = 1
            if not isinstance(segments, (np.ndarray, list)) or len(segments) == 0: #isinstance-> để biết segments có thuộc dạng này hay ko ->T/F
                 continue #nếu segments ko fai dạng array hoặc list hoặc ko có segment(segment=0) -> bỏ qua
                 
            for segment_pair in segments: #duyệt qua start/end của từng hành động
                 if not isinstance(segment_pair, (np.ndarray, list)) or len(segment_pair) != 2: 
                      continue #Nếu segment_pair ko fai dạng array hoặc list hoặc số segment_pair ko có 2 phần tử -> ko có start/end -> bỏ qua 
                      
                 start, end = segment_pair #tách start/end
                 start_frame = int(start) #biến thành số
                 end_frame = int(end)

                 if not frame_indices: #Nếu hàm list frame_indices ko có 
                     print(f"frame_indices empty for {video_id} Cannot map segments")
                     break
                     
                 sub_start = helpers.find_nearest_subsampled_idx(start_frame, frame_indices) 
                 sub_end = helpers.find_nearest_subsampled_idx(end_frame, frame_indices)

                 if sub_end > sub_start: #đảm bảo đúng quy luật tự nhiên 
                    action_annotations.append(
                        {
                            "action_id": int(action_idx - 1), 
                            "action_name": f"action{action_idx}", 
                            "start_frame": int(sub_start), 
                            "end_frame": int(sub_end),
                            "start_time": float(start_frame / max(fps, 1e-6)),
                            "end_time": float(end_frame / max(fps, 1e-6)),
                            "original_start": int(start_frame),
                            "original_end": int(end_frame),
                        }
                    )
            if not frame_indices: break 

    except Exception as e: #lỗi
         print(f"processing segments for {video_id}: {e}")

    annotation_data = {
        "video_id": video_id,
        "num_frames": int(len(frames)),
        "original_frames": int(total_frames),
        "fps": float(fps),
        "subsample_factor": int(subsample_factor),
        "frame_indices": [int(idx) for idx in frame_indices], #lấy từng số ở trong list
        "annotations": action_annotations,
        "frames_file": f"{video_id}_frames.npz",
    }
    
    annotation_path = annotation_folder / f"{video_id}_annotations.json" #Path(data/full_videos/train/annotations/{video_id}_annotations.json)
    try:
        with open(annotation_path, "w") as f:
            json.dump(annotation_data, f, indent=2) #(indent-thụt lề vào 2 khoảng trắng mỗi cấp độ lồng nhau trong JSON),(f-file đã đc mở để ss ghi)
    except Exception as e: #lỗi
        print(f"saving annotation JSON for {video_id}: {e}")
        return len(frames), len(action_annotations), False

    return len(frames), len(action_annotations), True

def process_split(split_name, video_dir, label_dir, output_dir_split, frame_size, subsample_factor):
    print(f"\nProcessing {split_name} split")
    print(f"Video dir: {video_dir}")
    print(f"Label dir: {label_dir}")
    print(f"Output dir: {output_dir_split}")
    
    if not video_dir.exists() or not label_dir.exists():
        print(f"Input directories for {split_name} not found Skipping split")
        return 0, 0, 0, 0 

    output_dir_split.mkdir(parents=True, exist_ok=True) #tự tạo file
    
    total_videos = 0
    total_frames = 0
    total_actions = 0
    error_count = 0
    processed_videos = 0

    label_files = sorted(list(label_dir.glob("*.mat"))) #Lấy các file .mat, chuyển thành list, sắp xếp theo tên (glob của PATH tìm ở trong những file đã chỉ)
    
    if not label_files:
        print(f"No mat files found in {label_dir} Skip")
        return 0, 0, 0, 0
        
    for label_path in tqdm(label_files, desc=f"Processing {split_name}"):
        mat_file = label_path.name #lấy tên file (1_label.mat)
        base_name = label_path.stem #lấy tên 1_label
        if "_label" in base_name:
            base_name = base_name.replace("_label", "") # bỏ label

        video_file = f"{base_name}_crop.mp4" # 1_crop.mp4
        video_path = video_dir / video_file #đưa 1_crop.mp4 vào đúng file video_dir

        if not video_path.exists():
            print(f"Missing video file {video_file} for label {mat_file} Skip")
            error_count += 1 # +1 lỗi
            continue
        
        total_videos += 1 # +1 video 
        
        num_frames, num_actions, success = prepare_full_video(
            video_path, label_path, output_dir_split, frame_size, subsample_factor
        )
        
        if success:
             processed_videos += 1
             total_frames += num_frames
             total_actions += num_actions
        else:
             error_count += 1
             
    print(f"{split_name} split processing complete")
    print(f"Processed videos: {processed_videos}/{total_videos}")
    print(f"Errors encountered: {error_count}")
    
    dataset_stats = {
        "split": split_name,
        "processed_videos": int(processed_videos),
        "total_videos_in_split" : int(total_videos),
        "total_frames_processed": int(total_frames),
        "total_actions_found": int(total_actions),
        "frame_size": int(frame_size),
        "subsample_factor": int(subsample_factor),
        "file_format": "npz_compressed"
    }

    stats_path = output_dir_split / "dataset_stats.json"
    try:
        with open(stats_path, "w") as f:
            json.dump(dataset_stats, f, indent=2)
        print(f"Saved dataset stats for {split_name} to {stats_path}")
    except Exception as e:
        print(f"saving dataset stats for {split_name}: {e}")
        
    return processed_videos, total_frames, total_actions, error_count

def main():
    parser = argparse.ArgumentParser(description="Preprocess MERL Shopping dataset videos.") #Tạo ArgumentParser với mô tả về script
    parser.add_argument(#Khai báo hàm config
        '--config', # tên đường dẫn file cấu hình
        default='configs/config.yaml', # giá trị mặc định nếu người dùng không truyền
        help='Path to configuration file'# nội dung sẽ hiển thị khi --help
    )#python preprocess_raw_data.py --config /home/user/doan/config.yaml -> args.config == "/home/user/doan/config.yaml"
    parser.add_argument(#Khai báo hàm split
        "--split", 
        type=str, # kiểu dữ liệu
        choices=["train", "val", "test", "all"],# chỉ chấp nhận vài giá trị nhất định
        default="all", 
        help="Dataset split(s) to process."
    )#python preprocess_raw_data.py --split train -> train
    args = parser.parse_args() #Gọi parse_args() để lấy kết quả phân tích #Check khi truyền input đầu vào xem có phù hợp ko

    try: #Mở file config
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)#dạng dictionary #load những setup đơn giản
    except FileNotFoundError:#Lỗi đường dẫn
        print(f"Config file not found at {args.config}")
        return
    except Exception as e:#Lỗi khác
        print(f"loading config file: {e}")
        return

    data_cfg = cfg['data'] #Lấy thông tin
    prep_cfg = cfg['preprocessing']
    frame_size = prep_cfg['frame_size']
    subsample_factor = prep_cfg['subsample_factor']
    
    base_raw_video_dir = Path(data_cfg['raw_video_dir'])#đổi đường dẫn để chạy đc trên Win hay Linux
    base_raw_label_dir = Path(data_cfg['raw_label_dir'])
    base_processed_dir = Path(data_cfg['processed_dir']) #Path(data/full_videos)
    
    splits_to_process = [] #tạo list
    if args.split == "train" or args.split == "all":
        splits_to_process.append("train") #thêm train vào danh sách 
    if args.split == "val" or args.split == "all":
        splits_to_process.append("val")
    if args.split == "test" or args.split == "all":
        splits_to_process.append("test")
        
    if not splits_to_process:
        print("No valid split selected")
        return
        
    grand_total_videos = 0
    grand_total_frames = 0
    grand_total_actions = 0
    grand_total_errors = 0

    for split in splits_to_process:
        video_dir = base_raw_video_dir / split #video_dir = Path("Data/Videos…")/"train" (/ chỉ dùng khi Path)
        label_dir = base_raw_label_dir / split
        output_dir_split = base_processed_dir / split #Path(data/full_videos/train)
        
        processed, frames, actions, errors = process_split(
            split, video_dir, label_dir, output_dir_split, frame_size, subsample_factor
        )
        grand_total_videos += processed #Cộng dồn số video thành công vào tổng 
        grand_total_frames += frames
        grand_total_actions += actions
        grand_total_errors += errors

    print("\nOverall Preprocessing Summary")
    print(f"Total videos processed across all selected splits: {grand_total_videos}")
    print(f"Total frames generated: {grand_total_frames}")
    print(f"Total action segments found: {grand_total_actions}")
    print(f"Total errors encountered: {grand_total_errors}")

if __name__ == "__main__":
    main()
