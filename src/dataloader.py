import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
import src.utils.feature_extraction as feature_extraction
import src.utils.helpers as helpers

class FullVideoDataset(Dataset):
    def __init__(self, frames_dir, anno_dir, pose_dir, num_classes, window_size, mode='train'):
        self.frames_dir = Path(frames_dir) #Path(data/full_videos/train/frames)
        self.anno_dir = Path(anno_dir) #Path(data/full_videos/train/annotations)
        self.pose_dir = Path(pose_dir) #Path(data/full_videos/train/pose)
        self.num_classes = num_classes #5
        self.window_size = window_size #32
        self.mode = mode #train
        self.samples = [] #những thông tin mà mình dùng sliding window

        video_ids = [] 
        frame_files = [f.name for f in self.frames_dir.iterdir()] if self.frames_dir.exists() else [] #(.name-lấy tên file),(iterdir()-liệt kê ra các tập con có trong đường đẫn)
        
        for fname in frame_files:
            if fname.endswith("_frames.npz"): #1_1_frames.npz
                video_id = fname.replace("_frames.npz", "") #1_1
                anno_path = self.anno_dir / f"{video_id}_annotations.json" #Path(data/full_videos/train/annotations/1_1_annotations.json)
                pose_path = self.pose_dir / f"{video_id}_pose.npz" #Path(data/full_videos/train/annotations/1_1_pose.json)
                
                if anno_path.exists() and pose_path.exists():
                    video_ids.append(video_id)
                else:
                    missing = [] #cho biết là file nào không có 
                    if not anno_path.exists(): missing.append("annotations")
                    if not pose_path.exists(): missing.append("pose")
                    print(f"Skipping {video_id} - missing {', '.join(missing)}")
        
        for video_id in video_ids:
            self._process_video(video_id)
                
        print(f"[{mode}] Loaded {len(self.samples)} sliding windows from {len(video_ids)} videos")
        
    def _process_video(self, video_id):#xử lí nh trường hợp khi sliding window
        anno_path = self.anno_dir / f"{video_id}_annotations.json" #Path(data/full_videos/train/annotations/1_1_annotations.json)
        with open(anno_path, 'r') as f:
            anno = json.load(f)
        
        num_frames = anno["num_frames"] #số lượng frane 
        annotations = anno["annotations"] #thông tin các đoạn video được cắt theo label
        
        if num_frames < self.window_size: #số lượng frane < 32
            self._add_window(video_id, 0, num_frames, annotations) #tạo window tuwf frame 0->frane cuối
            return
            
        stride = self.window_size // 2 #overlap 50%
        #stride = 16
        for start in range(0, num_frames - self.window_size + 1, stride):
            end = start + self.window_size #0+32 #16+32 #32+32
            self._add_window(video_id, start, end, annotations)
            
        if (num_frames - self.window_size) % stride != 0: #Nếu frame dư ra ko chia hết 16
            start = num_frames - self.window_size
            end = num_frames
            self._add_window(video_id, start, end, annotations) #Thêm window
            
    def _add_window(self, video_id, start_idx, end_idx, all_annotations):#sliding window
        window_annos = []
        
        for anno in all_annotations:
            action_start = anno["start_frame"] #frame bắt đầu của đoạn video
            action_end = anno["end_frame"] #frame kết thúc của đoạn video
            
            if action_end < start_idx or action_start >= end_idx: #action nằm ngoài window
                continue  # No overlap
                
            rel_start = max(0, action_start - start_idx) #vị trí start của đoạn video đc cắt từ label so vs start đc tạo từ sliding
            rel_end = min(end_idx - start_idx, action_end - start_idx) #độ dài window PHẢI luôn 32->min
            #                    32              lớn hơn 32 thì sao
            if rel_end > rel_start: # đúng quy luật tự nhiên
                window_annos.append({
                    "action_id": anno["action_id"],
                    "start_frame": rel_start,
                    "end_frame": rel_end,
                    "original_start": anno["original_start"],
                    "original_end": anno["original_end"]
                })

        self.samples.append({
            "video_id": video_id,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "annotations": window_annos
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_id = sample["video_id"] #id của cả video 
        start_idx = sample["start_idx"] #frame start của window
        end_idx = sample["end_idx"] #frame end của window
        annotations = sample["annotations"]
        
        frames_path = self.frames_dir / f"{video_id}_frames.npz" #Path(data/full_videos/train/frames/1_1_frames.npz)
        npz_data = np.load(frames_path)
        all_frames = npz_data['frames'] #Lấy mảng frames (shape: [T, H, W, C]
        
        pose_path = self.pose_dir / f"{video_id}_pose.npz" #Path(data/full_videos/train/frames/1_1_pose.npz)
        pose_npz = np.load(pose_path)
        all_pose = pose_npz['pose'] #shape: [T, D]
        
        #if end_idx <= all_frames.shape[0]: #Nếu frame end của window < T
        frames = all_frames[start_idx:end_idx] #Lấy tất cả frame của window
        """else:
            frames = np.zeros((self.window_size, *all_frames.shape[1:]), dtype=all_frames.dtype) #mảng 0(32,H,W,C),(dtype-đảm bảo có cùng kiểu dữ liệu vs mảng cũ)
            actual_frames = all_frames[start_idx:end_idx] #Lấy số frame thực tế 
            frames[:actual_frames.shape[0]] = actual_frames # sẽ lấy frame thực tế từ window thừa chuyển vào window rỗng có size 32
            if actual_frames.shape[0] > 0: #
                frames[actual_frames.shape[0]:] = actual_frames[-1] #ko hiểu """
        
        #if end_idx <= all_pose.shape[0]:
        pose_data = all_pose[start_idx:end_idx]
        """else:
            pose_data = np.zeros((self.window_size, all_pose.shape[1]), dtype=all_pose.dtype)
            actual_pose = all_pose[start_idx:end_idx]
            pose_data[:actual_pose.shape[0]] = actual_pose
            if actual_pose.shape[0] > 0:
                pose_data[actual_pose.shape[0]:] = actual_pose[-1]"""
        
        velocity_data = feature_extraction.compute_velocity(pose_data) #Vận tốc
        
        pose_with_velocity = np.concatenate([pose_data, velocity_data], axis=1) #T,2xD (concatenate để nối các mảng numpy) (axis nhân theo hàng hay theo cột)
                
        frames = torch.from_numpy(frames).float() / 255.0 #(225-số các cường độ của màu) (chuyển từ array->tensor)
        frames = frames.permute(3, 0, 1, 2) #hoán đổi vị trí (C,T,H,N) phù hợp với CNN (các số chính là thứ tự của vị trí cũ)
    
        pose_with_velocity = torch.from_numpy(pose_with_velocity).float()
        

        action_masks = torch.zeros((self.num_classes, self.window_size), dtype=torch.float32)
        

        start_mask = torch.zeros((self.num_classes, self.window_size), dtype=torch.float32)
        end_mask = torch.zeros((self.num_classes, self.window_size), dtype=torch.float32)


        for anno in annotations:
            action_id = anno["action_id"] 
            s, e = anno["start_frame"], anno["end_frame"]  #start, end của window
            
            action_masks[action_id, s:e] = 1.0 #Gán 1 vào mảng zero để biết có action diễn ra
            
            start_mask[action_id] += helpers.gaussian_kernel(s, self.window_size, sigma=2.0)
            end_mask[action_id] += helpers.gaussian_kernel(e-1, self.window_size, sigma=2.0)
            
        start_mask = torch.clamp(start_mask, 0, 1) #clamp()-giới hạn các giá trị trong khoảng 0->1, nếu <0->0 nếu >1->1
        end_mask = torch.clamp(end_mask, 0, 1) 
        
        metadata = {
            "video_id": video_id,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "annotations": annotations,
        }

        
        return frames, pose_with_velocity,action_masks, start_mask, end_mask, metadata

def custom_collate_fn(batch):

    frames, pose_data, action_masks, start_masks, end_masks, metadata = zip(*batch)
    
    frames = torch.stack(frames)
    pose_data = torch.stack(pose_data)
    action_masks = torch.stack(action_masks)
    start_masks = torch.stack(start_masks)
    end_masks = torch.stack(end_masks)
    
    return frames, pose_data, action_masks, start_masks, end_masks, metadata

def get_train_loader(cfg, shuffle=True): #shuffle sẽ xáo trộn dữ liệu ngẫu nhiên ko theo nguyên tắc -> mô hình sẽ học tổng quát hơn và tránh overfitting
    data_cfg = cfg.get('data', {}) #lấy cấu hình từ data hoặc dict rỗng #cfg[data]
    train_cfg = cfg.get('base_model_training', {}) #cfg[base_model_training]
    global_cfg = cfg.get('global', {})

    frames_dir = Path(data_cfg.get('base_dir', 'Data')) / 'full_videos/train/frames' #Path(data/full_videos/train/frames)
    anno_dir = Path(data_cfg.get('base_dir', 'Data')) / 'full_videos/train/annotations' #Path(data/full_videos/train/annotations)
    pose_dir = Path(data_cfg.get('base_dir', 'Data')) / 'full_videos/train/pose' #Path(data/full_videos/train/pose)
    num_classes = global_cfg.get('num_classes', 5) 
    window_size = global_cfg.get('window_size', 32)
    batch_size = train_cfg.get('dataloader', {}).get('batch_size', 1) #cfg[base_model_training][dataloader][batch_size]=1
    num_workers = train_cfg.get('dataloader', {}).get('num_workers', 4)

    dataset = FullVideoDataset(
        frames_dir,
        anno_dir,
        pose_dir,
        num_classes=num_classes,
        window_size=window_size,
        mode='train'
    )
    return DataLoader(
        dataset,
        batch_size=batch_size, #1
        shuffle=shuffle, #True
        num_workers=num_workers, #4
        collate_fn=custom_collate_fn
    )

def get_val_loader(cfg, shuffle=False):
    data_cfg = cfg.get('data', {})
    train_cfg = cfg.get('base_model_training', {})
    global_cfg = cfg.get('global', {})

    frames_dir = Path(data_cfg.get('base_dir', 'Data')) / 'full_videos/val/frames'
    anno_dir = Path(data_cfg.get('base_dir', 'Data')) / 'full_videos/val/annotations'
    pose_dir = Path(data_cfg.get('base_dir', 'Data')) / 'full_videos/val/pose'
    num_classes = global_cfg.get('num_classes', 5)
    window_size = global_cfg.get('window_size', 32)
    val_batch_size = train_cfg.get('dataloader', {}).get('batch_size', 1)
    num_workers = train_cfg.get('dataloader', {}).get('num_workers', 4)

    dataset = FullVideoDataset(
        frames_dir,
        anno_dir,
        pose_dir,
        num_classes=num_classes,
        window_size=window_size,
        mode='val'
    )
    return DataLoader(
        dataset,
        batch_size=val_batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=custom_collate_fn
    )

def get_test_loader(cfg, shuffle=False):
    data_cfg = cfg.get('data', {})
    train_cfg = cfg.get('base_model_training', {})
    global_cfg = cfg.get('global', {})

    frames_dir = Path(data_cfg.get('base_dir', 'Data')) / 'full_videos/test/frames'
    anno_dir = Path(data_cfg.get('base_dir', 'Data')) / 'full_videos/test/annotations'
    pose_dir = Path(data_cfg.get('base_dir', 'Data')) / 'full_videos/test/pose'
    num_classes = global_cfg.get('num_classes', 5)
    window_size = global_cfg.get('window_size', 32)
    test_batch_size = train_cfg.get('dataloader', {}).get('batch_size', 1)
    num_workers = train_cfg.get('dataloader', {}).get('num_workers', 4)

    dataset = FullVideoDataset(
        frames_dir,
        anno_dir,
        pose_dir,
        num_classes=num_classes,
        window_size=window_size,
        mode='test'
    )
    return DataLoader(
        dataset,
        batch_size=test_batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=custom_collate_fn
    )
