import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
import src.utils.helpers as helpers

class FullVideoDataset(Dataset):
    def __init__(self, frames_dir, anno_dir, num_classes, window_size, mode='train'):
        self.frames_dir = Path(frames_dir)
        self.anno_dir = Path(anno_dir)
        self.num_classes = num_classes
        self.window_size = window_size
        self.mode = mode
        self.samples = []

        video_ids = []
        frame_files = [f.name for f in self.frames_dir.iterdir()] if self.frames_dir.exists() else []
        
        for fname in frame_files:
            if fname.endswith("_frames.npz"):
                video_id = fname.replace("_frames.npz", "")
                anno_path = self.anno_dir / f"{video_id}_annotations.json"
                
                if anno_path.exists():
                    video_ids.append(video_id)
                else:
                    print(f"Skipping {video_id} - missing annotations")
        
        for video_id in video_ids:
            self._process_video(video_id)
                
        print(f"[{mode}] Loaded {len(self.samples)} sliding windows from {len(video_ids)} videos")
        
    def _process_video(self, video_id):
        anno_path = self.anno_dir / f"{video_id}_annotations.json"
        with open(anno_path, 'r') as f:
            anno = json.load(f)
        
        num_frames = anno["num_frames"]
        annotations = anno["annotations"]
        
        if num_frames < self.window_size:
            self._add_window(video_id, 0, num_frames, annotations)
            return
            
        stride = 8
        
        for start in range(0, num_frames - self.window_size + 1, stride):
            end = start + self.window_size
            self._add_window(video_id, start, end, annotations)
            
        if (num_frames - self.window_size) % stride != 0:
            start = num_frames - self.window_size
            end = num_frames
            self._add_window(video_id, start, end, annotations)
            
    def _add_window(self, video_id, start_idx, end_idx, all_annotations):
        window_annos = []
        
        for anno in all_annotations:
            action_start = anno["start_frame"]
            action_end = anno["end_frame"]
            
            if action_end < start_idx or action_start >= end_idx:
                continue  # No overlap
                
            rel_start = max(0, action_start - start_idx)
            rel_end = min(end_idx - start_idx, action_end - start_idx)
            
            if rel_end > rel_start:
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
        video_id = sample["video_id"]
        start_idx = sample["start_idx"]
        end_idx = sample["end_idx"]
        annotations = sample["annotations"]
        
        frames_path = self.frames_dir / f"{video_id}_frames.npz"
        npz_data = np.load(frames_path)
        all_frames = npz_data['frames']
        
        if end_idx <= all_frames.shape[0]:
            frames = all_frames[start_idx:end_idx]
        else:
            frames = np.zeros((self.window_size, *all_frames.shape[1:]), dtype=all_frames.dtype)
            actual_frames = all_frames[start_idx:end_idx]
            frames[:actual_frames.shape[0]] = actual_frames
            if actual_frames.shape[0] > 0:
                frames[actual_frames.shape[0]:] = actual_frames[-1]
        
        frames = torch.from_numpy(frames).float() / 255.0 
        frames = frames.permute(3, 0, 1, 2)
        
        action_masks = torch.zeros((self.num_classes, self.window_size), dtype=torch.float32)
        start_mask = torch.zeros((self.num_classes, self.window_size), dtype=torch.float32)
        end_mask = torch.zeros((self.num_classes, self.window_size), dtype=torch.float32)

        for anno in annotations:
            action_id = anno["action_id"]
            s, e = anno["start_frame"], anno["end_frame"]
            
            # Ensure action_id is within bounds of our tensor dimensions
            if action_id < self.num_classes:
                action_masks[action_id, s:e] = 1.0
                
                start_mask[action_id] += helpers.gaussian_kernel(s, self.window_size, sigma=2.0)
                end_mask[action_id] += helpers.gaussian_kernel(e-1, self.window_size, sigma=2.0)
            else:
                print(f"Warning: action_id {action_id} exceeds num_classes {self.num_classes}")
            
        start_mask = torch.clamp(start_mask, 0, 1)
        end_mask = torch.clamp(end_mask, 0, 1)
        
        metadata = {
            "video_id": video_id,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "annotations": annotations,
        }
        
        return frames, action_masks, start_mask, end_mask, metadata

def custom_collate_fn(batch):
    frames, action_masks, start_masks, end_masks, metadata = zip(*batch)
    
    frames = torch.stack(frames)
    action_masks = torch.stack(action_masks)
    start_masks = torch.stack(start_masks)
    end_masks = torch.stack(end_masks)
    
    return frames, action_masks, start_masks, end_masks, metadata

def get_train_loader(cfg, shuffle=True):
    data_cfg = cfg.get('data', {})
    train_cfg = cfg.get('base_model_training', {})
    global_cfg = cfg.get('global', {})

    frames_dir = Path(data_cfg.get('base_dir', 'Data')) / 'full_videos/train/frames'
    anno_dir = Path(data_cfg.get('base_dir', 'Data')) / 'full_videos/train/annotations'
    num_classes = global_cfg.get('num_classes', 6)  # Now default to 6 classes (background + 5 actions)
    window_size = global_cfg.get('window_size', 32)
    batch_size = train_cfg.get('dataloader', {}).get('batch_size', 1)
    num_workers = train_cfg.get('dataloader', {}).get('num_workers', 4)

    dataset = FullVideoDataset(
        frames_dir,
        anno_dir,
        num_classes=num_classes,
        window_size=window_size,
        mode='train'
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=custom_collate_fn
    )

def get_val_loader(cfg, shuffle=False):
    data_cfg = cfg.get('data', {})
    train_cfg = cfg.get('base_model_training', {})
    global_cfg = cfg.get('global', {})

    frames_dir = Path(data_cfg.get('base_dir', 'Data')) / 'full_videos/val/frames'
    anno_dir = Path(data_cfg.get('base_dir', 'Data')) / 'full_videos/val/annotations'
    num_classes = global_cfg.get('num_classes', 6)  # Now default to 6 classes (background + 5 actions)
    window_size = global_cfg.get('window_size', 32)
    val_batch_size = train_cfg.get('dataloader', {}).get('batch_size', 1)
    num_workers = train_cfg.get('dataloader', {}).get('num_workers', 4)

    dataset = FullVideoDataset(
        frames_dir,
        anno_dir,
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
    num_classes = global_cfg.get('num_classes', 6)  # Now default to 6 classes (background + 5 actions)
    window_size = global_cfg.get('window_size', 32)
    test_batch_size = train_cfg.get('dataloader', {}).get('batch_size', 1)
    num_workers = train_cfg.get('dataloader', {}).get('num_workers', 4)

    dataset = FullVideoDataset(
        frames_dir,
        anno_dir,
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
