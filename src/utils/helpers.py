import numpy as np
import torch
import random
import os
import json
import pickle
from collections import defaultdict
from tqdm import tqdm

def find_nearest_subsampled_idx(original_idx, frame_indices):
    """Find the nearest subsampled frame index to the original frame index"""
    differences = np.abs(np.array(frame_indices) - original_idx) #ép biến list frame_indices thành mảng array -> - original_idx(idx gốc)
    #original_idx lấy từ start_frame và end_frame
    nearest_idx = np.argmin(differences) #lấy số nhỏ nhất trong array (gần nhất với idx gốc)
    return int(nearest_idx)  

def gaussian_kernel(center, window_size, sigma=1.0): #sử dụng gaussian_kernel để làm mờ nhãn bên ngoài hành động -> cải thiện khi train model
    x = torch.arange(window_size).float()
    return torch.exp(-((x - center) ** 2) / (2 * sigma**2))  #(exp là e mũ x)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Set seed={seed}")

def calculate_temporal_iou(pred_segment, gt_segment):
    """Calculate temporal IoU between prediction and ground truth segments"""
    pred_start, pred_end = pred_segment
    gt_start, gt_end = gt_segment
    
    # Đảm bảo end > start cho cả predicted và ground truth segments
    pred_start, pred_end = min(pred_start, pred_end), max(pred_start, pred_end)
    gt_start, gt_end = min(gt_start, gt_end), max(gt_start, gt_end)
    
    # Kiểm tra trường hợp segments bị thoái hóa
    if pred_start == pred_end or gt_start == gt_end:
        return 0.0
    
    # Calculate intersection
    intersection = max(0, min(pred_end, gt_end) - max(pred_start, gt_start))
    
    # Calculate union
    union = max(1, (pred_end - pred_start) + (gt_end - gt_start) - intersection)
    
    return intersection / union

def process_for_evaluation(detections, gt_annotations, action_masks, window_size, num_classes):
    """Process predictions and ground truth for evaluation metrics"""
    processed_preds = []
    processed_targets = []
    
    # Process each frame
    for t in range(window_size):
        for c in range(num_classes):
            # Check if frame t belongs to class c in ground truth
            is_gt_action = False
            for anno in gt_annotations:
                if anno['action_id'] == c and anno['start_frame'] <= t < anno['end_frame']:
                    is_gt_action = True
                    break
            
            # Check if frame t belongs to class c in predictions
            is_pred_action = False
            for det in detections:
                if det['action_id'] == c and det['start_frame'] <= t < det['end_frame']:
                    is_pred_action = True
                    break
            
            processed_preds.append(1 if is_pred_action else 0)
            processed_targets.append(1 if is_gt_action else 0)
    
    return processed_preds, processed_targets

def reconstruct_full_video_probs(video_id, all_raw_preds_loaded, all_batch_meta_loaded, num_classes, window_size):
    """ 
    Reconstructs full action, start, and end probability sequences for a video.
    Similar to functions in eval_hmm.py but combines all three and reads from loaded data.
    
    Args:
        video_id (str): ID of the video.
        all_raw_preds_loaded (list): Loaded list of tuples [(action_t, start_t, end_t), ...]
        all_batch_meta_loaded (list): Loaded list of lists of metadata dictionaries.

    Returns:
        tuple: (avg_action_probs, avg_start_probs, avg_end_probs, num_frames)
               Returns (None, None, None, 0) if video not found or error.
    """
    video_windows_indices = []
    max_end_frame = 0

    # 1. Find windows and video length
    for batch_idx, batch_meta in enumerate(all_batch_meta_loaded):
        for window_idx_in_batch, meta in enumerate(batch_meta):
            if meta['video_id'] == video_id:
                # Try to get actual window size from the loaded tensor shape if possible
                try:
                    # Check shape of action_probs tensor for this window
                    # Assuming batch_idx and window_idx_in_batch are valid
                    current_window_size = all_raw_preds_loaded[batch_idx][0][window_idx_in_batch].shape[0]
                except (IndexError, AttributeError):
                    # Fallback to default if shape access fails
                    # print(f"Warning: Could not determine window size for {video_id} batch {batch_idx}. Using default {WINDOW_SIZE}.")
                    current_window_size = window_size 
                
                video_windows_indices.append({
                    'batch_idx': batch_idx,
                    'window_idx_in_batch': window_idx_in_batch,
                    'start_frame': meta['start_idx'],
                    'window_len': current_window_size
                })
                # Use end_idx from metadata which should be accurate
                max_end_frame = max(max_end_frame, meta['end_idx'])

    if not video_windows_indices:
        # print(f"Warning: No windows found for video {video_id} during reconstruction.")
        return None, None, None, 0

    num_frames = max_end_frame
    if num_frames <= 0:
        return None, None, None, 0

    # 2. Initialize sum and count arrays (use float64 for sums)
    sum_action_probs = np.zeros((num_frames, num_classes), dtype=np.float64)
    sum_start_probs = np.zeros((num_frames, num_classes), dtype=np.float64)
    sum_end_probs = np.zeros((num_frames, num_classes), dtype=np.float64)
    counts = np.zeros((num_frames, num_classes), dtype=np.int16)

    # 3. Accumulate probabilities from windows
    for window_info in video_windows_indices:
        batch_idx = window_info['batch_idx']
        window_idx = window_info['window_idx_in_batch']
        start_f = window_info['start_frame']
        window_len = window_info['window_len']

        try:
            action_probs_win = all_raw_preds_loaded[batch_idx][0][window_idx].numpy().astype(np.float64)
            start_probs_win = all_raw_preds_loaded[batch_idx][1][window_idx].numpy().astype(np.float64)
            end_probs_win = all_raw_preds_loaded[batch_idx][2][window_idx].numpy().astype(np.float64)
            
            # Basic shape check
            if action_probs_win.shape[0] != window_len or \
               start_probs_win.shape[0] != window_len or \
               end_probs_win.shape[0] != window_len or \
               action_probs_win.shape[1] != num_classes or \
               start_probs_win.shape[1] != num_classes or \
               end_probs_win.shape[1] != num_classes:
                 print(f"Warning: Shape mismatch for window {video_id} batch {batch_idx} window {window_idx}. Expected T={window_len}, C={num_classes}. Skipping window.")
                 continue
                 
        except (IndexError, AttributeError) as e:
            print(f"Warning: Error accessing raw preds for window {video_id} batch {batch_idx} window {window_idx}: {e}. Skipping window.")
            continue

        global_start = start_f
        global_end = min(start_f + window_len, num_frames)
        local_end = global_end - global_start

        if local_end > 0:
            sum_action_probs[global_start:global_end, :] += action_probs_win[:local_end, :]
            sum_start_probs[global_start:global_end, :] += start_probs_win[:local_end, :]
            sum_end_probs[global_start:global_end, :] += end_probs_win[:local_end, :]
            counts[global_start:global_end, :] += 1

    # 4. Calculate average probabilities
    # Avoid division by zero for frames not covered by any window
    counts[counts == 0] = 1 
    avg_action_probs = sum_action_probs / counts
    avg_start_probs = sum_start_probs / counts
    avg_end_probs = sum_end_probs / counts
    
    # Clip to valid probability range [0, 1]
    avg_action_probs = np.clip(avg_action_probs, 0.0, 1.0)
    avg_start_probs = np.clip(avg_start_probs, 0.0, 1.0)
    avg_end_probs = np.clip(avg_end_probs, 0.0, 1.0)

    return avg_action_probs, avg_start_probs, avg_end_probs, num_frames

def generate_target_labels(video_id, anno_dir, num_frames, num_classes):
    """
    Generates the ground truth frame-level label sequence for a video.
    Similar to logic in learn_hmm_transitions.py.
    
    Args:
        video_id (str): ID of the video.
        anno_dir (str): Path to the directory containing annotation JSON files.
        num_frames (int): The total number of frames for this video (obtained from reconstruction).

    Returns:
        np.array: Sequence of frame labels (shape T,), or None if annotation not found/error.
    """
    if num_frames <= 0:
        return None
        
    anno_filename = f"{video_id}_annotations.json" # Assuming this naming convention
    anno_filepath = os.path.join(anno_dir, anno_filename)

    if not os.path.exists(anno_filepath):
        print(f"Warning: Annotation file not found for {video_id} at {anno_filepath}")
        return None

    try:
        with open(anno_filepath, 'r') as f:
            anno_data = json.load(f)
        
        # Use num_frames determined from reconstruction, ignore anno_data['num_frames'] potentially
        annotations = anno_data['annotations']
        
        # Initialize all frames to background
        frame_labels = np.full(num_frames, fill_value=num_classes, dtype=int)
        
        # Sort annotations by start time to handle overlaps correctly (last one wins)
        annotations.sort(key=lambda x: x['start_frame'])
        
        for anno in annotations:
            action_id = anno['action_id']
            start = anno['start_frame']
            end = anno['end_frame'] # Exclusive
            
            # Clip annotation boundaries to the actual video length
            start = max(0, start)
            end = min(num_frames, end)
            
            if start < end and 0 <= action_id < num_classes:
                frame_labels[start:end] = action_id
            elif action_id >= num_classes:
                 # print(f"Warning: Invalid action_id {action_id} in {anno_filename}. Skipping.")
                 pass # Just ignore invalid class IDs
                 
        return frame_labels
        
    except Exception as e:
        print(f"Error processing annotation file {anno_filepath}: {e}")
        return None
def process_predictions_for_rnn(predictions, num_classes, window_size, output_pkl, anno_dir, rnn_data_dir):
    """
    Process predictions and generate RNN training data.
    
    Args:
        predictions: Raw model predictions
        num_classes: Number of action classes
        window_size: Size of temporal window
        train_output_pkl: Path to training inference results pickle file
        train_anno_dir: Path to training annotations directory
        rnn_train_data_dir: Path to save processed RNN training data
    """
    print(f"\nProcessing Training Data from {output_pkl}...")
    try:
        with open(output_pkl, 'rb') as f:
            inference_results = pickle.load(f)
        raw_preds_loaded = inference_results['all_raw_preds']
        batch_meta_loaded = inference_results['all_batch_meta']
        print(f"Loaded raw training data for {len(raw_preds_loaded)} batches.")
        
        video_ids = sorted(list(set(meta['video_id'] for batch_meta in batch_meta_loaded for meta in batch_meta)))
        print(f"Found {len(video_ids)} unique training video IDs.")

        processed_train_count = 0
        skipped_train_count = 0
        for video_id in tqdm(video_ids, desc="Processing Train Videos"):
            # 1. Reconstruct probabilities
            avg_action_probs, avg_start_probs, avg_end_probs, num_frames = reconstruct_full_video_probs(
                video_id, raw_preds_loaded, batch_meta_loaded, num_classes, window_size
            )
            
            if num_frames is None or num_frames <= 0:
                skipped_train_count += 1
                continue
                
            # 2. Generate target labels
            target_labels = generate_target_labels(video_id, anno_dir, num_frames, num_classes)
            
            if target_labels is None:
                skipped_train_count += 1
                continue
                
            # 3. Combine features (Input X for RNN)
            # Shape: (T, num_classes * 3)
            input_features = np.concatenate([avg_action_probs, avg_start_probs, avg_end_probs], axis=1)
            
            # 4. Save as .npz
            output_path = os.path.join(rnn_data_dir, f"{video_id}.npz")
            try:
                np.savez_compressed(output_path, features=input_features, labels=target_labels)
                processed_train_count += 1
            except Exception as e:
                print(f"Error saving {output_path}: {e}")
                skipped_train_count += 1
                
        print(f"Finished processing training data. Processed: {processed_train_count}, Skipped: {skipped_train_count}")

    except FileNotFoundError:
        print(f"Error: {output_pkl} not found. Please run the inference step first.")
    except Exception as e:
        print(f"An error occurred during training data processing: {e}")


def calculate_global_gt(all_batch_meta, num_classes):
    global_action_gt_by_video = defaultdict(lambda: defaultdict(list))
    all_metadata_flat = [meta for batch_meta in all_batch_meta for meta in batch_meta]
    unique_video_ids_from_meta = set()
    for meta in all_metadata_flat:
         video_id = meta['video_id']
         unique_video_ids_from_meta.add(video_id)
         start_idx = meta['start_idx']
         # Ensure annotations exist and are iterable
         if isinstance(meta.get('annotations'), list):
             for anno in meta['annotations']:
                 # Basic validation of annotation structure
                 if isinstance(anno, dict) and all(k in anno for k in ('action_id', 'start_frame', 'end_frame')):
                     action_id = anno['action_id']
                     global_gt_start = start_idx + anno['start_frame']
                     global_gt_end = start_idx + anno['end_frame']
                     if 0 <= action_id < num_classes: # Ensure valid action_id
                          global_action_gt_by_video[video_id][action_id].append((global_gt_start, global_gt_end))
                 else:
                      print(f"Warning: Invalid annotation format found in metadata for video {video_id}: {anno}")
         elif 'annotations' in meta: # If key exists but is not a list
              print(f"Warning: Invalid 'annotations' type in metadata for video {video_id}: {type(meta['annotations'])}")

    final_global_gt = defaultdict(list)
    for video_id, actions in global_action_gt_by_video.items():
         for action_id, segments in actions.items():
             unique_segments_video_action = sorted(list(set(segments)))
             final_global_gt[action_id].extend(unique_segments_video_action)
    final_global_gt = {c: sorted(list(set(final_global_gt.get(c, [])))) for c in range(num_classes)}
    total_global_gt_segments = sum(len(v) for v in final_global_gt.values())
    return final_global_gt, total_global_gt_segments, global_action_gt_by_video
