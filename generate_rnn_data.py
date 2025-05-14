import torch
import argparse
import pickle
from tqdm import tqdm
import copy
import yaml
from pathlib import Path

import src.utils.helpers as helpers

try:
    import src.models.base_detector as base_detector
except ImportError:
    print("Could not import TemporalActionDetector from src models model_fixed Make sure the file exists")
    exit()
try:
    import src.dataloader as dataloader
except ImportError:
    print("Could not import get_train_loader/get_val_loader/get_test_loader from src dataloader")
    exit()
try:
    import src.losses as losses
except ImportError:
    print("ActionDetectionLoss not found")
    exit()

def run_inference(model, data_loader, device): #lấy data từ model 
    model.eval() #Chuyển model sang chế độ đánh giá (không cập nhật gradient)(vì để đánh giá nên sẽ ko dropout để có thể học hết, BatchNorm sẽ không phụ thuộc vào batch hiện tại đang chạy mà sẽ tính trung bình và phương sai trong cả quá trình training)
    all_raw_preds = []
    all_batch_meta = []

    print(f"Running inference on {len(data_loader.dataset)} samples") #số lượng window sẽ chạy (có tính len(sample))
    with torch.no_grad(): #tắt tính gradient-> tiết kiệm bộ nhớ 
        for batch in tqdm(data_loader, desc="Model Inference"): 
            try:
                frames, pose_data, _, _, _, metadata = batch #frames, pose_data, action_masks, start_masks, end_masks, metadata (_ là những thông tin không dùng đến)
            except ValueError: #Nếu thất bại
                 print("Batch structure mismatch Trying simplified unpack (frames, pose, meta) Check dataloader")
                 try: frames, pose_data, metadata = batch #Thử cái này nè
                 except ValueError: print("Cannot determine batch structure Exiting"); exit() #Lỗi thêm thì thôi ->cook

            frames = frames.to(device) #đưa dữ liệu vào đúng GPU/CPU
            if pose_data is not None: pose_data = pose_data.to(device) #trường hợp nếu frame ko dùng đến pose 

            with torch.cuda.amp.autocast(enabled=True): #kích hoạt Automatic Mixed Precision (AMP).Bình thg dùng float32 để tính->dùng float16 để tính toán đơn giản(nhân ma trận, conv) kết hợp float32 để tính đơn giản->nhanh và giảm vram
                predictions = model(frames, pose_data)

            action_probs = torch.sigmoid(predictions['action_scores']).cpu().detach() #predictions['action_scores']-các logit mà model dự đoán cho mỗi class tại mỗi frame, shape ((B, T, num_classes))
            start_probs = torch.sigmoid(predictions['start_scores']).cpu().detach() #.cpu để giảm tải VRAM sau khi tính, chuyển sang dạng tensor phù hợp cho Numpy, Pytorch
            end_probs = torch.sigmoid(predictions['end_scores']).cpu().detach()# detach để tách tensor thành tensor độc lập không giữ lịch sử tính toán trước -> Tiết kiệm bộ nhớ 
            #Dùng detach bởi vì là những hàm này sẽ chỉ sử dụng cho mục đích đánh giá hoặc đầu vào RNN nên sẽ tách ra để tiết kiệm bộ nhớ
            #detach dùng chỉ để đảm bảo và là best practice
            all_raw_preds.append((action_probs, start_probs, end_probs))
            all_batch_meta.append(copy.deepcopy(metadata))#Lưu giống như append(metadata) nhưng sẽ không cập nhật khi sửa đổi metadata
            #không bị ghi đè và tạo một bản mới khi chạy các batch 
    return all_raw_preds, all_batch_meta

def main(cfg, args):
    if cfg['global']['device'] == 'auto':#Nếu device: auto -> chọn cuda nếu GPU hỗ trợ or chạy = CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else: #chạy theo chỉ định (device:cuda)
        device = torch.device(cfg['global']['device'])

    num_classes = cfg['global']['num_classes'] #5
    window_size = cfg['global']['window_size'] #32
    dl_cfg = cfg['rnn_data_generation']['dataloader']
    data_cfg = cfg['data']

    checkpoint_path = args.checkpoint_path if args.checkpoint_path else cfg['rnn_data_generation']['base_checkpoint_to_use']

    log_dir = Path(data_cfg['logs']) #cfg['data'][logs]:PATH(logs)
    train_pkl_path = log_dir / data_cfg['train_inference_raw_name'] #PATH(logs/train_inference_raw.pkl)
    val_pkl_path = log_dir / data_cfg['val_inference_raw_name']
    test_pkl_path = log_dir / data_cfg['test_inference_raw_name']
    processed_dir = Path(data_cfg['processed_dir']) #PATH(data/full_videos)
    train_anno_dir = processed_dir / "train" / "annotations" #PATH(data/full_videos/train/annotations)
    val_anno_dir = processed_dir / "val" / "annotations"
    test_anno_dir = processed_dir / "test" / "annotations"
    rnn_base_dir = Path(data_cfg['rnn_processed_data']) #PATH(rnn_processed_data)
    rnn_train_data_dir = rnn_base_dir / "train" #PATH(rnn_processed_data/train)
    rnn_val_data_dir = rnn_base_dir / "val"
    rnn_test_data_dir = rnn_base_dir / "test"

    log_dir.mkdir(parents=True, exist_ok=True) #tạo file PATH(logs)
    rnn_train_data_dir.mkdir(parents=True, exist_ok=True) #tạo file PATH(rnn_processed_data/train)
    rnn_val_data_dir.mkdir(parents=True, exist_ok=True)
    rnn_test_data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}") 
    print(f"Loading base model checkpoint: {checkpoint_path}")
    #Khởi tạo mô hình TemporalActionDetector với num_classes=5, window_size=32
    model = base_detector.TemporalActionDetector(num_classes=num_classes, window_size=window_size)
    model = model.to(device) #Dùng CPU/GPU
    #Load checkpoint
    if Path(checkpoint_path).exists():
        print(f"Loading checkpoint")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)#load checkpoint vào bộ nhớ (map_location giúp bắt đúng device)
            if 'model_state_dict' in checkpoint: state_dict = checkpoint['model_state_dict'] #Lấy 'model_state_dict' ở trong checkpoint
            #state_dict lưu các tham số
            elif 'state_dict' in checkpoint: state_dict = checkpoint['state_dict'] #Lấy 'state_dict' ở trong checkpoint
            else: state_dict = checkpoint #Nếu checkpoint không chứa 2 key trên -> coi state_dict = checkpoint
            model.load_state_dict(state_dict) #Load trọng số trong model
            print(f"Successfully loaded model weights from epoch {checkpoint.get('epoch', 'N/A')}")
        except Exception as e: #lỗi khi ko load đc file 
            print(f"loading checkpoint: {e} Ensure valid path and compatible model")
            exit()
    else: #lỗi khi ko có file
        print(f"Checkpoint file not found at {checkpoint_path}")
        exit()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}") #tổng tham số của model #(numel()-số phần tử trong p),(1e6-1000000),(:.2f-định dạng float vs 2 chữ thập phân)
    #/cho 1e6 để cho số nhỏ hơn -> dễ nhìn cũng như dễ so sánh với các mô hình khác
    print("Preparing Training Set Inference")
    train_loader = dataloader.get_train_loader(cfg) #load data đã được chuẩn bị
    train_raw_preds, train_batch_meta = run_inference(model, train_loader, device)# dữ liệu đầu vào cho RNN hoặc để đánh giá
    print(f"\nSaving training inference results to: {train_pkl_path}")
    try:
        with open(train_pkl_path, 'wb') as f: #wb-ghi dạng nhị phân nhưng ko đọc được vì đc đóng gói -> nhanh, gọn, an toàn để lưu và khôi phục 
            pickle.dump({'all_raw_preds': train_raw_preds, 'all_batch_meta': train_batch_meta}, f) #chuyển dạng dict sang dạng bytes
        print("Successfully saved training inference results")
    except Exception as e: print(f"saving training inference results: {e}")

    print("Preparing Validation Set Inference")
    val_loader = dataloader.get_val_loader(cfg)
    val_raw_preds, val_batch_meta = run_inference(model, val_loader, device)
    print(f"\nSaving validation inference results to: {val_pkl_path}")
    try:
        with open(val_pkl_path, 'wb') as f:
            pickle.dump({'all_raw_preds': val_raw_preds, 'all_batch_meta': val_batch_meta}, f)
        print("Successfully saved validation inference results")
    except Exception as e: print(f"saving validation inference results: {e}")

    print("Preparing Test Set Inference")
    test_loader = dataloader.get_test_loader(cfg)
    test_raw_preds, test_batch_meta = run_inference(model, test_loader, device)
    print(f"\nSaving test inference results to: {test_pkl_path}")
    try:
        with open(test_pkl_path, 'wb') as f:
            pickle.dump({'all_raw_preds': test_raw_preds, 'all_batch_meta': test_batch_meta}, f)
        print("Successfully saved test inference results")
    except Exception as e: print(f"saving test inference results: {e}")

    print("Processing raw data into RNN input format")

    print("Processing Training Data for RNN")
    helpers.process_predictions_for_rnn(
        num_classes=num_classes,
        window_size=window_size,
        output_pkl=train_pkl_path,
        anno_dir=train_anno_dir,
        rnn_data_dir=rnn_train_data_dir
    )

    print("Processing Validation Data for RNN")
    helpers.process_predictions_for_rnn(
        num_classes=num_classes,
        window_size=window_size,
        output_pkl=val_pkl_path,
        anno_dir=val_anno_dir,
        rnn_data_dir=rnn_val_data_dir
    )
    
    print("Processing Test Data for RNN")
    helpers.process_predictions_for_rnn(
        num_classes=num_classes,
        window_size=window_size,
        output_pkl=test_pkl_path,
        anno_dir=test_anno_dir,
        rnn_data_dir=rnn_test_data_dir
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate raw data for RNN Post-Processor using a trained base model.")
    parser.add_argument('--config', default='configs/config.yaml', help='Path to configuration file')
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Override base model checkpoint path from config") #ghi đè nếu muốn đổi đg dẫn để lưu
    
    args = parser.parse_args()
    
    try:
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Config file not found at {args.config}")
        exit()
    except Exception as e:
        print(f"loading config file: {e}")
        exit()
        
    main(cfg, args) 