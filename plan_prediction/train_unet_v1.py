import os
import argparse
import json
import threading
import datetime
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path
from http.server import SimpleHTTPRequestHandler, HTTPServer
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

import segmentation_models_pytorch as smp

# =====================================================================
# Configuration Dataclass
# =====================================================================
@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 8
    lr: float = 2e-4
    img_size: int = 256
    limit: Optional[int] = None
    save_every: int = 5
    sample_every: int = 1
    workers: int = 4
    compile: bool = False
    port: int = 8080
    base_dir: str = r"c:\Users\hanbi\Downloads\RPLAN dataset\Network"
    resume: bool = True  # Automatically resume training if checkpoint exists
    model_version: str = "1.0.0-segmentation"
    max_grad_norm: float = 1.0  # Max gradient norm for clipping
    loss_ema_decay: float = 0.95  # Decay rate for Loss EMA tracking
    encoder_name: str = "resnet34"  # Efficient default encoder for RTX 4060
    use_class_weights: bool = True  # Perform class imbalance analysis & weights
    dice_weight: float = 1.0  # Weight multiplier for Dice Loss in Combined Loss

# =====================================================================
# Global Training Status (Shared with Web Server Thread)
# =====================================================================
class TrainingStatus:
    def __init__(self):
        self.epoch: int = 0
        self.total_epochs: int = 0
        self.loss: float = 0.0
        self.loss_ema: float = 0.0
        self.pixel_acc: float = 0.0
        self.miou: float = 0.0
        self.dice: float = 0.0
        self.status: str = "Инициализация"

training_status = TrainingStatus()

# =====================================================================
# Color mappings and Segmentation Helpers
# =====================================================================
ROOM_COLORS = {
    0: (255, 235, 156),   # LivingRoom
    1: (255, 179, 138),   # MasterRoom
    2: (255, 192, 203),   # Kitchen
    3: (160, 210, 255),   # Bathroom
    4: (220, 200, 170),   # DiningRoom
    5: (230, 210, 250),   # ChildRoom
    6: (170, 240, 190),   # StudyRoom
    7: (255, 218, 185),   # SecondRoom
    8: (210, 245, 210),   # GuestRoom
    9: (220, 220, 220),   # Balcony
    10: (175, 235, 235),  # Entrance
    11: (240, 230, 160),  # Storage
    12: (240, 200, 240),  # WalkIn
}

# 13 is Background / White
BACKGROUND_COLOR = (255, 255, 255)

# Convert map colors to array for fast mask-to-RGB conversion
class_colors = np.array([
    [255, 235, 156], # 0
    [255, 179, 138], # 1
    [255, 192, 203], # 2
    [160, 210, 255], # 3
    [220, 200, 170], # 4
    [230, 210, 250], # 5
    [170, 240, 190], # 6
    [255, 218, 185], # 7
    [210, 245, 210], # 8
    [220, 220, 220], # 9
    [175, 235, 235], # 10
    [240, 230, 160], # 11
    [240, 200, 240], # 12
    [255, 255, 255]  # 13 (Background)
], dtype=np.uint8)

def rgb_to_mask(img_pil: Image.Image) -> np.ndarray:
    """Vectorized exact color mapping from RGB image to class index mask."""
    arr = np.array(img_pil)
    H, W, _ = arr.shape
    # Default to background class (13)
    mask = np.full((H, W), 13, dtype=np.uint8)
    
    # Exact-match mapping for all room color categories
    for cls, color in ROOM_COLORS.items():
        match = (arr == color).all(axis=-1)
        mask[match] = cls
        
    return mask.astype(np.int64)

def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Converts a class index mask of shape (H, W) back to an RGB image array."""
    H, W = mask.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for i in range(14):
        rgb[mask == i] = class_colors[i]
    return rgb

# =====================================================================
# Dashboard HTML Template
# =====================================================================
def get_dashboard_html() -> str:
    return """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Монитор Сегментации Планировок RPLAN</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-base: #0b0f19;
            --bg-surface: #151d30;
            --bg-card: #1e293b;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --primary: #06b6d4;
            --primary-glow: rgba(6, 182, 212, 0.15);
            --success: #10b981;
            --border: #334155;
        }

        body {
            font-family: 'Outfit', sans-serif;
            background-color: var(--bg-base);
            color: var(--text-main);
            margin: 0;
            padding: 0;
            display: flex;
            flex-direction: column;
            min-height: 100vh;
        }

        header {
            background-color: var(--bg-surface);
            border-bottom: 1px solid var(--border);
            padding: 15px 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        header h1 {
            margin: 0;
            font-size: 22px;
            font-weight: 700;
            background: linear-gradient(135deg, #38bdf8, var(--primary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .status-badge {
            background-color: rgba(6, 182, 212, 0.1);
            border: 1px solid var(--primary);
            color: var(--primary);
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 600;
        }

        .main-container {
            display: flex;
            flex: 1;
            padding: 20px;
            gap: 20px;
            max-width: 1600px;
            margin: 0 auto;
            width: calc(100% - 40px);
            box-sizing: border-box;
        }

        .left-panel {
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 20px;
            max-width: 450px;
        }

        .right-panel {
            flex: 2;
            background-color: var(--bg-surface);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 24px;
            display: flex;
            flex-direction: column;
            align-items: center;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        }

        .card {
            background-color: var(--bg-surface);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
        }

        .stat-box {
            background-color: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 12px 8px;
            text-align: center;
        }

        .stat-label {
            font-size: 11px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 5px;
        }

        .stat-value {
            font-size: 18px;
            font-weight: 700;
            color: var(--text-main);
        }

        .history-list {
            max-height: 400px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 8px;
            padding-right: 5px;
        }

        .history-list::-webkit-scrollbar {
            width: 6px;
        }
        .history-list::-webkit-scrollbar-thumb {
            background-color: var(--border);
            border-radius: 3px;
        }

        .history-item {
            background-color: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 12px 16px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: all 0.2s ease;
        }

        .history-item:hover {
            border-color: var(--primary);
            background-color: rgba(6, 182, 212, 0.05);
            transform: translateX(3px);
        }

        .history-item.active {
            border-color: var(--primary);
            background-color: var(--primary-glow);
            font-weight: 600;
        }

        .image-viewer {
            width: 100%;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 15px;
        }

        .image-container {
            background-color: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 10px;
            display: flex;
            justify-content: center;
            align-items: center;
            max-width: 100%;
            overflow: hidden;
            box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.06);
        }

        .image-container img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            display: block;
        }

        .legend-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            width: 100%;
            max-width: 768px;
            text-align: center;
            font-size: 14px;
            font-weight: 600;
            color: var(--text-muted);
        }
    </style>
</head>
<body>

    <header>
        <h1>RPLAN Segmentation Monitor</h1>
        <div id="status-badge" class="status-badge">Загрузка...</div>
    </header>

    <div class="main-container">
        <!-- Left Panel: Stats and History -->
        <div class="left-panel">
            <div class="card">
                <h3 style="margin-top:0; margin-bottom:15px; font-size:16px;">Параметры Обучения</h3>
                <div class="stats-grid">
                    <div class="stat-box">
                        <div class="stat-label">Эпоха</div>
                        <div id="stat-epoch" class="stat-value">-</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Loss (EMA)</div>
                        <div id="stat-loss" class="stat-value">-</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Pixel Accuracy</div>
                        <div id="stat-acc" class="stat-value">-</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Mean IoU / Dice</div>
                        <div id="stat-miou-dice" class="stat-value">-</div>
                    </div>
                </div>
            </div>

            <div class="card" style="flex: 1; display:flex; flex-direction:column;">
                <h3 style="margin-top:0; margin-bottom:15px; font-size:16px;">История генераций</h3>
                <div id="history-list" class="history-list">
                    <!-- Dynamic history items -->
                </div>
            </div>
        </div>

        <!-- Right Panel: Main Image Viewer -->
        <div class="right-panel">
            <div class="image-viewer">
                <h3 id="viewer-title" style="margin:0; font-size:18px;">Текущая планировка</h3>
                <div class="legend-grid">
                    <div>Входная Граница</div>
                    <div>Предсказание маски</div>
                    <div>Ground Truth</div>
                </div>
                <div class="image-container">
                    <img id="main-image" src="" alt="Ожидание генерации первого сэмпла...">
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentActiveFile = "";
        let knownSamples = [];

        async function updateStatus() {
            try {
                const response = await fetch("/api/status");
                const data = await response.json();
                
                document.getElementById("status-badge").innerText = data.status;
                document.getElementById("stat-epoch").innerText = data.current_epoch + " / " + data.total_epochs;
                document.getElementById("stat-loss").innerText = Number(data.loss_ema).toFixed(6);
                document.getElementById("stat-acc").innerText = Number(data.pixel_acc * 100).toFixed(2) + "%";
                document.getElementById("stat-miou-dice").innerText = Number(data.miou * 100).toFixed(2) + "% / " + Number(data.dice * 100).toFixed(2) + "%";

                const historyList = document.getElementById("history-list");
                let isNewAdded = false;

                data.samples.forEach(sample => {
                    if (!knownSamples.includes(sample)) {
                        knownSamples.push(sample);
                        isNewAdded = true;
                    }
                });

                if (isNewAdded || historyList.children.length === 0) {
                    historyList.innerHTML = "";
                    const sortedSamples = [...knownSamples].sort((a, b) => {
                        const matchA = a.match(/\\d+/);
                        const matchB = b.match(/\\d+/);
                        const numA = matchA ? parseInt(matchA[0]) : 0;
                        const numB = matchB ? parseInt(matchB[0]) : 0;
                        return numB - numA;
                    });

                    sortedSamples.forEach((sample, index) => {
                        const match = sample.match(/\\d+/);
                        const epochNum = match ? match[0] : "?";
                        const div = document.createElement("div");
                        div.className = "history-item";
                        if (sample === currentActiveFile || (currentActiveFile === "" && index === 0)) {
                            div.className += " active";
                            if (currentActiveFile === "") {
                                selectImage(sample);
                            }
                        }
                        div.innerHTML = `<span>Эпоха ${epochNum}</span><span style="font-size: 12px; color: var(--text-muted);">${sample}</span>`;
                        div.onclick = () => selectImage(sample);
                        historyList.appendChild(div);
                    });
                }
            } catch (err) {
                console.error("Ошибка обновления статуса:", err);
            }
        }

        function selectImage(filename) {
            currentActiveFile = filename;
            document.getElementById("main-image").src = "/samples/" + filename + "?t=" + new Date().getTime();
            const match = filename.match(/\\d+/);
            const epochNum = match ? match[0] : "?";
            document.getElementById("viewer-title").innerText = "Детали сегментации: Эпоха " + epochNum;
            
            const items = document.querySelectorAll(".history-item");
            items.forEach(item => {
                if (item.innerHTML.includes(filename)) {
                    item.className = "history-item active";
                } else {
                    item.className = "history-item";
                }
            });
        }

        setInterval(updateStatus, 3000);
        updateStatus();
    </script>
</body>
</html>
"""

# =====================================================================
# Dashboard HTTP Request Handler
# =====================================================================
class MonitorHandler(SimpleHTTPRequestHandler):
    base_dir: Path = Path(r"c:\Users\hanbi\Downloads\RPLAN dataset\Network")

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(get_dashboard_html().encode("utf-8"))
        elif self.path.startswith("/samples/"):
            filename = self.path[len("/samples/"):]
            if "?" in filename:
                filename = filename.split("?")[0]
            sample_file = self.base_dir / "diffusion_samples" / filename
            if sample_file.exists() and sample_file.is_file():
                self.send_response(200)
                self.send_header("Content-type", "image/png")
                self.end_headers()
                with open(sample_file, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, "Sample image not found")
        elif self.path == "/api/status":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            
            sample_dir = self.base_dir / "diffusion_samples"
            samples_list = []
            if sample_dir.exists() and sample_dir.is_dir():
                samples_list = sorted([f.name for f in sample_dir.glob("*.png")])
                
            status_data = {
                "current_epoch": training_status.epoch,
                "total_epochs": training_status.total_epochs,
                "loss": training_status.loss,
                "loss_ema": training_status.loss_ema,
                "pixel_acc": training_status.pixel_acc,
                "miou": training_status.miou,
                "dice": training_status.dice,
                "status": training_status.status,
                "samples": samples_list,
            }
            self.wfile.write(json.dumps(status_data).encode("utf-8"))
        else:
            self.send_error(404, "Not found")

def start_monitor_server(port: int, base_dir: Path) -> None:
    try:
        MonitorHandler.base_dir = base_dir
        server = HTTPServer(("0.0.0.0", port), MonitorHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        print(f"\n🖥️  [Dashboard] Монитор запущен по адресу: http://localhost:{port}/")
    except Exception as e:
        print(f"Не удалось запустить сервер мониторинга: {e}")

# =====================================================================
# 1. Dataset for Boundary-to-Full Layout
# =====================================================================
class LayoutDataset(Dataset):
    """
    RPLAN dataset loader for boundary images (conditioning) and class mask targets.
    
    Pairs boundary and full layout target files by having matching names:
      boundary_dir: boundary_XXX.png (or XXX.png)
      full_dir:     layout_XXX.png   (or XXX.png)
    """
    def __init__(self, boundary_dir: Path, full_dir: Optional[Path], img_size: int = 256, limit: Optional[int] = None):
        self.boundary_dir = Path(boundary_dir)
        self.full_dir = Path(full_dir) if full_dir else None
        self.img_size = img_size
        
        self.filenames: List[str] = sorted([f.name for f in self.boundary_dir.glob("*.png") if f.is_file()])
        if limit:
            self.filenames = self.filenames[:limit]
            
        self.transform_img = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        filename = self.filenames[idx]
        cond_path = self.boundary_dir / filename
        cond_img = Image.open(cond_path).convert("RGB")
        cond_tensor = self.transform_img(cond_img)
        
        if self.full_dir:
            target_path = self.full_dir / filename
            target_img = Image.open(target_path).convert("RGB")
            # Nearest neighbor interpolation ensures category boundary exactness without color blending
            target_img_resized = target_img.resize((self.img_size, self.img_size), Image.Resampling.NEAREST)
            target_mask = rgb_to_mask(target_img_resized)
            target_tensor = torch.from_numpy(target_mask).long()
            return cond_tensor, target_tensor
            
        return cond_tensor

# =====================================================================
# 2. Forward inference helper using semantic segmentation model
# =====================================================================
@torch.no_grad()
def sample_layout(model: nn.Module, cond_images: torch.Tensor) -> torch.Tensor:
    """
    Forward pass semantic segmentation inference pipeline.
    Transforms predicted class logits back to RGB layout color space tensor of shape (B, 3, H, W).
    """
    model.eval()
    logits = model(cond_images)
    preds = torch.argmax(logits, dim=1).cpu().numpy() # shape (B, H, W)
    
    B, H, W = preds.shape
    rgb_layouts = []
    for i in range(B):
        rgb_arr = mask_to_rgb(preds[i])
        # Transpose and normalize to match [0, 1] range expected by visual grid builders
        layout_tensor = torch.from_numpy(rgb_arr).float() / 255.0
        layout_tensor = layout_tensor.permute(2, 0, 1) # (3, H, W)
        rgb_layouts.append(layout_tensor)
        
    return torch.stack(rgb_layouts).to(cond_images.device)

# =====================================================================
# 3. Checkpoint Auto-Resume Helper
# =====================================================================
def find_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """Scans the checkpoint directory and returns the path of the latest epoch checkpoint."""
    ckpts = list(checkpoint_dir.glob("diffusion_epoch_*.pth"))
    if not ckpts:
        return None
    epochs = []
    for ckpt in ckpts:
        try:
            epoch_num = int(ckpt.stem.split("epoch_")[-1])
            epochs.append((epoch_num, ckpt))
        except ValueError:
            continue
    if not epochs:
        return None
    return max(epochs, key=lambda x: x[0])[1]

# =====================================================================
# 4. Metrics Tracker Implementation
# =====================================================================
def compute_metrics(preds: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float, float]:
    """Computes Pixel Accuracy, mean IoU (mIoU), and Mean Dice Score over 14 classes."""
    # 1. Pixel Accuracy
    correct = (preds == targets).sum().item()
    total = targets.numel()
    pixel_acc = correct / total if total > 0 else 0.0
    
    # 2. IoU and Dice Score per class
    ious = []
    dices = []
    for c in range(14):
        pred_c = (preds == c)
        target_c = (targets == c)
        
        intersection = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()
        target_sum = target_c.sum().item()
        pred_sum = pred_c.sum().item()
        
        # Skip class if it is not present in both target and prediction
        if target_sum == 0 and pred_sum == 0:
            continue
            
        iou = intersection / union if union > 0 else 0.0
        dice = (2.0 * intersection) / (pred_sum + target_sum) if (pred_sum + target_sum) > 0 else 0.0
        
        ious.append(iou)
        dices.append(dice)
        
    mean_iou = sum(ious) / len(ious) if ious else 0.0
    mean_dice = sum(dices) / len(dices) if dices else 0.0
    
    return pixel_acc, mean_iou, mean_dice

# =====================================================================
# 5. Combined Loss Function (CrossEntropyLoss + Multiclass DiceLoss)
# =====================================================================
class CombinedLoss(nn.Module):
    def __init__(self, ce_weight: Optional[torch.Tensor] = None, dice_weight: float = 1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=ce_weight)
        self.dice_weight = dice_weight
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)
        
        # Multiclass Soft Dice Loss
        probs = torch.softmax(logits, dim=1)
        B, C, H, W = logits.shape
        
        # One-hot target representation
        targets_one_hot = torch.zeros_like(logits)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1.0)
        
        intersection = (probs * targets_one_hot).sum(dim=(0, 2, 3))
        cardinality = (probs + targets_one_hot).sum(dim=(0, 2, 3))
        
        smooth = 1.0
        dice_loss = 1.0 - ((2.0 * intersection + smooth) / (cardinality + smooth)).mean()
        
        return ce_loss + self.dice_weight * dice_loss

# =====================================================================
# 6. Training Loop
# =====================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Train a semantic segmentation model for RPLAN layout layouts.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--img_size", type=int, default=256, help="Resolution of images (e.g. 256 or 128).")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of training samples (for testing).")
    parser.add_argument("--save_every", type=int, default=1, help="Save checkpoints every N epochs.")
    parser.add_argument("--sample_every", type=int, default=1, help="Generate and save samples every N epochs.")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers (default 4).")
    parser.add_argument("--compile", action="store_true", help="Compile model using PyTorch 2.x compile.")
    parser.add_argument("--port", type=int, default=8080, help="Port for the monitoring web dashboard.")
    parser.add_argument("--base_dir", type=str, default=r"c:\Users\hanbi\Downloads\RPLAN dataset\Network", help="Base directory of the project.")
    parser.add_argument("--resume", type=bool, default=True, help="Whether to auto-resume from checkpoints.")
    parser.add_argument("--model_version", type=str, default="1.0.0-segmentation", help="Model version metadata.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping.")
    parser.add_argument("--loss_ema_decay", type=float, default=0.95, help="Decay rate for loss EMA.")
    parser.add_argument("--encoder_name", type=str, default="resnet34", help="Encoder model type.")
    parser.add_argument("--use_class_weights", type=bool, default=True, help="Compute class balancing weights.")
    parser.add_argument("--dice_weight", type=float, default=1.0, help="Dice Loss weight in Combined Loss.")
    args = parser.parse_args()

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        img_size=args.img_size,
        limit=args.limit,
        save_every=args.save_every,
        sample_every=args.sample_every,
        workers=args.workers,
        compile=args.compile,
        port=args.port,
        base_dir=args.base_dir,
        resume=args.resume,
        model_version=args.model_version,
        max_grad_norm=args.max_grad_norm,
        loss_ema_decay=args.loss_ema_decay,
        encoder_name=args.encoder_name,
        use_class_weights=args.use_class_weights,
        dice_weight=args.dice_weight
    )

    # Opt 1: cuDNN benchmarking
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    base_dir = Path(config.base_dir)
    train_boundary_dir = base_dir / "generated_images" / "train_boundary"
    train_full_dir = base_dir / "generated_images" / "train_full"
    
    checkpoint_dir = base_dir / "diffusion_checkpoints"
    sample_dir = base_dir / "diffusion_samples"
    checkpoint_dir.mkdir(exist_ok=True)
    sample_dir.mkdir(exist_ok=True)

    # Dataloader Setup
    print("Loading dataset...")
    dataset = LayoutDataset(train_boundary_dir, train_full_dir, img_size=config.img_size, limit=config.limit)
    
    # Opt 2: DataLoader optimizations: pin_memory and persistent_workers
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        drop_last=True,
        num_workers=config.workers if device == "cuda" else 0,
        pin_memory=(device == "cuda"),
        persistent_workers=(device == "cuda" and config.workers > 0)
    )
    print(f"Dataset size: {len(dataset)} samples. Batches: {len(dataloader)}")

    # Initialize Segmentation UNet Model
    print(f"Initializing smp.Unet with encoder: {config.encoder_name}...")
    model = smp.Unet(
        encoder_name=config.encoder_name,
        encoder_weights=None,  # Trained from scratch to fit RPLAN outlines
        in_channels=3,
        classes=14
    ).to(device)

    # Fused AdamW optimizer
    use_fused = (device == "cuda")
    try:
        optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4, fused=use_fused)
    except Exception:
        optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)

    # Automatic Mixed Precision (AMP)
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    # Load validation images once to prevent next(iter()) inside epoch loops
    val_loader = DataLoader(dataset, batch_size=4, shuffle=False)
    val_batch = next(iter(val_loader))
    val_conds, val_targets = val_batch[0].to(device), val_batch[1].to(device)

    # Class distribution analysis at training startup to address layout class imbalances
    class_weights = None
    if config.use_class_weights:
        # Sample a subset of training data to analyze distribution quickly
        num_samples = min(2000, len(dataset))
        print(f"Analyzing dataset class distribution (sampling {num_samples} images for speed)...")
        class_counts = torch.zeros(14, dtype=torch.int64)
        indices = torch.randperm(len(dataset))[:num_samples]
        for idx in tqdm(indices, desc="Analyzing classes"):
            _, mask = dataset[idx.item()]
            unique, counts = torch.unique(mask, return_counts=True)
            for u, c in zip(unique, counts):
                class_counts[u.item()] += c.item()
                
        total_pixels = class_counts.sum().item()
        class_freqs = class_counts.float() / total_pixels
        class_freqs = torch.clamp(class_freqs, min=1e-6)
        
        # Calculate median frequency weights
        median_freq = torch.median(class_freqs[class_freqs > 1e-5])
        class_weights = (median_freq / class_freqs).to(device)
        # Cap max weights to prevent gradient explosion on extremely rare classes
        class_weights = torch.clamp(class_weights, min=0.1, max=10.0)
        
        print("\n📊  [Class Distribution Analysis & Weights]")
        for i in range(14):
            percentage = class_freqs[i].item() * 100
            print(f"Class {i:2d}: {class_counts[i].item():12d} pixels ({percentage:6.2f}%) | Weight: {class_weights[i].item():.4f}")
        print("")

    # Construct loss criterion (Weighted CrossEntropy + Dice Loss)
    criterion = CombinedLoss(ce_weight=class_weights, dice_weight=config.dice_weight)

    # Resume Training logic with verification logging
    start_epoch = 1
    best_loss = float('inf')
    
    # Try to load best loss from existing best_model.pth if it exists
    best_model_path = checkpoint_dir / "best_model.pth"
    if best_model_path.exists():
        try:
            best_checkpoint = torch.load(best_model_path, map_location=device)
            best_loss = best_checkpoint.get('loss', float('inf'))
            print(f"Loaded best loss from existing best_model.pth: {best_loss:.6f}")
        except Exception as e:
            print(f"Could not load best loss from best_model.pth: {e}")

    latest_ckpt = find_latest_checkpoint(checkpoint_dir)
    checkpoint = None
    if config.resume and latest_ckpt:
        print(f"🤖  [Resume] Найдена последняя запись: {latest_ckpt.name}. Загрузка состояния...")
        try:
            checkpoint = torch.load(latest_ckpt, map_location=device)
        except Exception as e:
            print(f"⚠️  [Ошибка] Не удалось загрузить чекпоинт {latest_ckpt.name}: {e}")
            print("   Возможно, файл поврежден. Пробуем найти предыдущие чекпоинты...")
            ckpts = sorted(list(checkpoint_dir.glob("diffusion_epoch_*.pth")), key=lambda x: int(x.stem.split("epoch_")[-1]) if x.stem.split("epoch_")[-1].isdigit() else 0, reverse=True)
            for ckpt in ckpts:
                if ckpt == latest_ckpt:
                    continue
                try:
                    print(f"🤖  [Resume] Пробуем альтернативный чекпоинт: {ckpt.name}...")
                    checkpoint = torch.load(ckpt, map_location=device)
                    latest_ckpt = ckpt
                    break
                except Exception as ex:
                    print(f"⚠️  [Ошибка] Не удалось загрузить {ckpt.name}: {ex}")
            if checkpoint is None:
                print("❌  [Resume] Не удалось загрузить ни один чекпоинт. Начинаем обучение с нуля.")

    if checkpoint is not None:
        # 1. Restore model
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("   - Модель успешно восстановлена")
        else:
            print("   - [WARN] model_state_dict отсутствует в чекпоинте!")
            
        # 2. Restore optimizer
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("   - Оптимизатор успешно восстановлен")
        else:
            print("   - [WARN] optimizer_state_dict отсутствует в чекпоинте!")
            
        # 3. Restore scaler
        if scaler and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict']:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print("   - GradScaler успешно восстановлен")
        elif scaler:
            print("   - GradScaler не был восстановлен (отсутствует в чекпоинте)")
            
        start_epoch = checkpoint.get('epoch', 0) + 1
        training_status.epoch = checkpoint.get('epoch', 0)
        training_status.loss = checkpoint.get('loss', 0.0)
        training_status.loss_ema = checkpoint.get('loss_ema', training_status.loss)
        training_status.pixel_acc = checkpoint.get('pixel_acc', 0.0)
        training_status.miou = checkpoint.get('miou', 0.0)
        training_status.dice = checkpoint.get('dice', 0.0)
        
        if best_loss == float('inf'):
            best_loss = checkpoint.get('best_loss', checkpoint.get('loss', float('inf')))
            
        print(f"🤖  [Resume] Обучение возобновлено с эпохи {start_epoch} (best_loss: {best_loss:.6f})")
    else:
        print("🌱  [Start] Обучение запущено с нуля.")

    # PyTorch 2.x Compile option (Safe platform check for Windows)
    if config.compile:
        if os.name == 'nt':
            print("\n⚠️  [Предупреждение] torch.compile() официально не поддерживается на Windows из-за отсутствия компилятора Triton.")
            print("   Компиляция пропущена, обучение запущено в стандартном режиме.\n")
        else:
            try:
                print("Compiling model for extra speedup (this takes a minute)...")
                model = torch.compile(model)
            except Exception as e:
                print(f"Не удалось скомпилировать модель: {e}")

    # Start Dashboard Server
    training_status.total_epochs = config.epochs
    training_status.status = "Идет обучение"
    start_monitor_server(port=config.port, base_dir=base_dir)

    print("Starting training loop...")
    running_loss_ema = training_status.loss_ema if training_status.loss_ema > 0.0 else None

    for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_pixel_acc = 0.0
        epoch_miou = 0.0
        epoch_dice = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config.epochs}")
        for conds, targets in pbar:
            conds = conds.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with AMP autocast
            if scaler:
                with torch.amp.autocast("cuda"):
                    logits = model(conds)
                    loss = criterion(logits, targets)
                scaler.scale(loss).backward()
                
                # Gradient clipping with scaler
                if config.max_grad_norm is not None and config.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(conds)
                loss = criterion(logits, targets)
                loss.backward()
                
                # Gradient clipping without scaler
                if config.max_grad_norm is not None and config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    
                optimizer.step()
            
            loss_val = loss.item()
            if running_loss_ema is None:
                running_loss_ema = loss_val
            else:
                running_loss_ema = config.loss_ema_decay * running_loss_ema + (1.0 - config.loss_ema_decay) * loss_val
                
            # Batch metrics tracking
            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                pixel_acc, mean_iou, mean_dice = compute_metrics(preds, targets)
                
            epoch_loss += loss_val
            epoch_pixel_acc += pixel_acc
            epoch_miou += mean_iou
            epoch_dice += mean_dice
            
            pbar.set_postfix(
                loss=loss_val, 
                loss_ema=running_loss_ema, 
                acc=pixel_acc, 
                miou=mean_iou
            )
            
        avg_loss = epoch_loss / len(dataloader)
        avg_acc = epoch_pixel_acc / len(dataloader)
        avg_miou = epoch_miou / len(dataloader)
        avg_dice = epoch_dice / len(dataloader)
        print(f"Epoch {epoch} complete. Loss: {avg_loss:.6f} | Acc: {avg_acc:.4f} | mIoU: {avg_miou:.4f} | Dice: {avg_dice:.4f}")
        
        # Update Web Dashboard status
        training_status.epoch = epoch
        training_status.loss = avg_loss
        training_status.loss_ema = running_loss_ema
        training_status.pixel_acc = avg_acc
        training_status.miou = avg_miou
        training_status.dice = avg_dice
        
        # Save checkpoint
        if epoch % config.save_every == 0 or epoch == config.epochs:
            ckpt_path = checkpoint_dir / f"diffusion_epoch_{epoch}.pth"
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'loss': avg_loss,
                'loss_ema': running_loss_ema,
                'pixel_acc': avg_acc,
                'miou': avg_miou,
                'dice': avg_dice,
                'best_loss': best_loss,
                'config': asdict(config),
                'model_version': config.model_version,
                'created_at': datetime.datetime.now().isoformat(),
            }
            torch.save(checkpoint_data, ckpt_path)
            
            # Save corresponding JSON metadata file
            meta_path = checkpoint_dir / f"diffusion_epoch_{epoch}.json"
            meta_data = {
                'epoch': epoch,
                'loss': avg_loss,
                'loss_ema': running_loss_ema,
                'pixel_acc': avg_acc,
                'miou': avg_miou,
                'dice': avg_dice,
                'best_loss': best_loss,
                'model_version': config.model_version,
                'created_at': checkpoint_data['created_at'],
                'config': asdict(config)
            }
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta_data, f, indent=4, ensure_ascii=False)
            print(f"Saved checkpoint and metadata: {ckpt_path}")
            
        # Check and save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_ckpt_path = checkpoint_dir / "best_model.pth"
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'loss': avg_loss,
                'loss_ema': running_loss_ema,
                'pixel_acc': avg_acc,
                'miou': avg_miou,
                'dice': avg_dice,
                'best_loss': best_loss,
                'config': asdict(config),
                'model_version': config.model_version,
                'created_at': datetime.datetime.now().isoformat(),
            }
            torch.save(checkpoint_data, best_ckpt_path)
            
            # Save corresponding JSON metadata file
            meta_path = checkpoint_dir / "best_model.json"
            meta_data = {
                'epoch': epoch,
                'loss': avg_loss,
                'loss_ema': running_loss_ema,
                'pixel_acc': avg_acc,
                'miou': avg_miou,
                'dice': avg_dice,
                'best_loss': best_loss,
                'model_version': config.model_version,
                'created_at': checkpoint_data['created_at'],
                'config': asdict(config)
            }
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta_data, f, indent=4, ensure_ascii=False)
            print(f"🔥  [Best Model] New best average loss: {best_loss:.6f}. Saved {best_ckpt_path}")
            
        # Sample and save intermediate segmentation visual layouts
        if epoch % config.sample_every == 0:
            training_status.status = f"Генерация сэмпла (эпоха {epoch})"
            print("Generating segmentation prediction grid...")
            
            model.eval()
            with torch.no_grad():
                if scaler:
                    with torch.amp.autocast("cuda"):
                        logits = model(val_conds)
                else:
                    logits = model(val_conds)
                preds = torch.argmax(logits, dim=1).cpu().numpy() # shape (4, H, W)
                
            # Generate comparison grid (Boundary | Predicted Segmentation | Ground Truth Layout)
            grid_img = Image.new("RGB", (256 * 3, 256 * 4))
            for i in range(4):
                # 1. Denormalized boundary image
                cond_denorm = (val_conds[i].cpu() * 0.5) + 0.5
                cond_pil = transforms.ToPILImage()(cond_denorm)
                grid_img.paste(cond_pil.resize((256, 256)), (0, i * 256))
                
                # 2. Predicted Layout
                pred_rgb = mask_to_rgb(preds[i])
                pred_pil = Image.fromarray(pred_rgb)
                grid_img.paste(pred_pil, (256, i * 256))
                
                # 3. Ground Truth Layout
                target_mask = val_targets[i].cpu().numpy()
                target_rgb = mask_to_rgb(target_mask)
                target_pil = Image.fromarray(target_rgb)
                grid_img.paste(target_pil, (512, i * 256))
                
            sample_path = sample_dir / f"epoch_{epoch}_sample.png"
            grid_img.save(sample_path)
            print(f"Saved validation sample grid: {sample_path}")
            
            training_status.status = "Идет обучение"

    training_status.status = "Обучение завершено"
    print("Training finished successfully!")

if __name__ == "__main__":
    main()
