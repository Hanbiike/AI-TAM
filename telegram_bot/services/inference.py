import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import segmentation_models_pytorch as smp

# Цветовая палитра как в оригинальной модели
class_colors = np.array([
    [255, 235, 156], # 0: LivingRoom
    [255, 179, 138], # 1: MasterRoom
    [255, 192, 203], # 2: Kitchen
    [160, 210, 255], # 3: Bathroom
    [220, 200, 170], # 4: DiningRoom
    [230, 210, 250], # 5: ChildRoom
    [170, 240, 190], # 6: StudyRoom
    [255, 218, 185], # 7: SecondRoom
    [210, 245, 210], # 8: GuestRoom
    [220, 220, 220], # 9: Balcony
    [175, 235, 235], # 10: Entrance
    [240, 230, 160], # 11: Storage
    [240, 200, 240], # 12: WalkIn
    [255, 255, 255]  # 13: Background (White)
], dtype=np.uint8)

def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Конвертирует маску индексов в RGB-изображение."""
    H, W = mask.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for i in range(14):
        rgb[mask == i] = class_colors[i]
    return rgb

class PlanGenerator:
    def __init__(self, model_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Инициализируем U-Net
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None, # Веса с нуля, так как мы обучали под RPLAN
            in_channels=3,
            classes=14
        ).to(self.device)

        # Загружаем чекпоинт
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()

        # Трансформации для входа такие же, как при обучении
        self.transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=Image.Resampling.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    @torch.no_grad()
    def predict(self, image: Image.Image) -> Image.Image:
        # Убедимся что RGB
        image_rgb = image.convert("RGB")
        # Подготавливаем тензор (добавляем batch dimension)
        tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)

        # Инференс
        logits = self.model(tensor)
        preds = torch.argmax(logits, dim=1).cpu().numpy()[0]

        # Конвертация в цвета
        rgb_arr = mask_to_rgb(preds)
        return Image.fromarray(rgb_arr)

