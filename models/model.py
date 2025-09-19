import torch
from PIL import Image
import numpy as np
from torchvision import transforms

from models.base import BaseModel
from models.cdan import CDAN
from utils.post_processing import enhance_contrast, enhance_color, sharpen


class Model(BaseModel):
    def __init__(self, network, **kwargs):
        super(Model, self).__init__(**kwargs)
        self.network = network.to(self.device)
        self.network.eval()  # modalità inferenza

    def preprocess(self, image: Image.Image):
        """Preprocessing: resize dinamico e conversione in tensore"""
        image_size = image.size  # (width, height)
        self.transforms = transforms.Compose([
            transforms.Resize((image_size[1], image_size[0])),  # 🔹 height, width
            transforms.ToTensor()
        ])
        return self.transforms(image).unsqueeze(0).to(self.device)

    def postprocess(self, output_tensor, original_size,
                    contrast_factor: float = None,
                    saturation_factor: float = None,
                    sharpen_strength: float = None):
        """Applica post-processing su tensore torch e converte in immagine PIL"""

        # 🔹 Copia tensore di output
        processed_tensor = output_tensor.clone()

        # 🔹 Applica miglioramenti solo se specificati
        if contrast_factor is not None:
            processed_tensor = enhance_contrast(processed_tensor, contrast_factor=contrast_factor)
        if saturation_factor is not None:
            processed_tensor = enhance_color(processed_tensor, saturation_factor=saturation_factor)
        if sharpen_strength is not None and sharpen_strength > 0:
            processed_tensor = sharpen(processed_tensor, strength=sharpen_strength)

        # 🔹 Conversione finale in PIL
        processed_tensor = processed_tensor.squeeze(0).cpu()
        output_image = (processed_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        image = Image.fromarray(output_image)

        return image.resize(original_size)

    def infer(self, image: Image.Image,
              contrast_factor: float = None,
              saturation_factor: float = None,
              sharpen_strength: float = None):
        """Inferenza completa: preprocess -> rete -> postprocess con parametri personalizzabili"""
        input_tensor = self.preprocess(image)
        with torch.no_grad():
            output_tensor = self.network(input_tensor)
        return self.postprocess(output_tensor, image.size,
                                contrast_factor=contrast_factor,
                                saturation_factor=saturation_factor,
                                sharpen_strength=sharpen_strength)
