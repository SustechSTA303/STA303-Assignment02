import torch
import torch.nn as nn

from Model.clip import clip

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CLIP(nn.Module):
    
    def __init__(self, 
                 VISUAL_BACKBONE: str,
                 prompt: str,
                 class_names: list[str]) -> None:
        super().__init__()
        model, preprocess = clip.load(name=VISUAL_BACKBONE, device=device, download_root="Model/checkpoint/")
        model.to(device)
        self.model = model
        self.set_prompt(prompt, class_names)

    def set_prompt(self,
                   prompt: str,
                   class_names: list[str]):
        text_inputs = torch.cat([clip.tokenize(f"{prompt} {c}") for c in class_names]).to(device)
        self.text_inputs = text_inputs
    
    def forward(self,
                image: torch.tensor):

        image_features = self.model.encode_image(image)
        text_features = self.model.encode_text(self.text_inputs)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.model.logit_scale.exp()

        logits = logit_scale * image_features @ text_features.t()

        return logits