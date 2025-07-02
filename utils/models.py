import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, BlipProcessor, BlipForConditionalGeneration, ViTModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=1)
        return torch.sum(weights * x, dim=1)

class Task1Model(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.blip = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(device)
        self.bert = AutoModel.from_pretrained("bert-base-multilingual-cased").to(device)

        vh = self.blip.vision_model.config.hidden_size
        th = self.bert.config.hidden_size
        self.attn_pool = AttentionPooling(th)
        total_dim = vh + 2 * th

        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, images, text_inputs, raw_texts):
        vis = self.blip.vision_model(pixel_values=images).last_hidden_state
        img_feat = vis[:, 0, :]

        self.blip.eval()
        with torch.no_grad():
            gen_ids = self.blip.generate(pixel_values=images, max_length=16, num_beams=1)
        self.blip.train()

        caps = blip_processor.batch_decode(gen_ids, skip_special_tokens=True)
        cap_inputs = tokenizer(caps, padding=True, truncation=True, max_length=96, return_tensors="pt").to(device)
        cap_hid = self.bert(**cap_inputs).last_hidden_state
        cap_feat = self.attn_pool(cap_hid)

        orig_hid = self.bert(**text_inputs).last_hidden_state
        orig_feat = self.attn_pool(orig_hid)

        combined = torch.cat([img_feat, cap_feat, orig_feat], dim=1)
        return self.classifier(combined)

class Task2Model(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(self.vit.config.hidden_size + self.bert.config.hidden_size, num_classes)

    def forward(self, image, text_inputs):
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        image_features = self.vit(image)['last_hidden_state'][:, 0, :]
        text_features = self.bert(**text_inputs)["last_hidden_state"][:, 0, :]
        combined = torch.cat((image_features, text_features), dim=-1)
        return self.fc(combined)

# Load BLIP/BERT tokenizers outside class
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
tokenizer_task2 = AutoTokenizer.from_pretrained("bert-base-uncased")

def load_task1_model():
    model = Task1Model()
    checkpoint = torch.load("checkpoints/best_model_task1_rank(1).pt", map_location=device)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device).eval()
    return model


def load_task2_model():
    model = Task2Model()
    checkpoint = torch.load("checkpoints/best_model_checkpoint_task2.pth", map_location=device)

    # If checkpoint is a full dictionary with keys like 'model_state_dict', use this:
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device).eval()
    return model

