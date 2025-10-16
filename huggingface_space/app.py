import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import os

# CIFAR-100 class names
CLASS_NAMES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
    'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
    'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, drop_rate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate
        self.shortcut = None

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = self.relu(self.bn1(x))
        shortcut = self.shortcut(out) if self.shortcut is not None else x
        out = self.conv1(out)
        out = self.relu(self.bn2(out))
        out = self.conv2(out)
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return out + shortcut


class NetworkBlock(nn.Module):
    def __init__(self, num_layers, in_channels, out_channels, block, stride, drop_rate):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(
                block(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    stride if i == 0 else 1,
                    drop_rate,
                )
            )
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=10, num_classes=100, drop_rate=0.3):
        super().__init__()
        assert (depth - 4) % 6 == 0, "Depth should be 6n+4"
        n = (depth - 4) // 6
        k = widen_factor
        n_channels = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = nn.Conv2d(3, n_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = NetworkBlock(n, n_channels[0], n_channels[1], BasicBlock, stride=1, drop_rate=drop_rate)
        self.block2 = NetworkBlock(n, n_channels[1], n_channels[2], BasicBlock, stride=2, drop_rate=drop_rate)
        self.block3 = NetworkBlock(n, n_channels[2], n_channels[3], BasicBlock, stride=2, drop_rate=drop_rate)
        self.bn = nn.BatchNorm2d(n_channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(n_channels[3], num_classes)

        # --- He initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        return self.fc(out)


print("Loading model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WideResNet(depth=28, widen_factor=10, num_classes=100).to(device)
checkpoint = torch.load('huggingface_space/wrn_best_model.pth', map_location=device)
model = WideResNet(depth=28, widen_factor=10, num_classes=100).to(device)
model.load_state_dict(checkpoint['model_state_dict'])  
model.eval()
print(f"Model loaded on {device}")

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

def predict(image):
    if image is None:
        return None
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'), 'RGB')
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    
    top5_prob, top5_idx = torch.topk(probabilities, 5)
    results = {CLASS_NAMES[idx]: float(prob) for idx, prob in zip(top5_idx, top5_prob)}
    
    return results

examples = []
if os.path.exists('examples'):
    example_files = [f for f in os.listdir('examples') if f.endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    examples = [[os.path.join('examples', f)] for f in sorted(example_files)]
    print(f"Found {len(examples)} example images")

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload CIFAR-like Image"),
    outputs=gr.Label(num_top_classes=5, label="Top-5 Predictions"),
    title="🎯 WideResNet-28-10 CIFAR-100 Classifier (81.36 % Accuracy)",
    description="""
    ### Upload any image for classification

    **Model achieves 81.36 % Top-1 accuracy on the CIFAR-100 test set**

    **Architecture:** WideResNet-28-10 (36 M parameters)  
    **Training:** 200 epochs on CIFAR-100 from scratch  
    **Techniques:** MixUp · CutMix · Label Smoothing · OneCycleLR · Strong Augmentations

    Try the examples below or upload your own image!
    """,
    article="""
    ### Training Details
    - **Optimizer:** SGD (momentum 0.9, weight decay 5e-4)
    - **Scheduler:** OneCycleLR (max LR = 0.3, pct_start = 0.3, cosine annealing)
    - **Loss:** Cross-Entropy + Label Smoothing (0.1)
    - **Augmentations:** Padding-Crop, Flip, ShiftScaleRotate, CoarseDropout / Noise
    - **Regularization:** Dropout 0.3 inside residual blocks · MixUp/CutMix prob. 0.2
    - **Batch Size:** 256   |  **Epochs:** 200   | 
    - **Implementation:** Trained entirely from scratch on official CIFAR-100 split  

    ### Performance
    - **Top-1 Accuracy:** 81.36 %  
    - **Checkpoint:** [NeuralNoble / wrn-28-10-cifar-100-81.36](https://huggingface.co/NeuralNoble/wrn-28-10-cifar-100-81.36)
    """,
    examples=examples if examples else None,
    theme=gr.themes.Soft(),
    allow_flagging="never",
    cache_examples=False
)

if __name__ == "__main__":
    demo.launch()