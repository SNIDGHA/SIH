import numpy as np
import torch
import torchvision.transforms as transforms
from flask import Flask, request, send_file, render_template
from PIL import Image
import io
import torch.nn as nn
import pickle

app = Flask(__name__)


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), # Pads the input tensor using the reflection of the input boundary
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features), 
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_block):
        super(GeneratorResNet, self).__init__()

        channels = input_shape[0]

        # Initial Convolution Block
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True)
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_block):
            model += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2), # --> width*2, heigh*2
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features

        # Output Layer
        model += [nn.ReflectionPad2d(channels),
                  nn.Conv2d(out_features, channels, 7),
                  nn.Tanh()
                 ]

        # Unpacking
        self.model = nn.Sequential(*model) 

    def forward(self, x):
        return self.model(x)


# Initialize the model
input_shape = (3, 256, 256)  # Adjust based on your model's expected input
num_residual_blocks = 9  # Adjust based on your model

# model = GeneratorResNet(input_shape, num_residual_blocks)
# model_path = 'generator.sav'  # Path to your .sav file

# Load the model from the .sav file

model=torch.load('backup.pth',map_location=torch.device('cpu'))
# with open(model_path, 'rb') as f:
#     model = pickle.load(f)
# model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize(
        (256, 256)),  # Adjust size based on your model's input size
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


@app.route('/', methods=["GET"])
def home():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return {"error": "No file provided"}, 400

    file = request.files['file'].read()
    img = Image.open(io.BytesIO(file)).convert(
        'RGB')  # Ensure image is in RGB mode
    img = transform(img).unsqueeze(0)  # Transform and add batch dimension

    with torch.no_grad():
        output = model(img)

    output_img = output.squeeze().permute(1, 2, 0).cpu().numpy()

    output_img = (output_img - output_img.min()) / (
        output_img.max() - output_img.min())  # Normalize to [0, 1]
    response_image = Image.fromarray((output_img * 255).astype(np.uint8))

    buffered = io.BytesIO()
    response_image.save(buffered, format="PNG")
    buffered.seek(0)

    return send_file(buffered, mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)
