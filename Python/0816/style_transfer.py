import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image

# device 
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

# output size
img_size = 512 if torch.cuda.is_available() else 128

loader_tranforms = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor()
])

def image_loader(image_name) :
    image = Image.open(image_name)
    image = loader_tranforms(image).unsqueeze(0)
    return image.to(device, torch.float)


# style image path, content image path
style_image_path = "./style.jpg"
content_image_path = "./cat.png"
style_image = image_loader(style_image_path)
content_image = image_loader(content_image_path)

unloader = transforms.ToPILImage()

def imshow(tensor, title=None) :
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    if title is not None : 
        plt.title(title)
    plt.pause(0.001)

plt.figure()
imshow(style_image, title="Style Image")

plt.figure()
imshow(content_image, title="Content Image")
plt.show()

cnn = models.vgg19(pretrained=True).features.to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

class Normalization(nn.Module) :
    def __init__(self, mean, std) :
        super(Normalization, self).__init__() 
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)
    
    def forward(self, img) :
        return (img - self.mean) / self.std

content_layers_default = ['conv_4']
style_layers_default = ['conv_1','conv_2','conv_3','conv_4','conv_5']
