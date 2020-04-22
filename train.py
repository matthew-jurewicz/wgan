from torchvision import datasets, transforms
import random
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

img_size = 64
batch_size = 16

data = datasets.ImageFolder(
    root='./data', 
    transform=transforms.Compose([
        transforms.RandomHorizontalFlip(), 
        transforms.Resize(size=(img_size,img_size)), 
        transforms.ToTensor()
    ])
)

rand = random.sample(range(len(data)), k=4)
_, ax = plt.subplots(2, 2)
for i in range(4):
    x, _ = data[rand[i]]
    ax[i // 2][i % 2].imshow(x.permute(1, 2, 0))
plt.show()

data = torch.utils.data.DataLoader(
    data, 
    batch_size, 
    shuffle=True
)

model = torch.hub.load(
    github='facebookresearch/pytorch_GAN_zoo:master', 
    model='DCGAN', 
    pretrained=False, 
    useGPU=torch.cuda.is_available(), 
    dimG=256, 
    dimD=256, 
    lambdaGP=10
)

fig, ax = plt.subplots(1, 1)
for i in range(10):
    for j, (x, _) in enumerate(tqdm(data)):
        loss = model.optimizeParameters(x)

    z, _ = model.buildNoiseData(n_samples=1)
    out = 255 * (model.test(z)[0] + 1) / 2
    ax.imshow(out.type(torch.uint8).permute(1, 2, 0))
    fig.canvas.draw()

model.save('model.pt')

model.load('model.pt')

z, _ = model.buildNoiseData(n_samples=1)
out = 255 * (model.test(z)[0] + 1) / 2
plt.imshow(out.type(torch.uint8).permute(1, 2, 0))
plt.show()
