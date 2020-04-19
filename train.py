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
_, axes = plt.subplots(2, 2)
for i in range(4):
    x, _ = data[rand[i]]
    axes[i // 2][i % 2].imshow(x.numpy().transpose(1, 2, 0))
plt.show()

data = torch.utils.data.DataLoader(
    data, 
    batch_size, 
    shuffle=True
)

model = torch.hub.load(
    github='facebookresearch/pytorch_GAN_zoo:hub', 
    model='DCGAN', 
    pretrained=False, 
    useGPU=torch.cuda.is_available()
)

model.train()
for i in range(10):
    for j, (x, _) in enumerate(tqdm(data)):
        loss = model.optimizeParameters(x)

    z, _ = model.buildNoiseData(n_samples=1)
    plt.imshow(model.test(z)
                    .numpy().transpose(1, 2, 0))

torch.save(model.state_dict(), 'model.pt')

model.load_state_dict(torch.load('model.pt'))
model.eval()

z, _ = model.buildNoiseData(n_samples=1)
plt.imshow(model.test(z)
                .numpy().transpose(1, 2, 0))
