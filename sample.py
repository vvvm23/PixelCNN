import torch
import torch.nn.functional as F
import torchvision

import sys
import time
from tqdm import tqdm
from model import PixelCNN

TRY_CUDA = True
IMAGE_DIM = (1,28,28)
NB_SAMPLES = 10
NB_CLASSES = 10

def get_device():
    if TRY_CUDA == False:
        return torch.device('cpu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

device = torch.device('cuda' if TRY_CUDA and torch.cuda.is_available() else 'cpu')
print(f"> Using device {device}")

try:
    print(f"> Loading PixelCNN from file {sys.argv[1]}")
    model = PixelCNN(IMAGE_DIM, 16, 5, 256, 10).to(device)
    model.load_state_dict(torch.load(sys.argv[1]))
    model.eval()
    print("> Loaded PixelCNN succesfully!")
except:
    print("! Failed to load state dict!")
    print("! Make sure model is of correct size and path is correct!")
    exit()

with torch.no_grad():
    sample = torch.zeros(NB_SAMPLES*NB_CLASSES, *IMAGE_DIM).to(device)
    cond = torch.tensor([d for d in range(NB_CLASSES) for _ in range(NB_SAMPLES)]).to(device)

    pb = tqdm(total=IMAGE_DIM[0]*IMAGE_DIM[1]*IMAGE_DIM[2])

    for c in range(IMAGE_DIM[0]):
        for i in range(IMAGE_DIM[1]):
            for j in range(IMAGE_DIM[2]):
                pred = model(sample, cond).to(device)
                pred = F.softmax(pred[:, :, c, i, j], dim=1)
                sample[:, c, i, j] = torch.multinomial(pred, 1).float().squeeze() / 255.0
                pb.update(1)
                
    save_id = int(time.time())
    torchvision.utils.save_image(sample, f"samples/zero-{save_id}.png", nrow=NB_CLASSES)
