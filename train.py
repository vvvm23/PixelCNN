import torch
import torch.nn as nn
import torchvision

import copy
import time
from tqdm import tqdm

from model import PixelCNN

TRY_CUDA = True
BATCH_SIZE = 256
NB_EPOCHS = 25
MODEL_SAVING = True

def get_device():
    if TRY_CUDA == False:
        return torch.device('cpu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

device = torch.device('cuda' if TRY_CUDA and torch.cuda.is_available() else 'cpu')
print(f"> Using device {device}")
print("> Instantiating PixelCNN")
model = PixelCNN((1,28,28), 16, 5, 256, 10).to(device)

print("> Loading dataset")
train_dataset = torchvision.datasets.KMNIST('data', train=True, download=True, transform=torchvision.transforms.ToTensor())
test_dataset = torchvision.datasets.KMNIST('data', train=False, download=True, transform=torchvision.transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

optim = torch.optim.Adam(model.parameters())
crit = nn.CrossEntropyLoss()

save_id = int(time.time())
best_loss = 999.
best_model = copy.deepcopy(model.state_dict())

for ei in range(NB_EPOCHS):
    print(f"\n> Epoch {ei+1}/{NB_EPOCHS}")
    train_loss = 0.0
    eval_loss = 0.0

    model.train()
    for x, h in tqdm(train_loader):
        optim.zero_grad()
        x, h = x.to(device), h.to(device)
        target = (x*255).long()

        pred = model(x, h)
        loss = crit(pred.view(BATCH_SIZE, 256, -1), target.view(BATCH_SIZE, -1))
        train_loss += loss.item()
        loss.backward()
        optim.step()

    model.eval()
    with torch.no_grad():
        for i, (x, h) in enumerate(tqdm(test_loader)):
            optim.zero_grad()
            x, h = x.to(device), h.to(device)
            target = (x[:,0]*255).long()

            pred = model(x, h)
            loss = crit(pred.view(BATCH_SIZE, 256, -1), target.view(BATCH_SIZE, -1))
            eval_loss += loss.item()

            if i == 0:
                img = torch.cat([x, torch.argmax(pred, dim=1).unsqueeze(1)], dim=0)
                torchvision.utils.save_image(img, f"samples/pixelcnn-{ei}.png")

    print(f"> Training Loss: {train_loss / len(train_loader)}")
    print(f"> Evaluation Loss: {eval_loss / len(test_loader)}")
    
    torch.save(model.state_dict(), f"models/{save_id}-{ei}-pixelcnn.pt")
    if eval_loss / len(test_loader) < best_loss:
        best_model = copy.deepcopy(model.state_dict())
        best_loss = eval_loss / len(test_loader)

torch.save(best_model, f"models/{save_id}-best-pixelcnn.pt")
