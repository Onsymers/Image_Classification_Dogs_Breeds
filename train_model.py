#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import smdebug.pytorch as smd
import torchvision.models as models
import torchvision.transforms as transforms
import argparse
import logging
import os
import sys

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

#TODO: Import dependencies for Debugging andd Profiling
from sagemaker.debugger import Rule, ProfilerRule, rule_configs


def test(model, test_loader, criterion, device, epoch_no, hook):
    logger.info(f"Epoch: {epoch_no} - Testing Model" )
    
    model.eval()
    hook.set_mode(smd.modes.EVAL) 
    running_loss = 0
    corrects = 0
    
    with torch.no_grad(): 
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            pred = outputs.argmax(dim=1, keepdim=True)
            running_loss += loss.item() * inputs.size(0) 
            corrects += pred.eq(labels.view_as(pred)).sum().item() 
    
        total_loss = running_loss / len(test_loader.dataset)
        total_acc = corrects/ len(test_loader.dataset)
        logger.info(
            f"\nTest set: Average loss: {total_loss}, Accuracy: {corrects}/{len(test_loader.dataset)}({total_acc*100}%)\n"
        )

def train(model, train_loader, criterion, optimizer, device, epoch_no, hook):
    
    logger.info(f"Epoch: {epoch_no} - Training Model" )
    model.train()
    
    hook.set_mode(smd.modes.TRAIN)
    
    running_loss = 0
    corrects = 0
    
    for batch_idx ,(inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * inputs.size(0) 
        pred = outputs.argmax(dim=1,  keepdim=True)
        
        corrects += pred.eq(labels.view_as(pred)).sum().item() #calculate the running corrects
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            logger.info(
                "Train Epoch: {} Accuracy: [{}/{} ({:.0f}%)]\tLoss: {:.4f}/t".format(
                    epoch_no,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    (batch_idx / len(train_loader)) * 100,
                    loss.item()
                )
            )
    return model
    
def net():
    model = models.resnet50(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad = False
        
    model.fc = nn.Sequential(nn.Linear(num_features, 256), #Adding our own fully connected layers
                             nn.ReLU(inplace = True),
                             nn.Linear(256, 156),
                             nn.ReLU(inplace = True),
                             nn.Linear(156, 133), # 133 DOG BREEDS
                            )
    return model

def create_data_loaders(data_path, batch_size):
    
    train_dataset_path = os.path.join(data_path, "train")
    test_dataset_path = os.path.join(data_path, "test")
    
    training_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize(256),
        transforms.RandomResizedCrop((224, 224)),
        transforms.ToTensor()]
    )
    
    testing_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop((224, 224)),
        transforms.ToTensor()]
    )
    
    train_dataset = torchvision.datasets.ImageFolder(root=train_dataset_path, transform=training_transform)    
    test_dataset = torchvision.datasets.ImageFolder(root=test_dataset_path, transform=testing_transform)
    
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size )
    
    return train_data_loader, test_data_loader

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on Device {device}")
    logger.info(
        f"Hyperparameters : LR: {args.lr},  Eps: {args.eps}, Weight-decay: {args.weight_decay}, Batch Size:{args.batch_size}, Epoch: {args.epochs}"
    )
    
    logger.info(f"Data Dir Path: {args.data_dir}")
    logger.info(f"Model Dir  Path: {args.model_dir}")
    logger.info(f"Output Dir  Path: {args.output_dir}")
    
    model=net()
    
    model = model.to(device)
    
    hook = smd.Hook.create_from_json_file()
    hook.register_module(model)
    
    train_data_loader, test_data_loader = create_data_loaders(args.data_dir, args.batch_size )
    
    loss_criterion = nn.CrossEntropyLoss()
    hook.register_loss(loss_criterion)
    optimizer = optim.AdamW(model.fc.parameters(), lr=args.lr, eps= args.eps, weight_decay = args.weight_decay)
    
    for epoch_no in range(1, args.epochs +1 ):
        logger.info(f"Epoch {epoch_no} - Starting Training phase.")
        model=train(model, train_data_loader, loss_criterion, optimizer, device, epoch_no, hook)
        logger.info(f"Epoch {epoch_no} - Starting Testing phase.")
        test(model, test_data_loader, loss_criterion, device, epoch_no, hook)
    
    
    
    
    torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pth'))
    logger.info(f"Model is saved at {path}")

if __name__=='__main__':
    
    parser=argparse.ArgumentParser()
    parser.add_argument(  "--batch_size", type = int, default = 64, metavar = "N", help = "input batch size for training (default: 64)" )
    parser.add_argument( "--epochs", type=int, default=2, metavar="N", help="number of epochs to train (default: 2)"    )
    parser.add_argument( "--lr", type = float, default = 0.1, metavar = "LR", help = "learning rate (default: 1.0)" )
    parser.add_argument( "--eps", type=float, default=1e-8, metavar="EPS", help="eps (default: 1e-8)" )
    parser.add_argument( "--weight_decay", type=float, default=1e-2, metavar="WEIGHT-DECAY", help="weight decay coefficient (default 1e-2)" )
                        
    parser.add_argument('--data_dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    
    args=parser.parse_args()
    
    main(args)
