import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from models.vit import ViT  # Make sure to import the correct model class
from models.resnet import ResNet18

eval_model = "resnet18"
if eval_model == "vit":
    checkpoint_path = './checkpoint/vit-4-ckpt.t7'
    model = ViT(
    image_size = 32,
    patch_size = 4,
    num_classes = 10,
    dim = 512,
    depth = 6,
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1
    )
    log_dir = './log/log_vit_patch4.txt'
    nstart = 100
elif eval_model == "resnet18":
    checkpoint_path = './checkpoint/res18-4-ckpt.t7'
    model = ResNet18()
    log_dir = './log/log_res18_patch4.txt'
    nstart = 143

checkpoint = torch.load(checkpoint_path)

model.load_state_dict(checkpoint['model'])
model.eval()
model.cuda()

print('train_loss' in checkpoint and 'train_acc' in checkpoint)


# Load the test dataset
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
if  __name__ == '__main__':
    # Evaluate the model
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    accuracy = correct / total
    print(f'Accuracy of the model on the test images: {accuracy * 100:.2f}%')

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # Plot loss and accuracy curves (if you saved them during training)
    import matplotlib.pyplot as plt

    # Initialize lists to store the extracted data
    epochs = []
    learning_rates = []
    val_losses = []
    accuracies = []

    # Read the log file and extract the relevant data
    with open(log_dir, 'r') as file:
        for line in file:
            if "Epoch" in line:
                # Split the line into components
                components = line.split(',')
                # Extract the epoch number, learning rate, validation loss, and accuracy
                epoch = int(components[0].split(' ')[-1])
                lr = float(components[1].split(' ')[-1])
                val_loss = float(components[2].split(' ')[-1])
                acc = float(components[3].split(' ')[-1])
                # Append the extracted data to the respective lists
                epochs.append(epoch)
                learning_rates.append(lr)
                val_losses.append(val_loss)
                accuracies.append(acc)
    epochs = epochs[nstart:nstart+200]
    accuracies = accuracies[nstart:nstart+200]
    val_losses = val_losses[nstart:nstart+200]
    print(max(accuracies))
    # Plot the validation loss curve
    plt.figure(figsize=(10, 8))
    plt.plot(epochs, val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss Curve')
    plt.legend()

    # Plot the accuracy curve
    plt.figure(figsize=(10, 8))
    plt.plot(epochs, accuracies, label='Accuracy', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curve')
    plt.legend()

    plt.tight_layout()
    plt.show()