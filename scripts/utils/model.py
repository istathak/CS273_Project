import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torchvision.models as models
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.models import ColumnDataSource
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

output_notebook()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.BCELoss()

#  CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 28 * 28, 128) 
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)  
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Train and evaluate model
def train_cnn(train_loader, num_epochs=10, lr=0.001):
    
    model = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    accs = []
    losses=[]
    for epoch in range(num_epochs):
        # Training 
        model.train()
        train_loss, train_corrects = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_corrects += (predicted == labels).sum().item()

        train_loss /= len(train_loader)
        train_acc = train_corrects / len(train_loader.dataset) * 100
        accs.append(train_acc)
        losses.append(train_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    epochs = list(range(1, num_epochs + 1))
    train_source = ColumnDataSource(data={"epoch": epochs, "train_acc": accs, "train_loss": losses})
    p = figure(title="Training and Train Loss", x_axis_label="Epochs", y_axis_label="Accuracy")
    p.line("epoch", "train_acc", source=train_source, legend_label="Train Accuracy", color="blue", line_width=2)
    p.line("epoch", "train_loss", source=train_source, legend_label="Train Loss", color="red", line_width=2)
    show(p)
    print("Training complete!")

    return model
#todo: split the train accuracy/ train loss plot. 

def test_model(model,val_loader):
    # Evaluation 
    model.eval()
    val_loss, val_corrects = 0.0, 0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            predicted = (outputs > 0.5).float()

            val_corrects += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(outputs.cpu().numpy())

    val_acc = val_corrects / len(val_loader.dataset) * 100

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    precision = precision_score(all_labels, (all_preds > 0.5).astype(int))
    recall = recall_score(all_labels, (all_preds > 0.5).astype(int))
    f1 = f1_score(all_labels, (all_preds > 0.5).astype(int))
    roc_auc = roc_auc_score(all_labels, all_preds)

    print(f"Validation Accuracy: {val_acc:.2f}%")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, ROC AUC: {roc_auc:.4f}")


    return

#Confusion matrix
def plot_confusion_matrix(all_labels, all_preds):
    y_pred_binary = (all_preds > 0.5).astype(int)

    cm = confusion_matrix(all_labels, y_pred_binary)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Malignant"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix for CNN Model")
    plt.show()

    return

#  ROC curve
def plot_roc_curve(all_labels, all_preds):
    fpr, tpr, _ = roc_curve(all_labels, all_preds) 
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--") 
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.show()
    
    return

