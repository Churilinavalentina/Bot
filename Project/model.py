import torch.nn as nn
from torchmetrics.classification import BinaryAUROC

class MyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=512):
        super().__init__()
        # Линейный слой: принимает input_dim признаков, выдает 1 число
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        # Сигмоида превращает любое число в вероятность от 0 до 1
        self.sigmoid = nn.Sigmoid()
        self.silu = nn.SiLU()
        self.bn = nn.BatchNorm1d(num_features=input_dim)

    def forward(self, x):
        x = self.bn(x)
        x = self.silu(self.linear1(x))
        x = self.silu(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x

def calc_metrics(preds, true, threshold=0.5):
    auc = BinaryAUROC()
    plabels = (preds > threshold).float()

    tp = (plabels * true).sum()
    tn = ((1-plabels) * (1-true)).sum()
    fp = (plabels * (1 - true)).sum()
    fn = ((1-plabels) * true).sum()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)

    print(f"tp: {tp:.2f} /tn: {tn:.2f} / fp: {fp:.2f} / fn: {fn:.2f}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"ROC-AUC: {auc(preds, true):.2f}")