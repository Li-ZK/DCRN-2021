import numpy as np
import torch

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    if torch.cuda.device_count() > 1:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cuda')
else:
    device = torch.device('cpu')


def kappa_coef(cm):
    
    # API for calculating kappa coefficient, input is confusion matrix
    
    total_value = np.sum(cm)
    true_value = 0
    true_each_class = np.zeros(cm.shape[0])
    pred_each_class = np.zeros(cm.shape[1])
    for i in range(cm.shape[0]):
        true_each_class[i] = np.sum(cm[i, :])
        pred_each_class[i] = np.sum(cm[:, i])
        true_value += cm[i, i]
    p_o = true_value/total_value
    p_e = np.sum(true_each_class * pred_each_class)/ (total_value ** 2)
    kappa = (p_o - p_e)/(1 - p_e)
    return kappa

def find_max(preds):

    y_hat = torch.zeros(preds.shape[0]).to(device)

    for i in range(preds.shape[0]):
        maxval = torch.max(preds[i])
        for j in range(preds.shape[1]):
            if preds[i][j] == maxval:
                y_hat[i] = j
                break
    return y_hat
             

def acc_calculate(preds, labels, cm):
    y_hat = find_max(preds)
    equals = y_hat == labels
    acc = torch.mean(equals.type(torch.FloatTensor))
    for i in range(len(y_hat)):
        cm[labels[i].item(), int(y_hat[i].item())] += 1
    return acc

def evalution(dataloader, model, n_classes):
    model.eval()
    running_acc = 0
    cm = np.zeros((n_classes, n_classes))
    with torch.no_grad():
        for (info, fake_label, true_label) in dataloader:
            info = info.unsqueeze(1)
            info = info.to(device, torch.float32)
            true_label = true_label.to(device, torch.long)
            res = model(info)
            running_acc += acc_calculate(res, true_label, cm)
        aa = [cm[i, i]/ np.sum(cm[i, :]) for i in range(n_classes)]
        kappa = kappa_coef(cm)
    return running_acc/len(dataloader), aa, kappa