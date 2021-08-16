import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np
import os,random,sys,time
from torch.utils.data import DataLoader
from Libraries.extract_patches import get_data_train
from Libraries.dataset import TrainDataset,TestDataset
from Libraries.common import count_parameters, AverageMeter
from Libraries.losses import *
from Model_Architecture import UNET_Family
from config import parse_args
from collections import OrderedDict
from Libraries.logger import Logger, Print_Logger
from Libraries.metrics import Evaluate
from tqdm import tqdm

def get_dataloader(args):
    image_patches, mask_patches = get_data_train(args.df_train_path, args.train_patch_height, 
                                                 args.train_patch_width, args.stride_height, 
                                                 args.stride_width, args)
#     print("Image Dimension: {}. Mask Dimension: {}".format(image_patches.shape, mask_patches.shape))
    val_ind = random.sample(range(mask_patches.shape[0]), 
                            int(np.floor(args.val_ratio*mask_patches.shape[0])))
    train_ind =  set(range(mask_patches.shape[0])) - set(val_ind)
    train_ind = list(train_ind)
    train_set = TrainDataset(image_patches[train_ind,...], 
                             mask_patches[train_ind,...], mode="train")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    val_set = TrainDataset(image_patches[val_ind,...], 
                           mask_patches[val_ind,...], mode="val")
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader

def get_dataloader_overfitt(args):
    image_patches, mask_patches = get_data_train(args.df_train_path, args.train_patch_height, 
                                                 args.train_patch_width, args.stride_height, 
                                                 args.stride_width, args)
#     print("Image Dimension: {}. Mask Dimension: {}".format(image_patches.shape, mask_patches.shape))
    val_ind = random.sample(range(mask_patches.shape[0]), 
                            int(np.floor(args.val_ratio*mask_patches.shape[0])))
    train_ind =  set(range(mask_patches.shape[0])) - set(val_ind)
    train_ind = list(train_ind)
    train_set = TrainDataset(image_patches[train_ind,...], 
                             mask_patches[train_ind,...], mode="val")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    val_loader = train_loader
    
    return train_loader, val_loader

# train 
def train(train_loader,net,criterion,optimizer,device):
    net.train()
    train_loss = AverageMeter()
    for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader), total=len(train_loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), inputs.size(0))
    log = OrderedDict([('train_loss',train_loss.avg)])
    
    return log

def DICE_Score(prediction, GT):
    dice = np.sum(2.0 * prediction * GT) / (np.sum(prediction) + np.sum(GT))
    return dice

# val 
def val(val_loader,net,criterion,device):
    net.eval()
    val_loss = AverageMeter()
    evaluater = Evaluate()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(val_loader), total=len(val_loader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            val_loss.update(loss.item(), inputs.size(0))

            outputs = outputs.data.cpu().numpy()
            targets = targets.data.cpu().numpy()
            evaluater.add_batch(targets,outputs[:,1])
    log = OrderedDict([('val_loss', val_loss.avg), 
                       ('val_acc', evaluater.confusion_matrix()[1]), 
                       ('val_f1', evaluater.f1_score()),
                       ('val_auc_roc', evaluater.auc_roc())])
    return log

def main():
    args = parse_args()
    save_path = os.path.join(args.outf, args.save)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    cudnn.benchmark = True
    log = Logger(save_path)
    sys.stdout = Print_Logger(os.path.join(save_path,'train_log.txt'))
    print('The computing device used is: ','GPU' if device.type=='cuda' else 'CPU')
    # Load Network
    net = UNET_Family.U_Net(args.in_channels, args.classes).to(device)
    #net = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=1, out_channels=2, init_features=32, pretrained=False).to(device)
    print("Total number of parameters: " + str(count_parameters(net)))
    # Save the model structure to the tensorboard file
    log.save_graph(net,torch.randn((1,1,128,128)).to(device).to(device=device))
    # Initialize Loss Function
    criterion = CrossEntropyLoss2d()
    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.N_epochs, eta_min=0)
    # Create dataloaders
    train_loader, val_loader = get_dataloader(args)
#     train_loader, val_loader = get_dataloader_overfitt(args)
    print(len(train_loader), len(val_loader))
    # Initialize the best epoch and performance(AUC of ROC)
    #best = {'epoch':0,'AUC_roc':0.5}
    best = {'epoch':0,'f1':0.1}
    trigger = 0  # Early stop Counter
    
    for epoch in range(args.N_epochs):
        print('\nEPOCH: %d/%d --(learn_rate:%.6f) | Time: %s' % \
              (epoch, args.N_epochs,optimizer.state_dict()['param_groups'][0]['lr'], time.asctime()))
        # Training
        train_log = train(train_loader,net,criterion, optimizer,device)
        # Validation
        val_log = val(val_loader,net,criterion,device)
        # Add log information
        log.update(epoch,train_log,val_log) 
        lr_scheduler.step()
        # Save checkpoint of latest and best model.
        state = {'net': net.state_dict(),'optimizer':optimizer.state_dict(),'epoch': epoch}
        torch.save(state, os.path.join(save_path, 'latest_model.pth'))
        trigger += 1
        if val_log['val_f1'] > best['f1']:
            print('\033[0;33mSaving best model!\033[0m')
            torch.save(state, os.path.join(save_path, 'best_model.pth'))
            best['epoch'] = epoch
            best['f1'] = val_log['val_f1']
            trigger = 0
        print('Best performance at Epoch: {} | F1_Score: {}'.format(best['epoch'],best['f1']))
        # early stopping
#         if not args.early_stop is None:
#             if trigger >= args.early_stop:
#                 print("=> early stopping")
#                 break
                
        torch.cuda.empty_cache()
        
    
if __name__ == '__main__':
    main()

