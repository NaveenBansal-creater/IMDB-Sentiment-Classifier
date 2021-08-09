#import config
import joblib
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

import pdb
import datetime
from torch.optim.lr_scheduler import StepLR

def  train(tb,epochs,model,train_loader,val_loader,batch_size,optimizer,criterion,device):
    
    best_val_loss=100000000

    for e in range(epochs):
            
            ###################
            # Training Phase
            ###################
            model.train()
            train_losses = []
            train_accuracy=[]
            for batch_indx,data in enumerate(train_loader):
                
               
                # Reset all gradients
                optimizer.zero_grad()

                # Start with a zero state value for each new batch
                hidden = model.zero_state(batch_size)
                
                inputs, labels = data['x'].to(device), data['y'].to(device)
                out = model.forward(inputs, hidden)
                
                loss = criterion(out,labels.flatten())
                loss.backward()
                train_losses.append(loss.item())
                
                y_pred = torch.argmax(out,dim=1)
                accuracy = (y_pred==labels.long().squeeze()).sum().item()/y_pred.shape[0]
                train_accuracy.append(accuracy)
                
                # LET"S CLIP JUST IN CASE
                nn.utils.clip_grad_norm_(model.parameters(),max_norm=5)
                
                optimizer.step()
                
            tb.add_scalar("Train Loss per epoch", loss.item(), e)
            tb.add_scalar("Train Accuracy per epoch", accuracy, e)
                
                
                
                
                
            ########################
            # validation phase   
            ########################
            model.eval()
            
            val_losses = []
            val_accuracy=[]
            
            for batch_indx,data in enumerate(val_loader):

                hidden = model.zero_state(batch_size)

                inputs, labels = data['x'].to(device), data['y'].to(device)

                out = model.forward(inputs, hidden)

                loss = criterion(out,labels.flatten())
                val_losses.append(loss.item())
                
                y_pred = torch.argmax(out,dim=1)
                accuracy = (y_pred==labels.long().squeeze()).sum().item()/y_pred.shape[0]
                val_accuracy.append(accuracy)
                
            tb.add_scalar("Val Loss per epoch", loss.item(), e)
            tb.add_scalar("Val Accuracy per epoch", accuracy, e)    
                

            print(f"{datetime.datetime.now().time()}:Epoch:{e} {np.mean(train_losses)}" \
                  f" {np.mean(train_accuracy)} {np.mean(val_losses)} {np.mean(val_accuracy)}")
            
            if np.mean(val_losses) < best_val_loss:
                best_val_loss = np.mean(val_losses)
                model_name = f"epoch_{e}_train_loss_{np.mean(train_losses)}_val_loss_{best_val_loss}.model"
                torch.save(model.state_dict(),model_name)
            
    tb.close()