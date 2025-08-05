# test modeles

## analize model 

The first thing we should do after creating a model, before training it, is to check whether the model's architecture and its initial weights cause vanishing or exploding gradients. For that, we use the following function.

'''
from vision.hook import analis_model_with_weight_grads

analis_model_with_weight_grads(dic_model,
        '/content/drive/MyDrive/Model_4/model_epoch_0.pth',
        labels_batch,
        data_batch,
        loss_nf= nn.BCEWithLogitsLoss(),
        model = model_4_0
                 )

'''

This function plots the output of each layer and the gradients of its weights. This helps us understand whether the status of the gradients is good or not.


