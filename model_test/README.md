# test modeles

## analize model 

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