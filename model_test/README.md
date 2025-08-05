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


## plot random image


After training, you can use the following function to analyze the error. Its output consists of the images, their actual labels, and the labels predicted by the model. This helps us understand on which data the model is making the most mistakes.


'''

from vision.modeles_test import plot_random_image


plot_random_image(
        dic_model,
        list_dataset_dir = [dir_extended_all_folber_val , dir_extended_cut_image_non_core_class_val ],
        model_save_dir = '/content/drive/MyDrive/Model_4' ,

        extension = '.pth',
        text_to_find = 'model_epoch_205',
        num_samples = 100 ,
        figsize=(4, 4),
        show_images = 'all' # 'all' or 'TT' or 'FF' or 'FT' or 'TF' or 'all T' or 'all F'
        ,threshold = 0.5
        )

'''


## test accuracy for model


The following function evaluates the models using the validation dataset, saves their results as a dataframe, and returns it. This is useful for comparing the models with one another.


'''

from vision.modeles_test import test_model


resutl = test_model(
        dic_model,
        list_dataset_dir = [dir_dataset_orgnal_val,dir_extended_all_folber_val , dir_extended_cut_image_non_core_class_val ],
        model_loss_funciont = nn.BCEWithLogitsLoss(),
        model_save_dir = '/content/drive/MyDrive/Model_4' ,
        BATCH_SIZE = 2 ** 7,
        extension = '.pth',
        text_to_find = 'model_epoch_205',
        threshold = 0.5
         )


'''