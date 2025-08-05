

# Model Analysis and Testing Utilities

This document outlines a set of utility functions for analyzing, evaluating, and better understanding the performance of deep learning models. These tools are divided into three main sections:

1.  **Pre-Training Gradient Analysis**: To ensure the initial health of the model.
2.  **Post-Training Visual Error Analysis**: To qualitatively understand model mistakes.
3.  **Quantitative Performance Evaluation**: To compare models using standard metrics.

-----

## 1\. Analyze Model Gradients (Pre-Training)

Before starting the time-consuming training process, it's crucial to inspect the model's initial state. This function helps diagnose potential issues like **Vanishing Gradients** or **Exploding Gradients** by analyzing the outputs and weight gradients of each layer.

### Usage

This function performs a single forward and backward pass on a data batch and then plots the distribution of activations and weight gradients for each layer.

```python
from vision.hook import analis_model_with_weight_grads

analis_model_with_weight_grads(
    dic_model,
    model_path='/content/drive/MyDrive/Model_4/model_epoch_0.pth',
    labels_batch=labels_batch,
    data_batch=data_batch,
    loss_fn=nn.BCEWithLogitsLoss(),
    model=model_4_0
)
```

### Parameters

  * `dic_model`: A dictionary containing the model's architecture information.
  * `model_path`: Path to the model file with initial weights (typically at epoch 0).
  * `labels_batch`: Labels corresponding to the data batch.
  * `data_batch`: A batch of input data.
  * `loss_fn`: The loss function used to calculate gradients (e.g., `$nn.BCEWithLogitsLoss()$`).
  * `model`: The model object to be analyzed.

-----

## 2\. Plot Random Images for Error Analysis

After a model is trained, a qualitative analysis of its errors is highly valuable. This function helps to understand the types of data on which the model makes the most mistakes. Its output includes the input image, its actual label, and the label predicted by the model.

### Usage

```python
from vision.modeles_test import plot_random_image

plot_random_image(
    dic_model,
    list_dataset_dir=[dir_extended_all_folber_val, dir_extended_cut_image_non_core_class_val],
    model_save_dir='/content/drive/MyDrive/Model_4',
    extension='.pth',
    text_to_find='model_epoch_205',
    num_samples=100,
    figsize=(4, 4),
    show_images='all',  # Options: 'all', 'TT', 'FF', 'FT', 'TF', 'all T', 'all F'
    threshold=0.5
)
```

### Parameters

  * `dic_model`: A dictionary containing the model's architecture information.
  * `list_dataset_dir`: A list of paths to the datasets for sampling.
  * `model_save_dir`: The directory where saved models are located.
  * `extension`: The file extension for model files (e.g., `.pth`).
  * `text_to_find`: A substring in the filename to locate the specific model version (e.g., `model_epoch_205`).
  * `num_samples`: The number of random images to display.
  * `figsize`: The size of each plotted image.
  * `show_images`: A filter to display images based on prediction correctness.
      * `'all'`: Show all samples.
      * `'TT'`: (True-True) Show only correct positive predictions.
      * `'FF'`: (False-False) Show only correct negative predictions.
      * `'FT'`: (False-True) Show only Type I errors (False Positives).
      * `'TF'`: (True-False) Show only Type II errors (False Negatives).
      * `'all T'`: Show all correct predictions (both TT and FF).
      * `'all F'`: Show all incorrect predictions (both FT and TF).
  * `threshold`: The decision threshold for converting model outputs to binary classes (typically $0.5$).

-----

## 3\. Quantitative Model Evaluation

This function is designed for the quantitative and comparative evaluation of models on a validation dataset. It returns a pandas DataFrame of the results, making it easy to compare different models or training epochs.

### Usage

```python
from vision.modeles_test import test_model

result_df = test_model(
    dic_model,
    list_dataset_dir=[dir_dataset_orgnal_val, dir_extended_all_folber_val, dir_extended_cut_image_non_core_class_val],
    model_loss_function=nn.BCEWithLogitsLoss(),
    model_save_dir='/content/drive/MyDrive/Model_4',
    BATCH_SIZE=2**7,
    extension='.pth',
    text_to_find='model_epoch_205',
    threshold=0.5
)
```

### Parameters

  * `dic_model`: A dictionary containing the model's architecture information.
  * `list_dataset_dir`: A list of paths to the validation datasets for evaluation.
  * `model_loss_function`: The loss function to calculate loss during evaluation.
  * `model_save_dir`: The directory where saved models are located.
  * `BATCH_SIZE`: The batch size to use for evaluation.
  * `extension`: The file extension for model files.
  * `text_to_find`: A substring in the filename to locate the specific model.
  * `threshold`: The decision threshold for classification.

### Output

The function returns a `pandas.DataFrame` containing evaluation metrics (like accuracy, F1-score, loss, etc.) for the model on the specified datasets.