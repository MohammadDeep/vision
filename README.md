# vision
## install package vision
Go to directory -> vision_project
run -> pip install -e.



## Downlaod dataset: 
  !wget http://images.cocodataset.org/zips/val2017.zip -P ./data/

  !unzip ./data/val2017.zip -d ./data

  !wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P ./data/

  !unzip ./data/annotations_trainval2017.zip -d ./data

  !wget http://images.cocodataset.org/zips/train2017.zip -P ./data/

  !unzip ./data/train2017.zip -d ./data/
  

## Create dataset folder for classefication :
- run 'Create_dataset.py'


## Folder structure
- models
  - Python files, each of which trains different models and stores the results and models.
    - Part 1: Specifying the Datasets ->"dic_name_folber_clases_val" and "dic_name_folber_clases"
    - Part 2: Creating 'transforms' and creating 'DataLoaders'
    - Part 3: Specifying the Model Structure
    - Part 4:
      - Specifying optimization and cost functions.
      - training the model, and testing it.
      - save model and history.