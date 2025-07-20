# vision
## install package vision
- Go to directory -> "vision/vision_project"

- run -> pip install -e.



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
