
## pour commencer

1. Construire l'extension cython en place:
```python setup.py build_ext --inplace```
2. Rununer  detect.py 
```python detect.py```

## pour entrainer le model:

## telecharger data set et la mettre dans le fichier courant  , lien de data set : https://www.kaggle.com/datasets/draaslan/blood-cell-detection-dataset

## training
$ python flow --model cfg/tiny-yolo-voc-3c.cfg --load bin/tiny-yolo-voc.weights --train --gpu .7 --annotation dataset/Training/Annotations --dataset dataset/Training/Images --lr 1e-3 --epoch 100

## lancement de projet realiser avec tkinter:
 python application.py