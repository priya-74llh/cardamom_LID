# Language Identification

## Install Dependencies

After create and activate venv, install dependencies.

```
pip install -r requirements.txt
```

## Data Format
    <text>  <lang_id>
Input Data Format :- The data needs to be in CSV format or text file where there will be two columns in following order - (1) Language Id (2) The Data. The columns should be tab seperated

## How To Train


In _**src/Vocab_char.py**_, function _**read_data**_ Variable - Labels value is to be changed.

```commandline
python train.py --batch_size 30 --lr 0.001 --num_epochs 3 --train_test train --supv_upnspv supv --no_of_classes 5 --train_file train.txt --test_file test_set.txt
```

## How To Infer

```commandline
python infer.py
```

[//]: # (### Sentence)

[//]: # (![demo_image]&#40;img.png&#41;)

