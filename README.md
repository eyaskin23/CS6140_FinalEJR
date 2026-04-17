
# Musical Note Classification
## CS6140 Final Project Spring 2026

Robert Zando, Evelyn Yaskin, Jason Ingersoll 

Instructions for this project:
1. Download and extract data.zip from google drive link:

https://drive.google.com/file/d/1dEo3elgQpm3H_aX-DV4g7S35T5RUqQ_s/view?usp=sharing

Save it into folder "data"

2. Running each method:

CNN:
```
cd CNN
python3 CNN_implementation.py
```

FFN:
```
cd SVM
python3 FNN_implementation.py
```

SVM:
```
cd SVM
python3 svm_implementation.py
```

NOTE: 
In SVM_implementation.py, FNN_implementation.py, and CNN_implementation.py:
```
input_array_train = NoteDataset(r"../data", split="train")
```
You will need to change this line between forward slash/back slash depending on which device you are on. 
(Forward slash for mac, back slash for windows) 


