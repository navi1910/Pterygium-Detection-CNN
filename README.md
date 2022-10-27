
# Eye Pterygium detection Using CNN
CNN stands for Convolutional Neural networks. CNN is used when we have image data. This project was built using Google Colab and Tensorflow. Here we have used CNN for Image Classification. Classification is a type of supervised learning.

<img src="https://github.com/navi1910/Pterygium-Detection-CNN/blob/master/colab_logo.png" width=25% height=25%>

## Project Objective
The objective of the project is to build a CNN which can classify and differentiate between images of Normal Eye and Eye with Pterygium.

## Methods
- Deep Learning Neural Networks
- Visualizations
- zipfile
- Feature Engineering
- Data Scaling
- Saving the Model using os

## Technologies used
- Python
- Google Colab
- os
- Tensorflow
- Keras
- OpenCV
- Tensorboard
- Matplotlib
- `keras.utils.image_dataset_from_directory`

## Project Description
The project was built as a part of an Internship at Dr. Agarwal's Eye hospital, Tirunelveli.
The aim of the is to create a neural network model that can classify if the image is Eye with Pterygium Positive or Pterygium Negative.

<img src="<img src="https://github.com/navi1910/Pterygium-Detection-CNN/blob/master/colab_logo.png" width=25% height=25%>

Since Classificationis part of supervised learning, the model is provided with labelled dataset for training. These include images with and without Pterygium with different labels.
This project was built on Google Colab platform.

## Procedure
- The required python modules are imported.
- The zipfile with labelled images is unzipped.
- `tf.keras.utils.image_dataset_from_directory`, `data.as_numpy_iterator`, `data_iterator.next` from `keras` is used to import the images in the form of arrays and batches.
###### Note: Images need to be of jpeg or jpg format for this model to work.
- The batches are then scaled and visualized.

<img src='https://github.com/navi1910/Pterygium-Detection-CNN/blob/master/eye_batch.png' width=50% height=50%>

- labels 
    - `1` -> Pterygium Positive
    - `0` -> Pterygium Negative
- The batches are split into Training, Validation and Testing sets.

## Model Building
- The required modules are again imported.
- `Conv2D`, `MaxPooling2D`, `Flatten` and  `Dense` layers are  to `Sequential` from `keras`.
- `compile` is used to compile the Network with,
    - `'adam'` optimizer
    - `'BinaryCrossentropy'` from `keras` as loss
    - `['accuracy']` metrics

- The Neural Network is summarized using `summary`

<img src="https://github.com/navi1910/Pterygium-Detection-CNN/blob/master/model_summary.png" width=50% height=50%>

- Directory is created to store callbacks history for tensorboard.
- Training set is fit to the Model.
- The epoch_accuracy and epoch_loss is visualized using `Tensorboard`

<img src="https://github.com/navi1910/Pterygium-Detection-CNN/blob/master/epoch_accuracy.png" width=30% height=30%>

<img src="https://github.com/navi1910/Pterygium-Detection-CNN/blob/master/epoch_loss.png" width=30% height=30%>

- The model is evaluated using metrics such as `Precision`, `Recall` and `BinaryAccuracy`
    - Model Score 
        - `Precision` = 1.0
        - `Recall` = 1.0
        - `BinaryAccuracy` = 1.0

- A function is defined to predict if the image is Pterygium Positive or Pterygium Negative.
    - `yhat` > 0.5 indicates Pterygium Positive
    - `yhat` <= 0.5 indicates Pterygium Negative

### Pterygium Negative
<img src="https://github.com/navi1910/Pterygium-Detection-CNN/blob/master/pterygium_negative.png" width=50% height=50%>

## Pterygium Positive
<img src="https://github.com/navi1910/Pterygium-Detection-CNN/blob/master/pterygium_positive.png" width=50% height=50%>

## Saving the Model
The model is saved to a h5 file using model.save and python os into a directory called model. The model can be loaded using load_model.

## Contact
[Naveen's LinkedIn](https://www.linkedin.com/in/naveen-a-902a671b3/)

## Acknowledgements
[Nicholas Renotte](https://www.youtube.com/channel/UCHXa4OpASJEwrHrLeIzw7Yg)
