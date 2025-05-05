# Music-Genre-Classification-with-xAI

This project focuses on classifying music genres using spectrogram images and CNN models. It includes interpretability through Grad-CAM visualizations.

## Files

* `genre_classifier_resnet18.pth`: Trained ResNet18 model (https://www.kaggle.com/models/mananbhutiani/resnet18-for-music-genre-classification)
* `VGGNET.pth`: Trained VGGNet model (https://www.kaggle.com/models/mananbhutiani/vggnet-for-music-genre-classification)
* `customcnn+gradcam.ipynb`: Jupyter Notebook for custom CNN, training, evaluation, and Grad-CAM visualization

## Features

* Classifies music genres using spectrograms
* Uses ResNet18, VGGNet, and a custom CNN
* Applies Grad-CAM to highlight important spectrogram regions
* Supports model evaluation and visualization

## Requirements

Install the following Python packages:

```
torch
torchvision
matplotlib
librosa
opencv-python
notebook
```

## Usage

1. Open the notebook `customcnn+gradcam.ipynb`
2. Load a spectrogram dataset (e.g., GTZAN converted to images)
3. Load or train models
4. Evaluate performance
5. Use Grad-CAM to visualize model attention

## Dataset

This project assumes usage of a dataset such as GTZAN, converted into Mel-spectrogram images.

## References

* Grad-CAM: [https://arxiv.org/abs/1610.02391](https://arxiv.org/abs/1610.02391)
* GTZAN Dataset: [http://marsyas.info/downloads/datasets.html](http://marsyas.info/downloads/datasets.html)
* PyTorch Docs: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)

## Author

Manan Bhutiani
Saraswata Ghosh
Ayush Ghosh



