# Traffic Sign Classification using CNN

This project endeavors to create an automated classifier for traffic signs, utilizing deep learning techniques. The focus lies in the recognition and categorization of diverse traffic sign types. The primary aim is to establish a precise and efficient classification system capable of identifying sign types from images.

## Dataset

The dataset contains 43 classes of traffic signs, each represented by labeled images. It includes various speed limits, no-entry signs, pedestrian crossings, and more.

## Model Training

### Data Preparation:

- Images are loaded and resized to 30x30 pixels.
- Images are categorized based on their class labels.

### Data Splitting:

- The dataset is split into training and testing sets.

### CNN Model Architecture:

- A CNN model is constructed using Convolutional, MaxPooling, Flatten, Dense, and Dropout layers.
- The model is compiled with categorical cross-entropy loss and the Adam optimizer.

### Model Training:

- The model is trained on the training set for 10 epochs.
- Training and validation accuracy/loss are monitored.

### Training Visualization:

- Graphs are plotted to visualize training and validation accuracy/loss.

### Model Evaluation:

- The trained model is evaluated on the test dataset.
- Accuracy and classification report are displayed.

### Model Saving:

- The trained model is saved as 'traffic_classifier.h5' for future use.

## Usage

### Train the Model

1. Clone the repository:

    ```bash
    git clone https://github.com/Razib91lightspeed/Traffic-Sign-Classification-using-CNN.git
    ```

2. Navigate to the project directory:

    ```bash
    cd Traffic-Sign-Classification
    ```

3. Run the Jupyter Notebook `TrafficSignClassification.ipynb` to train the model.

### Test the Model

- After training, the model is saved as 'traffic_classifier.h5'.

- To run the GUI for testing:

    ```bash
    python traffic_sign_classifier_gui.py
    ```

- Upload an image using the GUI and click "Classify Image" to get the predicted traffic sign class.

Note: Ensure that the required libraries are installed. You can install them using:

```bash
pip install -r requirements.txt
```

## Contributors

- Razib Hasan
- Huang Xiaosi

## License

This project is licensed under the MIT License.
