# Sperm Morphology Image Classification

This project focuses on the classification of human sperm morphology using a Convolutional Neural Network (CNN) built with PyTorch. The pipeline includes data loading, preprocessing, model training, evaluation, and visualization.

## Project Structure

```
Sperm-Morphology-Image-Classification/
├── data/                           # Dataset containing input images
├── images/                         # Images used in README.md file 
├── notebooks/                      # Research & development notebooks and files they generate
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_model_evaluation.ipynb
├── models/                         # Models trained using notebooks and Python scripts
├── scripts/                        # Standalone Python scripts
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── grid_search.py
├── .gitignore
├── requirements.txt                # List of used packages
└── README.md                       # Project documentation
```

## Dataset

The dataset comprises labeled images of sperm cells categorized based on morphological quality. The data is organized into normal sperm, abnormal sperm and non sperm sets, each containing images in bmp format.

**Dataset Structure:**

This project uses the [Sperm Morphology Image Data Set (SMIDS)](https://www.kaggle.com/datasets/orvile/sperm-morphology-image-data-set-smids) from Kaggle.

- The dataset contains labeled images for classification of different states of sperm cells.
- It hasn't been split into training, validation, and test sets.
- Images are labelled by being in class corresponding folders and by having class name in their filename.

To use this dataset:
1. Download it from Kaggle.
2. Unzip the files and place `Abnormal_Sperm/`, `Non_Sperm/` and `Normal_Sperm/` in `data/` folder.
3. The folder structure inside `data/` should match the following requirements:

```
data/
├── Abnormal_Sperm/
├── Non_Sperm/
└── Normal_Sperm/
```

Each folder contains bmp file named in the following format `Class_name (image number).bmp` for example `Normal_Sperm (1).bmp`.

## Data Exploration Summary

The exploratory data analysis (see `notebooks/data_exploration.ipynb`) revealed the following key insights about the dataset:

- **Classes Distribution**: The dataset includes three sperm morphology classes, each containing approximately 1,000 images. The distribution is balanced, which reduces the need for resampling techniques or class weighting during training.
- **Image Resolutions**: There is significant variability in image resolution across the dataset, with 1,914 unique resolutions identified. This irregularity could potentially affect model performance if not addressed through proper resizing or normalization. The five most common resolutions are visualized in the figure below.
- **Dataset Quality**: The dataset appears clean and well-labeled. No major issues were identified during inspection, aside from the resolution variance mentioned above.
- **Visual Inspection**: Sample images were reviewed to validate the class labels and confirm that they represent the intended morphology categories. The images are of good quality and suitable for classification tasks.

These insights were used to inform the training strategy and later model evaluation.

![class_distribution](https://github.com/KacperTurek/Sperm-Morphology-Image-Classification/blob/main/images/class_distribution.png?raw=true)

![top_5_resolutions](https://github.com/KacperTurek/Sperm-Morphology-Image-Classification/blob/main/images/top_5_resolutions.png?raw=true)

![sample_images](https://github.com/KacperTurek/Sperm-Morphology-Image-Classification/blob/main/images/sample_images.png?raw=true)

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/KacperTurek/Sperm-Morphology-Image-Classification.git
cd Sperm-Morphology-Image-Classification
```

2. **Create a virtual environment (optional but recommended):**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```


## Usage

### 1. Exploratory Data Analysis (EDA)
- Navigate to `notebooks/data_exploration.ipynb`
- Load and visualize the dataset
- Review class distribution, sample images, and labels

### 2. Initial Model Training
- Open `notebooks/model_training.ipynb`
- Train a multiple CNN models using grid search selection of the hyperparameters
- Track training loss, F1, and other metrics

### 3. Model Evaluation
- Use `notebooks/model_evaluation.ipynb`
- Evaluate the best model on the test set
- Analyze metrics like accuracy, precision, recall, confusion matrix
- Visualize predictions and failure cases

## Python Scripts for Automation

### Train a Model
```bash
python scripts/train.py \
    --data data \
    --epochs 50 \
    --patience 5 \
    --batch 32 \
    --lr 0.0001 \
    --optimizer Adam \
    --dropout_rate 0.5 \
    --weight_decay 0.1 \
    --model_path models/train_script
```
**Defaults (picked based on the results of grid search):**
- `--data`: data
- `--epochs`: 50
- `--patience`: 10
- `--batch`: 32
- `--lr`: 0.0001
- `--optimizer`: Adam
- `--dropout_rate`: 0.35
- `--weight_decay`: 0
- `--model_path`: models/train_script

### Evaluate a Trained Model
```bash
python scripts/evaluate.py \
    --dataset models/train_script/2025-05-19_12:15_test_dataset.csv \
    --batch 32 \
    --dropout_rate 0.5 \
    --model_path models/train_script \
    --model_name: 2025-05-19_12:15.pt
```
**Defaults:**
- `--dataset`: models/train_script/2025-05-19_12:15_test_dataset.csv
- `--batch`: 32
- `--dropout_rate`: 0.35
- `--model_path`: models/train_script
- `--model_name`: 2025-05-19_12:15.pt

### Predict on New Images or Video
```bash
python scripts/predict.py \
    --image_path data/Normal_Sperm/Normal_Sperm (609).bmp \
    --model_path models/train_script \
    --weights 2025-05-19_12:15.pt \
    --results_path results.csv \
```
**Defaults:**
- `--image_path`: data/Normal_Sperm/Normal_Sperm (609).bmp
- `--model_path`: models/train_script
- `--weights`: 2025-05-19_12:15.pt
- `--results_path`: results.csv

### Perform Hyperparameter Grid Search
```bash
python scripts/grid_search.py \
    --data data \
    --epochs 50 \
    --patience 5 \
    --batch 32 \
    --model_path models/grid_search_script \
```
**Defaults:**
- `--data`: data
- `--epochs`: 50
- `--patience`: 10
- `--batch`: 32
- `--model_path`: models/grid_search_script


## Grid Search Results Summary

The `grid_search.py` script was used to explore multiple combinations of CNN hyperparameters such as learning rate, dropout rate, weight decay, and optimizer type. Each configuration was evaluated using validation accuracy.

- The results were logged in grid_search_results.csv and improved_grid_search_results.csv, with each row representing a distinct parameter set and its corresponding validation accuracy.
- The initial grid search was applied to a relatively simple CNN architecture. It systematically varied four key hyperparameters: learning rate, weight decay, optimizer (Adam or SGD), and dropout rate in the fully connected layers. This broad search allowed for identification of well-performing parameter regions, especially favoring Adam with moderate dropout and learning rates around 0.001.
- Based on the insights gained, a second, improved grid search was conducted on a more extensive CNN architecture—one that incorporated additional convolutional and fully connected layers for deeper feature learning. In this round, the number of hyperparameters was intentionally reduced to focus only on learning rate and dropout rate, while fixing the optimizer to Adam and using top-performing values from the first search (e.g., disabling weight decay).
- The improved model not only simplified the search space but also achieved better validation accuracy, indicating that architectural changes and targeted tuning both contributed to superior performance.

This two-stage grid search strategy—first for broad exploration and then for focused fine-tuning on a deeper model—was critical in developing a robust classifier that significantly outperformed the baseline configuration.

### Initial model grid search results

|   learning_rate |   weight_decay | optimizer   |   dropout_fc_rate | run_name      |   valid_acc |
|----------------:|---------------:|:------------|------------------:|:--------------|------------:|
|          0.001  |         0      | Adam        |               0.4 | grid_trial_0  |     85.1852 |
|          0.001  |         0.0001 | Adam        |               0.4 | grid_trial_4  |     85.1852 |
|          0.0003 |         0.0001 | Adam        |               0.5 | grid_trial_13 |     84.8148 |
|          0.001  |         0      | Adam        |               0.5 | grid_trial_1  |     84.4444 |
|          0.0003 |         0      | Adam        |               0.5 | grid_trial_9  |     84.0741 |
|          0.0003 |         0      | Adam        |               0.4 | grid_trial_8  |     82.963  |
|          0.0003 |         0.0001 | Adam        |               0.4 | grid_trial_12 |     82.963  |
|          0.001  |         0.0001 | Adam        |               0.5 | grid_trial_5  |     81.8519 |
|          0.001  |         0.0001 | SGD         |               0.5 | grid_trial_7  |     76.2963 |
|          0.001  |         0      | SGD         |               0.4 | grid_trial_2  |     75.1852 |
|          0.001  |         0.0001 | SGD         |               0.4 | grid_trial_6  |     75.1852 |
|          0.001  |         0      | SGD         |               0.5 | grid_trial_3  |     70      |
|          0.0003 |         0      | SGD         |               0.5 | grid_trial_11 |     63.7037 |
|          0.0003 |         0.0001 | SGD         |               0.5 | grid_trial_15 |     36.6667 |
|          0.0003 |         0      | SGD         |               0.4 | grid_trial_10 |     34.0741 |
|          0.0003 |         0.0001 | SGD         |               0.4 | grid_trial_14 |     34.0741 |

**grid_trial_0 training results**

![grid_trial_training_results](https://github.com/KacperTurek/Sperm-Morphology-Image-Classification/blob/main/images/grid_trial_0_training_results.png?raw=true)

### Improved model grid search results

|   learning_rate |   dropout_fc_rate | run_name              |   valid_acc |
|----------------:|------------------:|:----------------------|------------:|
|          0.0005 |              0.35 | improved_grid_trial_7 |     87.7778 |
|          0.001  |              0.3  | improved_grid_trial_0 |     87.4074 |
|          0.0007 |              0.4  | improved_grid_trial_5 |     87.4074 |
|          0.0005 |              0.3  | improved_grid_trial_6 |     87.4074 |
|          0.001  |              0.4  | improved_grid_trial_2 |     87.037  |
|          0.0007 |              0.35 | improved_grid_trial_4 |     87.037  |
|          0.001  |              0.35 | improved_grid_trial_1 |     86.6667 |
|          0.0007 |              0.3  | improved_grid_trial_3 |     86.6667 |
|          0.0005 |              0.4  | improved_grid_trial_8 |     85.5556 |

**improved_grid_trial_7 training results**

![improved_grid_trial_training_results](https://github.com/KacperTurek/Sperm-Morphology-Image-Classification/blob/main/images/improved_grid_trial_7_training_results.png?raw=true)

## Evaluation of Best Model

The best-performing CNN model from the grid search (`best_model`) was selected based on its highest validation accuracy of 87.78%. The evaluation demonstrated the following key outcomes:
- Accuracy and Robustness: The model achieved consistently high validation accuracy across all three morphology classes, indicating a strong ability to generalize despite image resolution variability.
- Confusion Matrix: The confusion matrix showed clear separation between classes, with minimal misclassification. This reflects the model's effective learning of class-specific morphological features.
- Prediction Consistency: Across test samples, the model maintained reliable prediction patterns, even in the presence of noise or minor inconsistencies in sperm cell appearance.

This validates that the CNN grid search was successful, and the selected combination of learning rate, dropout rate, and other hyperparameters significantly improved model performance compared to baseline configurations.

## Analysis of Confusion Matrix

During evaluation of the best CNN model (`best_model`), the normalized confusion matrix showed strong overall performance with particularly high accuracy in classifying Abnormal_Sperm instances (88%).
The model also performed well in distinguishing Non-Sperm samples (83%), and showed solid identification of Normal_Sperm (76%), though with some confusion between Normal_Sperm and Abnormal_Sperm (21%).

This confusion likely stems from subtle morphological differences between normal and abnormal sperm cells, which can be challenging to distinguish, even for human annotators. Unlike previous object detection scenarios, there is no indication of labeling issues or missing annotations in this classification dataset.

The matrix confirms that the model has generalized effectively across all classes and is particularly reliable when identifying abnormal or non-sperm cases, which are critical for downstream clinical interpretations.

![confussion_matrix](https://github.com/KacperTurek/Sperm-Morphology-Image-Classification/blob/main/images/best_model_normalized_confusion_matrix.png?raw=true)

## Analysis of ROC Curve

The evaluation of the best CNN model (`best_model`) also included analysis of the ROC (Receiver Operating Characteristic) curves for each class.
The curves demonstrate that the model is capable of distinguishing between Normal_Sperm, Abnormal_Sperm, and Non-Sperm classes with high confidence.
Each class achieves a high Area Under the Curve (AUC), indicating strong separability and minimal overlap between the prediction distributions.

Notably, the model shows the highest AUC for the Abnormal_Sperm class, aligning with its strong performance in the confusion matrix. The Non-Sperm class also shows reliable discrimination, while Normal_Sperm maintains competitive AUC despite some shared visual features with abnormal samples.

These results validate the model’s ability to make consistent and confident predictions across multiple thresholds, confirming its robustness in multiclass classification scenarios.

![roc_curve](https://github.com/KacperTurek/Sperm-Morphology-Image-Classification/blob/main/images/best_model_roc_curve.png?raw=true)

## Next Steps
- Try more complex CNN architectures (e.g., ResNet, EfficientNet)
- Use data augmentation and regularization techniques
- Deploy model via a Flask or FastAPI backend
