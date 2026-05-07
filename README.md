# Iris Species Classifier

A machine learning project that classifies iris flower species using the famous Iris dataset. This project demonstrates the complete workflow of a classification task, from data preprocessing to model evaluation and visualization.

## Project Overview

The Iris dataset is a classic dataset in machine learning that contains 150 samples of iris flowers, each with four features:
- Sepal length (cm)
- Sepal width (cm) 
- Petal length (cm)
- Petal width (cm)

The goal is to classify each flower into one of three species:
- Setosa
- Versicolor
- Virginica

## Features

### Data Processing
- **Data Splitting**: Uses stratified train-test split (80/20) to maintain class balance
- **Feature Scaling**: StandardScaler applied to normalize features
- **Cross-Validation**: 5-fold cross-validation for reliable performance estimation

### Machine Learning Models
- **Random Forest Classifier**: Primary model with 100 estimators
- **Logistic Regression**: Alternative model for comparison (implemented but not used in main flow)

### Evaluation & Visualization
- **Cross-Validation Scores**: Prints accuracy scores from 5-fold CV
- **Classification Report**: Detailed performance metrics (precision, recall, f1-score)
- **Feature Importance**: Visualizes which features contribute most to predictions
- **Model Comparison**: Framework for comparing multiple models

## Project Structure

```
iris-species-classifier/
├── model.py              # Main implementation file
├── README.md            # Project documentation
└── .venv/               # Virtual environment
```

## Code Architecture

### Core Functions

- `main()`: Orchestrates the entire ML pipeline
- `split_data()`: Handles train-test splitting with stratification
- `random_forest_classification()`: Creates Random Forest pipeline
- `logistic_classification()`: Creates Logistic Regression pipeline
- `get_cross_val_score()`: Performs 5-fold cross-validation
- `get_classification_report()`: Generates detailed classification metrics
- `visualize_importance_by_features()`: Plots feature importance
- `visualize_cross_val_score()`: Compares model performance (framework)

### Pipeline Design

The project uses scikit-learn's Pipeline to chain preprocessing and modeling steps:

```python
Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(n_estimators=100, random_state=42))
])
```

## Dependencies

- `numpy`: Numerical computations
- `scikit-learn`: Machine learning algorithms and utilities
- `matplotlib`: Data visualization
- `time`: Performance timing

## Usage

### Setup
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Model
```bash
python model.py
```

### Expected Output
The script will output:
1. Cross-validation accuracy scores
2. Detailed classification report
3. Feature importance visualization
4. Total execution time

## Performance

The Random Forest classifier typically achieves:
- **Accuracy**: ~96-98% on cross-validation
- **Training Time**: ~0.1 seconds
- **Most Important Features**: Petal length and petal width

## Learning Objectives

This project demonstrates:
1. **Complete ML Workflow**: From data loading to model evaluation
2. **Pipeline Architecture**: Proper separation of preprocessing and modeling
3. **Model Evaluation**: Multiple metrics and validation techniques
4. **Visualization**: Feature importance and performance comparisons
5. **Code Organization**: Clean, modular function design

## Technical Details

### Data Characteristics
- **Samples**: 150 iris flowers
- **Features**: 4 numerical measurements
- **Classes**: 3 species (50 samples each)
- **Balance**: Perfectly balanced dataset

### Model Configuration
- **Random Forest**: 100 trees, random_state=42 for reproducibility
- **Test Split**: 20% of data, stratified by class
- **Cross-Validation**: 5-fold, accuracy scoring
- **Preprocessing**: StandardScaler for feature normalization

## Future Enhancements

Potential improvements:
- Add hyperparameter tuning with GridSearchCV
- Implement additional algorithms (SVM, Neural Networks)
- Add confusion matrix visualization
- Include model persistence (save/load trained models)
- Add command-line interface for parameter configuration
- Implement ensemble methods combining multiple models

## Educational Value

This project serves as an excellent introduction to:
- Supervised machine learning classification
- scikit-learn pipeline patterns
- Model evaluation techniques
- Feature importance analysis
- Cross-validation best practices

Perfect for beginners learning machine learning fundamentals and for experienced developers as a quick reference implementation.