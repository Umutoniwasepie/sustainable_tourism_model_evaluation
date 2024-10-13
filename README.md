# Sustainable_tourism_model_evaluation

## Introduction
This project aims to classify sustainable tourism activities based on their environmental impact (low vs. high) using machine learning models. The goal is to build a model that can accurately distinguish between high and low-impact activities to support sustainable tourism practices. Two neural network models were implemented:
- **Vanilla Model**: A basic neural network without optimizations.
- **Optimized Model**: An enhanced neural network that incorporates various optimization techniques to improve performance.

The project compares the performance of the two models using metrics like accuracy, confusion matrix, and classification report, with a detailed discussion on the optimization techniques applied.

## Project Implementation

### Data Loading and Preprocessing
1. **Data Loading**: The dataset used is `sustainable_Tourism_dataset.csv`, which includes four features: `co2_emissions`, `energy_consumption`, `tourism_activity`, and `impact` (target variable). It consists of 1,000 samples, with no missing values.
2. **Feature Selection**: The target variable `impact` was separated from the features (`co2_emissions`, `energy_consumption`, `tourism_activity`).
3. **Data Splitting**: The data was split into training (60%), validation (20%), and test (20%) sets.
4. **Data Standardization**: To ensure all feature values were in the same range, the `MinMaxScaler` was used to standardize the data.

### Vanilla Model Implementation
#### Vanilla Model Evaluation
- **Accuracy**: The vanilla model achieved a test accuracy of **0.8349** on the test set.
- **Loss**: The test loss was **0.3638**, indicating some overfitting.
- **Confusion Matrix**: The vanilla model had a relatively higher number of misclassifications.
- **Classification Report**: The model's precision, recall, and F1-scores indicated an acceptable baseline performance, but there was room for improvement in classifying both high and low-impact activities.

### Optimized Model Implementation
The optimized model incorporated several optimization techniques to improve performance:
1. **Dropout Regularization**:
   - **Principle**: Dropout reduced overfitting by randomly dropping a fraction of neurons during training, forcing the model to learn more generalized features.
   - **Application**: Dropout layers with a rate of 0.4 were added after each hidden layer to enhance generalization.
2. **Adam Optimizer**:
   - The Adam optimizer was used with a learning rate of 0.001 to improve convergence speed and accuracy.
3. **Early Stopping**:
   - **Principle**: Early stopping monitors the validation loss during training and stops the training process when the validation performance ceases to improve, thus preventing overfitting.
   - **Application**: The patience was set to 5 epochs, meaning the training would stop if the validation loss did not improve for 5 consecutive epochs.
   
#### Optimized Model Evaluation
- **Accuracy**: The optimized model achieved an accuracy of **0.865** on the test set.
- **Loss**: The test loss was **0.29**, indicating better-calibrated predictions than the vanilla model.
- **Confusion Matrix**: The optimized model showed fewer misclassifications, demonstrating improved performance.
- **Classification Report**: The model's precision, recall, and F1-scores were higher compared to the vanilla model, reflecting the benefits of the optimization techniques.

### Significance of Optimization Techniques
- **Dropout Regularization**: By adding dropout layers, the model was less prone to overfitting and generalized better to new data.
- **Adam Optimizer**: The adaptive learning rate helped the model converge faster and find a better local minimum in the loss surface.
- **Early Stopping**: This technique allowed the model to stop training before it started overfitting, ensuring better generalization on the test data.

## Comparison of Results
| Metric                         | Vanilla Model | Optimized Model |
|--------------------------------|---------------|-----------------|
| **Test Accuracy**              | 83%           | 86%             |
| **Test Loss**                  | 0.36          | 0.29            |
| **Precision (High Impact)**    | 0.86          | 0.89            |
| **Recall (High Impact)**       | 0.81          | 0.84            |
| **F1-Score (High Impact)**     | 0.83          | 0.86            |
| **Precision (Low Impact)**     | 0.82          | 0.84            |
| **Recall (Low Impact)**        | 0.86          | 0.89            |
| **F1-Score (Low Impact)**      | 0.84          | 0.87            |


## Conclusion
The project successfully demonstrated the use of neural networks for classifying sustainable tourism activities based on their environmental impact. The optimized model, incorporating dropout, the Adam optimizer, and early stopping, showed significant improvements over the vanilla model. These optimizations led to better generalization, fewer misclassifications, and higher classification metrics.
