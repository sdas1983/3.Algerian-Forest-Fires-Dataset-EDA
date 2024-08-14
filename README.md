# 3.Algerian Forest Fires Dataset Exploratory Data Analysis (EDA)

This project explores the Algerian Forest Fires dataset, performing data preprocessing, exploratory data analysis (EDA), and implementing a Logistic Regression model to predict forest fire occurrences.

## Dataset Overview

The dataset contains data related to forest fires in two regions of Algeria: Bejaia and Sidi-Bel Abbes. The dataset includes various meteorological and fire weather indices, along with a class label indicating the occurrence of fire.

## Project Structure

- **Data Preprocessing**
  - Load and clean the dataset.
  - Handle missing values and incorrect data.
  - Convert data types as needed.
  - Encode categorical variables.
  
- **Exploratory Data Analysis (EDA)**
  - Visualize the distribution of classes (fire/no fire).
  - Analyze the data by month and region.
  - Explore relationships between variables using boxplots, count plots, and correlation heatmaps.
  
- **Model Building**
  - Prepare the data for modeling.
  - Implement Logistic Regression for fire prediction.
  - Evaluate the model using various metrics (accuracy, precision, F1-score).
  - Visualize the model's performance with confusion matrix and ROC curve.

## Data Preprocessing

1. **Loading the Dataset**: The dataset is loaded from a CSV file and inspected for any missing values or erroneous data.

2. **Region Assignment**: The dataset is divided into two regions: Bejaia and Sidi-Bel Abbes.

3. **Data Cleaning**: Invalid rows are removed, and columns are stripped of any leading or trailing spaces.

4. **Type Conversion**: Necessary columns are converted to appropriate data types (e.g., `int`, `float`).

## Exploratory Data Analysis (EDA)

1. **Boxplots**: Visualize the distribution of continuous variables such as Temperature, Rain, and various fire weather indices.

2. **Class Distribution**: Analyze the distribution of fire/no fire classes across the dataset.

3. **Month-wise Analysis**: Explore the occurrence of forest fires by month and region.

4. **Correlation Heatmap**: Examine the correlations between various meteorological and fire weather indices.

## Model Building and Evaluation

1. **Data Preparation**: Split the dataset into training and testing sets. Apply standard scaling to the features.

2. **Logistic Regression Model**: Train a Logistic Regression model to predict the occurrence of forest fires.

3. **Model Evaluation**:
   - **Accuracy**: Measure the overall accuracy of the model.
   - **Precision and F1-Score**: Evaluate the model's precision and F1-score.
   - **Confusion Matrix**: Visualize the confusion matrix to understand the model's performance in predicting fire/no fire classes.
   - **ROC Curve**: Plot the ROC curve to evaluate the model's performance.

## Results and Visualizations

- **Class Distribution**: Visualized using both seaborn and Plotly.
- **Month-wise Fire Analysis**: Explored through count plots and bar charts, differentiated by region.
- **Correlation Heatmap**: Provides insights into the relationships between the various features.
- **Model Performance**: Assessed through confusion matrix, classification report, and ROC curve.

## Dependencies

- `pandas`
- `matplotlib`
- `seaborn`
- `plotly`
- `scikit-learn`
- `pickle`

These libraries are used for data manipulation, visualization, and model building.

## Conclusion

This project provides a comprehensive analysis of the Algerian Forest Fires dataset, exploring the data in detail and building a predictive model to assist in forest fire prevention and management.

---

**Note**: The code is structured to ensure clarity and reproducibility. The results of each analysis step are presented using appropriate visualizations, making it easy to understand the insights derived from the dataset.
