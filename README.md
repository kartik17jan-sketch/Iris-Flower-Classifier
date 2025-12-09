Iris Flower Classification using Logistic Regression

This project implements a machine learning classifier to predict the species of Iris flowers based on four numeric features from the classic Iris dataset.
A Logistic Regression model is trained to classify flowers into three species:

Setosa (0)

Versicolor (1)

Virginica (2)

The model achieves 100% accuracy on the test dataset.

Project Structure
iris-flower-classifier/
│
├── Iris Classifier.ipynb         # Main notebook
├── README.md                     # Documentation
├── requirements.txt              # Dependencies
│
└── results/                      # Visualizations (optional)
    ├── sepal_length_boxplot.png
    ├── sepal_width_boxplot.png
    ├── petal_length_boxplot.png
    └── petal_width_boxplot.png

Dataset Overview

Iris-Flower-Classifier_Iris Cla…

:

sepal length (cm)
sepal width (cm)
petal length (cm)
petal width (cm)
species


Dataset details:

Total samples: 150

Features: 4

Classes: 3 (0, 1, 2)

Balanced data: 50 samples per class (page 5) 

Iris-Flower-Classifier_Iris Cla…

Exploratory Data Analysis (EDA)

The notebook includes:



Mean

Standard deviation

Quartiles

Min/Max values

✔ No missing values (page 3)
iris_data.isnull().sum()

✔ Boxplots for all features vs species

Plots for:

Sepal length (page 3)

Sepal width (page 4)

Petal length (page 4)

Petal width (page 5)

Showing clear separability between species — especially setosa and virginica.

Preprocessing
1. Train–test split (page 6)
X_train, X_test, y_train, y_test = train_test_split(...)

2. Feature scaling (Standardization)

Scaling is applied to all four features (page 6):

scale = StandardScaler()
X_scaled = scale.fit_transform(X)

Model — Logistic Regression

Trained using:

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)


(Logistic Regression is perfect for small, clean, linearly separable datasets like Iris.)

Evaluation
✔ Predictions

y_pred = model.predict(X_test)

✔ Classification report (page 6)

All precision, recall, f1-score = 1.00
Accuracy = 1.00 (100%)

precision  recall  f1-score  support
0   1.00    1.00     1.00       10
1   1.00    1.00     1.00        9
2   1.00    1.00     1.00       11

accuracy: 1.00

✔ Confusion matrix (page 7)
[[10  0  0]
 [ 0  9  0]
 [ 0  0 11]]


Perfect classification across all species.

Prediction on New Input

The notebook includes a custom input for a new flower (page 7):

test_plant = {
 'sepal length (cm)' : 10,
 'sepal width (cm)'  : 5.4,
 'petal length (cm)' : 3.4,
 'petal width (cm)'  : 6.9
}
model.predict(plant_df)


Output:

array([2])   # Virginica

How to Run This Project
1. Clone the repository
git clone https://github.com/<username>/Iris-Flower-Classifier.git
cd Iris-Flower-Classifier

2. Install dependencies
pip install -r requirements.txt

3. Open the notebook

Run:

Iris Classifier.ipynb

requirements.txt
pandas
numpy
scikit-learn
matplotlib
seaborn

Future Improvements

Experiment with SVM, Decision Trees, Random Forest

Add cross-validation for more robust evaluation

Add PCA visualization (2D/3D plots)

Create a small Flask/Streamlit demo for live prediction

Author

Kartik Rajesh
B.Tech CSE (AI/ML)
Logistic Regression • Data Visualization • ML Classification
