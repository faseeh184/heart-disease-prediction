# Heart Disease Prediction

Classifies patients as having heart disease or not from clinical measurements. Because a missed diagnosis is costly, the main metric is **recall for the positive class**.


**Dataset:** 303 patients, 13 clinical features (age, sex, chest pain type, blood pressure, cholesterol, max heart rate, and more)

## Results (test set)
| Model | Accuracy | Recall (disease) | F1 (disease) |
|---|---|---|---|
| Decision Tree | 0.79 | 0.85 | 0.81 |
| SVM | 0.79 | **0.97** | 0.83 |
| KNN | **0.84** | 0.85 | 0.85 |
| Random Forest | **0.84** | 0.88 | 0.85 |

- **SVM** catches almost every patient (recall 0.97) but gives more false alarms.
- **Random Forest** gives the best balance of accuracy and recall.

## Approach
EDA → preprocessing (missing values, outliers, encoding, skew correction) → pipelines for scale-sensitive models → hyperparameter tuning (KNN, SVM, Decision Tree, Random Forest) → comparison on precision, recall, F1.

## Tech stack
Python · Pandas · Matplotlib · Seaborn · scikit-learn

## Run it
```bash
pip install -r requirements.txt
jupyter notebook heart-disease-prediction__4_1_.ipynb
```
The notebook was written on Kaggle. Change the data path in the "Read Dataset" step to `data/heart.csv`.

## Credits
Team project with Rahul Kumar. The notebook's structure, problem statement, and objectives are adapted from the public [Heart Disease Prediction project by Farzad Nekouee](https://github.com/FarzadNekouee/Heart_Disease_Prediction), which we used as a learning base.
Educational project, not a medical tool.
