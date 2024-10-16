import pandas as pd

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
 
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder

from sklearn.utils import shuffle

contract = pd.read_csv('/datasets/final_provider/contract.csv')
personal = pd.read_csv('/datasets/final_provider/personal.csv')
services = pd.read_csv('/datasets/final_provider/internet.csv')
phone = pd.read_csv('/datasets/final_provider/phone.csv')

contract['TotalCharges'] = pd.to_numeric(contract['TotalCharges'], errors='coerce') 
contract['TotalCharges'].fillna(contract['TotalCharges'].median(), inplace=True)

contract['target'] = contract['EndDate'].apply(lambda x: 1 if x != 'No' else 0)

merged_df = contract.merge(personal, on='customerID', how='left')
merged_df = merged_df.merge(services, on='customerID', how='left')
merged_df = merged_df.merge(phone, on='customerID', how='left')

merged_df.drop(columns=['customerID', 'BeginDate', 'EndDate'], inplace=True)

merged_df.columns = merged_df.columns.str.lower()

display(merged_df.head())

print(merged_df['multiplelines'].value_counts())
merged_df.info()

merged_df[['internetservice', 'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport', 'streamingtv', 'streamingmovies', 'multiplelines']] = merged_df[['internetservice', 'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport', 'streamingtv', 'streamingmovies', 'multiplelines']].fillna('Unknown')

encoded_data = merged_df
categorical_columns = ['type', 'paperlessbilling', 'paymentmethod', 'gender', 'partner', 'dependents', 'internetservice', 'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport', 'streamingtv', 'streamingmovies', 'multiplelines']
for column in categorical_columns:
    le = LabelEncoder()
    encoded_data[column] = le.fit_transform(encoded_data[column])

display(encoded_data.head())
encoded_data.info()

target = encoded_data['target']
features = encoded_data.drop('target', axis=1)
features_train_val, features_test, target_train_val, target_test = train_test_split(features, target, test_size=0.2, random_state=42, stratify=target)

features_train, features_val, target_train, target_val = train_test_split(features_train_val, target_train_val, test_size=0.25, random_state=42, stratify=target_train_val)

print(f"Tamaño del conjunto de entrenamiento: {features_train.shape[0]}")
print(f"Tamaño del conjunto de validación: {features_val.shape[0]}")
print(f"Tamaño del conjunto de prueba: {features_test.shape[0]}")

def upsample(features, target, repeat):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]
    
    features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
    target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)
    
    features_upsampled, target_upsampled = shuffle(features_upsampled, target_upsampled, random_state=12345)
    
    return features_upsampled, target_upsampled

features_valid_upsampled, target_valid_upsampled = upsample(features_val, target_val, 2)
features_train_upsampled, target_train_upsampled = upsample(features_train, target_train, 2)
print(target_valid_upsampled.value_counts(normalize=True))

features_train_upsampled, target_train_upsampled = upsample(features_train, target_train, 2)
features_val_upsampled, target_val_upsampled = upsample(features_val, target_val, 2)
features_test_upsampled, target_test_upsampled = upsample(features_test, target_test, 2)

logistic_regression = LogisticRegression(random_state=12345, max_iter=160)
logistic_regression.fit(features_train_upsampled, target_train_upsampled)
logistic_pred = logistic_regression.predict(features_val_upsampled)

random_forest = RandomForestClassifier(random_state=12345)
random_forest.fit(features_train_upsampled, target_train_upsampled)
rf_pred = random_forest.predict(features_val_upsampled)

decision_tree = DecisionTreeClassifier(random_state=12345)
decision_tree.fit(features_train_upsampled, target_train_upsampled)
dt_pred = decision_tree.predict(features_val_upsampled)

knn = KNeighborsClassifier()
knn.fit(features_train_upsampled, target_train_upsampled)
knn_pred = knn.predict(features_val_upsampled)

train_data = lgb.Dataset(features_train_upsampled, label=target_train_upsampled)
test_data = lgb.Dataset(features_test_upsampled, label=target_test_upsampled, reference=train_data)

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.01,
    'num_leaves': 9,
    'min_data_in_leaf': 20
}

lgbm_model = lgb.train(params, train_data, valid_sets=[test_data], early_stopping_rounds=19)
lgbm_pred = (lgbm_model.predict(features_test_upsampled, num_iteration=lgbm_model.best_iteration) > 0.5).astype(int)

cat_model = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6, eval_metric='F1', random_seed=12345, verbose=0)
cat_model.fit(features_train_upsampled, target_train_upsampled, eval_set=(features_test_upsampled, target_test_upsampled), use_best_model=True)
catboost_pred = cat_model.predict(features_test_upsampled)

models = {
    "Logistic Regression": logistic_pred,
    "Random Forest": rf_pred,
    "Decision Tree": dt_pred,
    "K-Nearest Neighbors": knn_pred,
    "LightGBM": lgbm_pred,
    "CatBoost": catboost_pred
}

for model_name, predictions in models.items():
    f1 = f1_score(target_val_upsampled, predictions)
    print(f"{model_name} - F1 Score: {f1:.4f}")

# Logistic Regression - F1 Score: 0.7209
# Random Forest - F1 Score: 0.6208
# Decision Tree - F1 Score: 0.5194
# K-Nearest Neighbors - F1 Score: 0.5949
# LightGBM - F1 Score: 0.6931
# CatBoost - F1 Score: 0.7287