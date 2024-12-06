import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, classification_report



df = pd.read_csv("diabetes_binary_health_indicators_BRFSS2015.csv")

# rename Diabetes_binary to Diabetes
df.rename(columns = {'Diabetes_binary' : 'Diabetes'}, inplace = True)
df.drop_duplicates(inplace=True)
#print(df.info())

y = df['Diabetes']

#print(X.head())

scaler = MinMaxScaler()
# Use features decided on
#select_features = ['HighBP', 'HighChol', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity',
                   #'HvyAlcoholConsump', 'DiffWalk', 'BMI', 'GenHlth', 'MentHlth', 'PhysHlth', 'Age', 'Education', 'Income']
#X = df[select_features]
prune_features = ['Diabetes']
#prune_features = ['Diabetes', 'CholCheck', 'Fruits', 'Veggies', 'AnyHealthcare', 'NoDocbcCost', 'Sex']
X = df.drop(columns=prune_features)
X = pd.DataFrame(scaler.fit_transform(X))

# split into testing and training set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1, stratify=y)

# further split into training and validation set
X_train2, X_valid, y_train2, y_valid = train_test_split(X_train, y_train, test_size = 0.25, random_state = 1, stratify=y_train)

print(X_train2.shape)
y_train2.value_counts()



smote = SMOTEENN(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
#X_resampled2, y_resampled2 = smote.fit_resample(X_train2, y_train2)

# create dict of hyperparams to tune
hyperparam_dict = {
    'classification__max_depth': [5],
    'classification__max_features': [5, 10],
    'classification__n_estimators': [100, 500, 1000]
}

model = Pipeline([
    ('sampling', SMOTEENN()),
    ('classification', RandomForestClassifier())
])

grid_search = GridSearchCV(estimator=model, param_grid = hyperparam_dict,scoring='f1', return_train_score=True,cv=4, verbose=3)
grid_search.fit(X_train, y_train)

# get best hyperparameters found
best_hyperparams = grid_search.best_params_
best_max_depth = best_hyperparams['classification__max_depth']
best_max_features = best_hyperparams['classification__max_features']
best_max_n_estimators = best_hyperparams['classification__n_estimators']
print("Best Max Depth:", best_max_depth, "Best max features:", best_max_features, "Best n iters:", best_max_n_estimators)

clf = RandomForestClassifier(max_depth=best_max_depth, max_features=best_max_features)#, n_estimators=best_max_n_estimators)
clf.fit(X_resampled, y_resampled)
y_pred_resampled = clf.predict(X_resampled)
print(classification_report(y_resampled, y_pred_resampled))
print()
y_pred_test = clf.predict(X_test)
print(classification_report(y_test, y_pred_test))

