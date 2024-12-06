import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, recall_score



df = pd.read_csv("diabetes_binary_health_indicators_BRFSS2015.csv")
#df = pd.read_csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
# rename Diabetes_binary to Diabetes
df.rename(columns = {'Diabetes_binary' : 'Diabetes'}, inplace = True)

#print(df.info())

y = df['Diabetes']

#print(X.head())

scaler = StandardScaler()
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
X_resampled2, y_resampled2 = smote.fit_resample(X_train2, y_train2)
print("Oversampled")

# create dict of hyperparams to tune
hyperparam_dict = {
    'classification__loss': ['hinge', 'log_loss', 'modified_huber'],
    'classification__alpha': [0.0001, 0.001, 0.01, 0.1],
}


best_loss = -1
best_alpha = -1
best_score = -1

for loss in hyperparam_dict['classification__loss']:
    for alpha in hyperparam_dict['classification__alpha']:
        clf = SGDClassifier(loss = loss, alpha = alpha)
        clf.fit(X_resampled2, y_resampled2)
        y_pred_valid = clf.predict(X_valid)
        score = recall_score(y_valid, y_pred_valid)
        print("Loss = ", loss, "alpha = ", alpha, "Recall = ", score)
        if score > best_score:
            best_score = score
            best_loss = loss
            best_alpha = alpha

print("Best loss = ", best_loss, "Best alpha:", best_alpha)

clf = SGDClassifier(loss = best_loss, alpha = best_alpha)

clf.fit(X_resampled, y_resampled)
y_pred_resampled = clf.predict(X_resampled)
print(classification_report(y_resampled, y_pred_resampled))

print()
y_pred_test = clf.predict(X_test)
print(classification_report(y_test, y_pred_test))
print()

'''
df2 = pd.read_csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
df2.rename(columns = {'Diabetes_binary' : 'Diabetes'}, inplace = True)
X_5050 = df2.drop(columns=prune_features)
X_5050 = pd.DataFrame(scaler.fit_transform(X_5050))
y_5050 = df2['Diabetes']

print("50/50 split test")
y_pred_5050 = clf.predict(X_5050)
print(classification_report(y_5050, y_pred_5050))
'''