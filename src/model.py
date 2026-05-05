import pandas as pd

data=pd.read_csv('Data/soil_measures.csv')

# print(data.info())
# print(data.head()) 

features=['N','P','K','ph']

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

X_train,X_test,y_train,y_test=train_test_split(data[features],data['crop'],test_size=0.2,random_state=42)

from sklearn.linear_model import LogisticRegression
model_performance={}
for feature in features:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[[feature]])
    X_test_scaled = scaler.transform(X_test[[feature]])
    model=LogisticRegression(max_iter=10000,solver='lbfgs')


    model.fit(X_train_scaled,y_train)
    y_pred=model.predict(X_test_scaled)
    
    model_performance[feature]=f1_score(y_test,y_pred,average='weighted')
    print(f"F1 Score for {feature}: {model_performance[feature]}")
    
correlation=data[features].corr()    
print(correlation)
import seaborn as sns

sns.heatmap(correlation, annot=True)
print("correlation heatmap displayed",sns.heatmap(correlation, annot=True))

#select the final features based on correlation and model perfromance

final_Features=['K','N','ph']

X_train,X_test,y_train,y_test=train_test_split(data[final_Features],data['crop'],test_size=0.2,random_state=42)
scaler = StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
model=LogisticRegression(max_iter=10000,solver='lbfgs')
model.fit(X_train_scaled,y_train)
y_pred=model.predict(X_test_scaled)
final_f1_score=f1_score(y_test,y_pred,average='weighted')
print(f"Final F1 Score with selected features: {final_f1_score}")

import joblib as jb
jb.dump(model, 'model/soil_crop_model.pkl')
print("Model saved successfully as 'model/soil_crop_model.pkl'")
