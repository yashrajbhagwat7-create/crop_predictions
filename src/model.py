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

