import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

#Create sample data
np.random.seed(42)
n_samples = 1000

#Generate features
size = np.random.normal(2000, 500, n_samples)
rooms = np.random.randint(2, 7, n_samples)
age = np.random.randint(0, 50, n_samples)
location = np.random.choice(['A', 'B', 'C'], n_samples)

#Generate target (price) with some relateionship to features
price = (
    size * 100 +                                #size has large impact
    rooms * 20000 +                             #rooms have medium impact
    -age * 1000 +                               #age has negative impact
    np.random.normal(0, 50000, n_samples)       #add some noise
)

#Create dataframe
data = pd.DataFrame({
    'size': size,
    'rooms': rooms,
    'age': age,
    'location': location,
    'price': price
})

#save the data
data.to_csv('housing_data.csv', index = False)
print("Sample data saved to 'housing_data.csv'")

#load and prepare data
try:
    
    #one-hot encode location
    data_encoded = pd.get_dummies(data, columns=['location'], prefix=['location'])

    #separate features and target
    x = data_encoded.drop('price', axis = 1)
    y = data_encoded['price']

    #split the data
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    #scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #train Lasso model
    lasso = Lasso(alpha=0.1)
    lasso.fit(X_train_scaled, y_train)

    #make predictions
    y_pred = lasso.predict(X_test_scaled)

    #calculate metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    #get feature imporatnce 
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'coefficient': abs(lasso.coef_)
    }).sort_values('coefficient', ascending = False)

    #print results
    print("\nModel Performance:")
    print(f"R™ Score: {r2:.4f}")
    print(f"Mean Absolute Error: ${mae:.2f}")

    print("\nFeature Imporatnce:")
    print(feature_importance)

    #save results
    results = {
        'feature_importance': feature_importance.to_dict('records'),
        'metrics':{
            'r2': r2,
            'mae': mae
        }
    }

    #save to JSON
    import json
    with open('model_results.json','w') as f:
        json.dump(results, f, indent = 4)
    print("\nResults saved to 'model_results.json'")

except Exception as e:
    print(f"An error occurred: {str(e)}")
