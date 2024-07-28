!pip install pandas numpy scikit-learn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

!unzip -q house-prices-advanced-regression-techniques.zip


ds=pd.read_csv("train.csv")

ds

features=["GrLivArea","BedroomAbvGr","FullBath"]
X=ds[features]
y=ds["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model=LinearRegression()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("Mean Squared Error:",mse)
print("R-squared:",r2)

#Plotting predictions against actual values
plt.scatter(y_test,y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted")
plt.show()

new_data=pd.DataFrame({
    "GrLivArea":[2000],
    "BedroomAbvGr":[3],
    "FullBath":[2]
})
predicted_price=model.predict(new_data)
print("Predicted Price:",predicted_price[0])
