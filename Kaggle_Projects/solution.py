import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("Audi_A1_listings.csv")
print(df.head(3))
print(df.info())
df = df.drop(columns=["index", "Type", "PS", "Number_of_Owners", "href", "PPY", "MileageRank", "PriceRank", "PPYRank", "Score"])
df["Engine"] = df["Engine"].str.replace("L", "")
df = df.rename(columns={"Price(£)": "Price"})
df = df.rename(columns={"Mileage(miles)": "Mileage"})
print(df["Engine"].info())
df["Engine"] = pd.to_numeric(df["Engine"], errors="coerce")
print(df) 
df = pd.get_dummies(df, columns=["Transmission", "Fuel"], drop_first=True)
print(df)

X = df.drop(columns=["Price"])
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

grid = {
    "n_estimators": [50, 100, 200],
    "max_depth" : [None, 10, 20],
    "min_samples_split": [2, 5, 10]
}
rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(param_grid = grid, estimator=rf, cv=5, scoring= "neg_mean_absolute_error", n_jobs= -1)
print(grid_search)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, pred)
print(pred, mae)