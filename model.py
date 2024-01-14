import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# reading the data.
df = pd.read_csv(r'health.csv')

X = df.drop('PremiumPrice', axis=1)
Y = df['PremiumPrice']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

# Import the Algorithm 

#from sklearn.tree import DecisionTreeRegressor

model = RandomForestClassifier()


#dtr = DecisionTreeRegressor()

# training of the model
model.fit(X_train, y_train)
#dtr.fit(X_train, y_train)

print(model.score(X_train, y_train))


pickle_out = open(r"model.pkl", "wb")
pickle.dump(model, pickle_out)
pickle_out.close()
