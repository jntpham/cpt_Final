from flask import Blueprint, jsonify, Flask, request
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from flask_restful import Api, Resource
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
titanic_api = Blueprint('titanic_api', __name__, url_prefix='/api/titanic')
api = Api(titanic_api)

class Predict(Resource):
    def post(self):
        try:
            data = request.get_json()
            passenger_data = pd.DataFrame(data, index=[0])
            
            # Preprocesssing
            passenger_data['sex'] = passenger_data['sex'].apply(lambda x: 1 if x == 'male' else 0)
            passenger_data['alone'] = passenger_data['alone'].apply(lambda x: 1 if x == True else 0)
            onehot = enc.transform(passenger_data[['embarked']]).toarray()
            cols = ['embarked_' + val for val in enc.categories_[0]]
            passenger_data[cols] = pd.DataFrame(onehot)
            passenger_data.drop(['embarked'], axis=1, inplace=True)
            
            # Predict the survival probability for the new passenger
            survival_prob = logreg.predict_proba(passenger_data)[:, 1]
            death_prob = 1 - survival_prob

            return {'death_percentage': float(death_prob * 100), 'survivability_percentage': float(survival_prob * 100)}, 200
        except Exception as e:
            return {'error': str(e)}, 400

api.add_resource(Predict, '/predict')

# Load Titanic dataset
titanic_data = sns.load_dataset('titanic')

# Preprocess the data
titanic_data.drop(['alive', 'who', 'adult_male', 'class', 'embark_town', 'deck'], axis=1, inplace=True)
titanic_data.dropna(inplace=True)
titanic_data['sex'] = titanic_data['sex'].apply(lambda x: 1 if x == 'male' else 0)
titanic_data['alone'] = titanic_data['alone'].apply(lambda x: 1 if x == True else 0)

# Encode categorical variables
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(titanic_data[['embarked']])
onehot = enc.transform(titanic_data[['embarked']]).toarray()
cols = ['embarked_' + val for val in enc.categories_[0]]
titanic_data[cols] = pd.DataFrame(onehot)
titanic_data.drop(['embarked'], axis=1, inplace=True)
titanic_data.dropna(inplace=True)

# Split the data into features and target
X = titanic_data.drop('survived', axis=1)
y = titanic_data['survived']

# Train the logistic regression model
logreg = LogisticRegression()
logreg.fit(X, y)

if __name__ == "__main__":
    app.run(debug=True)
