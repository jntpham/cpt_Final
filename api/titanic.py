from flask import Flask, Blueprint, request, jsonify
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import json, jwt
from flask import Blueprint, request, jsonify, current_app, Response
from flask_restful import Api, Resource # used for REST API building
from datetime import datetime
from auth_middleware import token_required


titanic_api = Blueprint('titanic_api', __name__,
                   url_prefix='/api/titanic')

api = Api(titanic_api)



class TitanicAPI:        
    class predict(Resource):  # User API operation for Create, Read.  THe Update, Delete methods need to be implemeented
      def post(self):
        data = request.get_json()
        passenger_data = pd.DataFrame(data, index=[0])
        passenger_data['sex'] = passenger_data['sex'].apply(lambda x: 1 if x == 'male' else 0)
        passenger_data['alone'] = passenger_data['alone'].apply(lambda x: 1 if x == True else 0)
        onehot = enc.transform(passenger_data[['embarked']]).toarray()
        cols = ['embarked_' + val for val in enc.categories_[0]]
        passenger_data[cols] = pd.DataFrame(onehot)
        passenger_data.drop(['embarked'], axis=1, inplace=True)
        prediction = dt.predict(passenger_data)[0]
        return jsonify({'survival_prediction': prediction})


api.add_resource(TitanicAPI, '/predict')

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

# Split the data into training and testing sets
X = titanic_data.drop('survived', axis=1)
y = titanic_data['survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a decision tree classifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)


if __name__ == "__main__":
    app.run(debug=True)