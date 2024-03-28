from flask import Blueprint, Flask, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from flask_restful import Api, Resource

app = Flask(__name__)
lung_api = Blueprint('lung_api', __name__, url_prefix='/api/lung')
api = Api(lung_api)

# Load the dataset
csv_file_path = "/home/nandanv/vscode/cpt_Final/survey_lung_cancer.csv"
df = pd.read_csv(csv_file_path)

class CancerPredict(Resource):
    def __init__(self):
        # Select features and target
        self.categorical_features = ['GENDER', 'SMOKING', 'ANXIETY']  
        self.numeric_features = ['AGE']  
        self.target = 'LUNG_CANCER'  

        # One-hot encode categorical features
        encoder = OneHotEncoder(drop='first')
        encoder.fit(df[self.categorical_features])  
        self.encoder = encoder  

        # Preprocess input data
        self.X_numeric = df[self.numeric_features]
        self.X_categorical = pd.DataFrame(encoder.transform(df[self.categorical_features]).toarray(),
                                           columns=encoder.get_feature_names_out(self.categorical_features))
        self.X = pd.concat([self.X_numeric, self.X_categorical], axis=1)
        self.y = df[self.target]

        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # Train a machine learning model
        self.model = RandomForestClassifier()
        self.model.fit(self.X_train, self.y_train)

    def post(self):
        try:
            # Get the JSON data from the request
            json_data = request.get_json()

            # Convert JSON data to DataFrame
            input_data = pd.DataFrame(json_data, index=[0])

            # Preprocess input data
            input_numeric = input_data[self.numeric_features]
            input_categorical = pd.DataFrame(self.encoder.transform(input_data[self.categorical_features]).toarray(),
                                              columns=self.encoder.get_feature_names_out(self.categorical_features))
            input_features = pd.concat([input_numeric, input_categorical], axis=1)

            # Make predictions
            probability = self.model.predict_proba(input_features)[0, 1]  # Get the probability for class 1
            return {'probability': probability}, 200
        except Exception as e:
            return {'error': str(e)}, 400

api.add_resource(CancerPredict, '/predict')

if __name__ == '__main__':
    app.register_blueprint(lung_api)
    app.run(debug=True)
