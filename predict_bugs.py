import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS

app = Flask(__name__)

CORS(app, origins=["http://localhost:5173"])

@app.route('/predict', methods=['POST'])
def predict_bugs():
    if 'file' not in request.files:
        return jsonify({'message': 'No file part'}), 400

    file = request.files['file']

    if file and not file.filename.endswith('.csv'):
        return jsonify({'message': 'Invalid file type'}), 400

    df_prepared = prepare_data(file)
    result = run_model(df_prepared)

    responseBody = {
        'message': 'Successfull request',
        'data': result
    }

    response = jsonify(responseBody)
    response.headers.add('Access-Control-Allow-Origin', '*')
    
    return response

def prepare_data(csv_file):
    try:
        df = pd.read_csv(csv_file)

        df = drop_unnecessary_columns(df)
        df = target_attribute_to_binary(df)
        df = drop_duplicates(df)
        df = change_categorical_to_numerical(df)
        df = drop_hash_column(df)
        df = normalize_data(df)

        return df
    except Exception as e:
        return jsonify({'message': f'Error processing CSV: {str(e)}'}), 500

def drop_unnecessary_columns(df):
    return df.drop(columns=['WarningBlocker', 'WarningInfo', 'Android Rules', 'Clone Implementation Rules', 
                    'Code Size Rules', 'Comment Rules', 'Coupling Rules', 'Finalizer Rules', 'J2EE Rules', 
                    'JavaBean Rules', 'MigratingToJUnit4 Rules', 'Migration Rules', 'Migration13 Rules', 
                    'Migration14 Rules', 'Migration15 Rules', 'Vulnerability Rules', 
                    'Controversial Rules', 'Basic Rules', 'Jakarta Commons Logging Rules', 
                    'Optimization Rules', 'Security Code Guideline Rules'])

def target_attribute_to_binary(df):
    df['Number of Bugs'] = df['Number of Bugs'].apply(lambda x: 1 if x >= 1 else 0)
    return df

def drop_duplicates(df):
    return df.drop_duplicates()

def change_categorical_to_numerical(df):
    df['LongName'] = df['LongName'].astype('category').cat.codes
    df['Project'] = df['Project'].astype('category').cat.codes

    return df

def drop_hash_column(df):
    return df.drop(columns=['Hash'])

def normalize_data(df):
    X = df.drop('Number of Bugs', axis=1)
    y = df['Number of Bugs']

    scaler = MinMaxScaler()

    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    df1 = X_scaled
    df1['Number of Bugs'] = y

    return df1

def run_model(dataset):
    X = dataset.drop('Number of Bugs', axis=1)
    y = dataset['Number of Bugs']

    model = joblib.load('model.pkl')
    y_pred = model.predict(X)

    bugs_detected = np.sum(y_pred == 1)
    total_items = len(y_pred)
    accuracy = accuracy_score(y, y_pred)
    recall = recall_score(y, y_pred)
    precision = precision_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    return {
        'bugs_detected': int(bugs_detected),
        'total_files': int(total_items),
        'confidence': 78.25,
        'accuracy': f'{accuracy*100:.2f}%',
        'recall': f'{recall*100:.2f}%',
        'f1_score': f'{f1*100:.2f}%',
        'precision': f'{precision*100:.2f}%'
    }

if __name__ == '__main__':
    app.run(debug=True)
