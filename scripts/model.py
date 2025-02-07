import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

def load_and_prepare_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    df = pd.DataFrame.from_dict(data, orient='index')
    
    features = [
        'total_inflow', 'total_outflow', 'net_cash_flow',
        'avg_monthly_inflow', 'avg_monthly_outflow',
        'cash_flow_volatility', 'negative_flow_months',
        'min_balance', 'max_balance', 'num_overdrafts', 'dscr'
    ]
    
    X = df[features]
    y = df['loan']
    
    return X, y

def train_loan_model(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    sample_weights = np.ones(len(X))
    
    net_cash_flow_col = X.columns.get_loc('net_cash_flow')
    sample_weights[X.iloc[:, net_cash_flow_col] < 0] = 2.0
    
    dscr_col = X.columns.get_loc('dscr')
    sample_weights[X.iloc[:, dscr_col] < 1.0] *= 1.5
    
    high_risk_mask = (X.iloc[:, net_cash_flow_col] < 0) & (X.iloc[:, dscr_col] < 1.0)
    sample_weights[high_risk_mask] *= 1.5
    
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight={0: 1.5, 1: 1},
        random_state=42,
        max_features='sqrt',
        bootstrap=True,
        n_jobs=-1
    )
    
    if len(X) < 5:
        from sklearn.model_selection import LeaveOneOut
        cv = LeaveOneOut()
        cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
    else:
        cv_scores = cross_val_score(model, X_scaled, y, cv=2, scoring='accuracy')
    
    model.fit(X_scaled, y, sample_weight=sample_weights)
    
    return model, scaler, cv_scores

def predict_loan(model, scaler, new_data):
    new_data_scaled = scaler.transform(new_data)
    
    prediction = model.predict(new_data_scaled)
    probabilities = model.predict_proba(new_data_scaled)
    
    automatic_denial = (
        (new_data['net_cash_flow'].values[0] < -10000) or
        (new_data['dscr'].values[0] < 0.95) or
        (new_data['negative_flow_months'].values[0] > 3)
    )
    
    if automatic_denial:
        prediction = np.array([0])
        probabilities = np.array([[0.9, 0.1]])
    
    return prediction, probabilities

def get_feature_importance(model, feature_names):
    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })
    return importances.sort_values('importance', ascending=False)

def main():
    X, y = load_and_prepare_data('bank_statement_metrics/all_statements_summary.json')
    
    model, scaler, cv_scores = train_loan_model(X, y)
    
    print("Average accuracy: {:.2f}% (+/- {:.2f}%)".format(
        cv_scores.mean() * 100, cv_scores.std() * 200))
    
    feature_importance = get_feature_importance(model, X.columns)
    print("\nFeature Importance:")
    print(feature_importance)
    
    sample_data = X.iloc[[0]]
    prediction, probabilities, confidence = predict_loan(model, scaler, sample_data)
    
    print("\nSample Prediction:")
    print("Loan Approved:" if prediction[0] == 1 else "Loan Denied")
    print("Probability of approval: {:.2f}%".format(probabilities[0][1] * 100))
    print("Prediction confidence: {:.2f}%".format(confidence[0] * 100))

if __name__ == "__main__":
    main()