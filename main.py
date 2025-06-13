# main.py
from model_pipeline import FinalPipeline
from generate_data import DefinitiveDataGenerator
from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def main():
    NUM_CUSTOMERS = 30000
    HISTORY_MONTHS = 9
    master_csv_file = Path("data/definitive_causal_data.csv")
    
    print("--- Generating Definitive Causal Dataset ---")
    generator = DefinitiveDataGenerator(seed=42)
    master_df = generator.generate(num_customers=NUM_CUSTOMERS, num_months=HISTORY_MONTHS + 3)
    master_df.to_csv(master_csv_file, index=False)
    
    print("\n--- Running Final, Lean Pipeline ---")
    pipeline = FinalPipeline(
        data_path=master_csv_file, 
        target_col='target_credit_score_movement',
        feature_cutoff_month=HISTORY_MONTHS
    )
    pipeline.run(n_trials=50)
    
    print("\n--- Final Model Evaluation ---")
    model = joblib.load(pipeline.output_path / "model.pkl")
    X_train, X_test, y_train, y_test_encoded = train_test_split(
        pipeline.X, pipeline.y, test_size=0.2, random_state=42, stratify=pipeline.y
    )
    
    # Retrain on 80% split for fair evaluation
    model.fit(X_train, y_train)
    preds_encoded = model.predict(X_test)
    
    print("\n--- TEST SET PERFORMANCE (on hold-out set) ---")
    # Use the pipeline's fitted label encoder to get original string labels for the report
    print(classification_report(pipeline.le.inverse_transform(y_test_encoded), 
                                pipeline.le.inverse_transform(preds_encoded), 
                                digits=3))

if __name__ == "__main__":
    main()