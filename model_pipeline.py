# model_pipeline.py
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import joblib
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

class FinalPipeline:
    def __init__(self, data_path, target_col, feature_cutoff_month, output_dir="outputs"):
        # This init method is correct
        if isinstance(data_path, pd.DataFrame):
            self.panel_df = data_path
        else:
            self.panel_df = pd.read_csv(data_path, dtype={'customer_id': 'int32', 'month': 'int8'})
        
        self.target_col = target_col
        self.feature_cutoff_month = feature_cutoff_month
        self.output_path = Path(output_dir)
        self.output_path.mkdir(exist_ok=True)
        self.le = LabelEncoder()
        
    def _feature_engineering(self):
        # This feature engineering is correct and powerful
        print("Performing lean, high-signal, point-in-time feature engineering...")
        feature_df = self.panel_df[self.panel_df['month'] <= self.feature_cutoff_month].copy()
        grp = feature_df.groupby('customer_id')
        last_month_df = grp.last()
        state_df = last_month_df[['credit_utilization_ratio', 'repayment_history_score', 'dpd_last_3_months', 'age', 'monthly_income']]
        history_6m = feature_df[feature_df['month'] > self.feature_cutoff_month - 6].groupby('customer_id')
        def get_trend_slope(series):
            if len(series) < 2: return 0
            return np.polyfit(range(len(series)), series, 1)[0]
        trend_df = pd.DataFrame({
            'util_trend_full_history': grp['credit_utilization_ratio'].apply(get_trend_slope),
            'repay_score_trend_full_history': grp['repayment_history_score'].apply(get_trend_slope)
        })
        volatility_df = grp.agg({'credit_utilization_ratio': 'std', 'repayment_history_score': 'std'}).rename(columns={'credit_utilization_ratio': 'util_volatility', 'repayment_history_score': 'repay_score_volatility'})
        features_only_df = state_df.join(trend_df).join(volatility_df)
        target_df = self.panel_df.groupby('customer_id')[self.target_col].first()
        final_df = features_only_df.join(target_df)
        final_df.dropna(subset=[self.target_col], inplace=True)
        final_df = final_df.fillna(0)
        print(f"Generated {len(final_df.columns)-1} powerful features.")
        return final_df

    def _prepare_data(self):
        # This method is correct
        self.df_fe = self._feature_engineering()
        self.y = self.le.fit_transform(self.df_fe[self.target_col])
        self.X = self.df_fe.drop(columns=[self.target_col])
        
    def _objective(self, trial):
        params = {
            'objective': 'multiclass', 'num_class': len(self.le.classes_),
            'metric': 'multi_logloss', 'verbosity': -1, 'boosting_type': 'gbdt',
            'n_estimators': 2000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 31, 150),
            'max_depth': trial.suggest_int('max_depth', 5, 10),
            
            # --- THIS IS THE FIX ---
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 150),
            # --- END FIX ---

            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0),
        }
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in skf.split(self.X, self.y):
            model = lgb.LGBMClassifier(**params)
            model.fit(self.X.iloc[train_idx], self.y[train_idx], 
                      eval_set=[(self.X.iloc[val_idx], self.y[val_idx])],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
            preds = model.predict(self.X.iloc[val_idx])
            scores.append(f1_score(self.y[val_idx], preds, average='weighted'))
        return np.mean(scores)

    def run(self, n_trials=50):
        # This method is correct
        self._prepare_data()
        print("\n--- Running Final Hyperparameter Tuning ---")
        study = optuna.create_study(direction='maximize')
        study.optimize(self._objective, n_trials=n_trials)
        best_params = study.best_params
        print(f"Best CV F1-score: {study.best_value}")
        
        self.train_final_model(best_params)

    def train_final_model(self, best_params):
        # This method is correct
        print("\nTraining final model...")
        final_params = best_params.copy()
        final_params.update({
            'objective': 'multiclass', 'num_class': len(self.le.classes_),
            'n_estimators': 4000, 'verbosity': -1,
        })
        self.model = lgb.LGBMClassifier(**final_params)
        self.model.fit(self.X, self.y)
        joblib.dump(self.model, self.output_path / "model.pkl")