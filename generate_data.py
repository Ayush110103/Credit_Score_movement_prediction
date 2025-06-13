# generate_data.py
import pandas as pd
import numpy as np
from pathlib import Path

class DefinitiveDataGenerator:
    """
    Generates a causally sound dataset where a customer's hidden archetype
    drives both their historical behavior and their future outcome.
    This creates a strong, learnable signal for the model.
    """
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)

    def generate(self, num_customers=30000, num_months=12):
        print("Generating DEFINITIVE dataset with archetype-driven causality...")
        
        archetypes = {
            'improving': {'fraction': 0.25, 'target': 'increase', 'util_mu': 0.4, 'util_trend': -0.02, 'repay_mu': 70, 'repay_trend': 0.5},
            'worsening': {'fraction': 0.25, 'target': 'decrease', 'util_mu': 0.6, 'util_trend': 0.03, 'repay_mu': 60, 'repay_trend': -0.6},
            'stable_good': {'fraction': 0.25, 'target': 'stable', 'util_mu': 0.2, 'util_trend': 0.0, 'repay_mu': 90, 'repay_trend': 0.0},
            'stable_bad': {'fraction': 0.25, 'target': 'stable', 'util_mu': 0.8, 'util_trend': 0.0, 'repay_mu': 50, 'repay_trend': 0.0},
        }

        # --- Assign Archetype and Static Features ---
        static_df = pd.DataFrame({'customer_id': range(num_customers)})
        archetype_list, target_list = [], []
        for arch, props in archetypes.items():
            count = int(num_customers * props['fraction'])
            archetype_list.extend([arch] * count)
            target_list.extend([props['target']] * count)

        # Ensure we have the correct number of customers
        while len(archetype_list) < num_customers: 
            archetype_list.append('stable_good')
            target_list.append('stable')
            
        perm = self.rng.permutation(num_customers)
        static_df['archetype'] = np.array(archetype_list)[perm]
        static_df['target_credit_score_movement'] = np.array(target_list)[perm]
        static_df['age'] = self.rng.integers(22, 65, num_customers)

        # --- Generate Time-Series Data based on Archetype ---
        customer_data = []
        for _, row in static_df.iterrows():
            arch_props = archetypes[row['archetype']]
            
            # Initialize features around the archetype's mean
            util = arch_props['util_mu'] + self.rng.normal(0, 0.1)
            repay_score = arch_props['repay_mu'] + self.rng.normal(0, 5)
            
            for month in range(1, num_months + 1):
                # Evolve features based on the archetype's defined trends + noise
                util += arch_props['util_trend'] + self.rng.normal(0, 0.03)
                repay_score += arch_props['repay_trend'] + self.rng.normal(0, 0.75)
                
                customer_data.append({
                    'customer_id': row['customer_id'],
                    'month': month,
                    'age': row['age'],
                    'credit_utilization_ratio': np.clip(util, 0.01, 0.99),
                    'repayment_history_score': np.clip(repay_score, 0, 100),
                    # Other features can be simplified as they are less important than the core signals
                    'dpd_last_3_months': 30 if repay_score < 60 and self.rng.random() < 0.2 else 0,
                    'monthly_income': self.rng.integers(50000, 80000)
                })
        
        panel_df = pd.DataFrame(customer_data)
        
        final_df = panel_df.merge(static_df[['customer_id', 'target_credit_score_movement']], on='customer_id', how='left')
  
        noise_mask = self.rng.random(len(final_df)) < 0.05 
        final_df.loc[noise_mask, 'target_credit_score_movement'] = self.rng.permutation(final_df.loc[noise_mask, 'target_credit_score_movement'])
        
        print("Definitive causally-driven dataset generated.")
        return final_df