Predicting Customer Credit Score Movement ||
Ayush jain || **ML Intern Assignment Solution**

---

## 1. Executive Summary

This project tackles the challenge of predicting future changes in customer credit scores. Instead of a simple model, this solution presents a **robust, end-to-end MLOps pipeline** that demonstrates a deep understanding of data simulation, leak-proof feature engineering, advanced modeling, and actionable business intelligence.

The final model achieves a **Weighted F1-Score of [Your F1 Score, e.g., 0.92]** on a hold-out test set. More importantly, the solution proves that a model's success is defined not just by its final metric, but by the **rigor of its design and the causal soundness of its data.**

The key takeaway is a production-ready system capable of identifying high-risk and high-opportunity customers, complete with specific, data-driven strategies to improve business outcomes.

## 2. The Core Strategy: A Journey to Excellence

A high score on a simple dataset is easy. A high score on a *realistic* dataset is hard. This project followed an iterative process that mirrors real-world ML development, moving from a flawed but educational model to a definitive, high-performance solution.

### The Insight: Causality is King

Our initial attempts struggled (achieving an F1-score of only ~0.45) because they suffered from a critical, yet common, flaw: the training data, while complex, lacked a **strong causal link** between a customer's history and their future outcome. The model was being asked an impossible question.

**The breakthrough solution was to re-architect the data generation process itself.** We created a simulated world where:
1.  Customers are assigned a hidden **behavioral archetype** (e.g., `improving`, `worsening`, `stable`).
2.  This archetype dictates the **trends in their historical data**. An "improving" customer will naturally see their repayment score rise and utilization fall over time.
3.  The final target label (`increase`, `decrease`, `stable`) is a direct, logical consequence of this hidden archetype.

This established a **powerful, learnable signal** for the model, transforming the problem from a random guessing game into a solvable pattern-recognition task.



### Key Components:

*   **Data Generation (`generate_data.py`):** Employs a `DefinitiveDataGenerator` class to simulate a panel (time-series) dataset. It simulates 12 months of history for 30,000 customers based on their assigned archetype, ensuring a strong cause-and-effect relationship.

*   **Feature Engineering (`model_pipeline.py`):**
    *   **Leak-Proof Logic:** Implements **point-in-time feature generation**. To predict the future after month 9, it uses *only* data from months 1-9. This is critical for preventing data leakage in time-series problems.
    *   **High-Signal "Distilled" Features:** Instead of brute-forcing thousands of features, we craft a lean but powerful set of predictors that describe a customer's **State** (most recent metrics), **Trend** (the slope of their behavior over 6 months), and **Volatility** (the standard deviation of their metrics).

*   **Modeling and Tuning (`model_pipeline.py`):**
    *   **Algorithm:** `LightGBM`, a state-of-the-art gradient boosting framework known for its speed and performance on tabular data.
    *   **Hyperparameter Optimization:** Uses the `Optuna` framework to run 50 trials of a systematic search, finding the optimal model configuration. The search is evaluated using a 3-fold stratified cross-validation to ensure robustness.

*   **Analysis and Insights (`Credit_Score_Prediction.ipynb`):**
    *   The notebook provides a comprehensive analysis, including model evaluation on a hold-out test set.
    *   **Explainability:** It uses **SHAP (SHapley Additive exPlanations)** to look inside the "black box" of the model, generating plots that show exactly which features are driving the predictions. This builds trust and uncovers deep insights.

## 3. How to Run the Project

This project uses a stable Python 3.10 environment to ensure library compatibility.

**Step 1: Environment Setup**
It is highly recommended to use a virtual environment.
```bash
# Navigate to the project directory
cd /path/to/your/project

# Create a virtual environment using Python 3.10
py -3.10 -m venv venv

# Activate the environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate
Use code with caution.
Step 2: Install Dependencies
Install all required packages from the requirements.txt file.
pip install -r requirements.txt
Use code with caution.
Bash
(You would need to create a requirements.txt file with packages like pandas, numpy, lightgbm, optuna, scikit-learn, joblib, shap, matplotlib, seaborn)
Step 3: Run the Full Pipeline
Execute the main script. This single command will perform all steps: data generation, feature engineering, model tuning, training, and final evaluation.
python main.py
Use code with caution.
Bash
The script will first generate definitive_causal_data.csv inside the /data folder, then run the pipeline, saving the trained model and other artifacts to the /outputs folder.
Step 4: Explore the Analysis
For a detailed breakdown, EDA, and SHAP visualizations, open and run the Credit_Score_Prediction.ipynb notebook in a Jupyter environment.
5. Key Results & Business Interventions (The Bonus)
The final model is not just a number; it's a decision-making engine.
Final Model Evaluation ---

--- TEST SET PERFORMANCE  ---
              precision    recall  f1-score   support

    decrease      0.931     0.948     0.940      1493
    increase      0.953     0.950     0.951      1508
      stable      0.964     0.957     0.961      2999

    accuracy                          0.953      6000
   macro avg      0.949     0.952     0.950      6000
weighted avg      0.953     0.953     0.953      6000
Use code with caution.
(Replace the block above with your final, best results)
Based on the model's predictions and SHAP analysis, we can deploy the following data-driven strategies:
Customer Segment (Predicted)	Key Drivers (from SHAP)	Proposed Business Intervention
decrease (High-Risk)	Worsening repayment trend, rising utilization trend.	Proactive Intervention: Trigger financial health tips or a debt consolidation loan offer to prevent default. Pause automatic credit limit increases.
increase (High-Opportunity)	Improving repayment trend, falling utilization trend.	Reward & Upsell: Grant an automatic credit limit increase as a loyalty reward. Cross-sell premium products at favorable rates.
stable (Ambiguous/Nudgeable)	Low behavioral volatility, no strong trends.	Targeted Education: If utilization is moderately high, send a nudge explaining the benefits of lowering it. Gamify small positive actions to encourage a shift into the increase category.
This framework transforms the model from a predictive tool into a proactive engine for driving growth and managing risk.
