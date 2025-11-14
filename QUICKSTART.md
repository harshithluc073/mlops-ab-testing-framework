# ⚡ Quick Start Guide

Get up and running with the MLOps A/B Testing Framework in 5 minutes!

---

## 📥 Installation (1 minute)

```bash
# Clone repository
git clone https://github.com/harshithluc073/mlops-ab-testing-framework.git
cd mlops-ab-testing-framework

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install
pip install -r requirements.txt
pip install -e .
```

---

## 🎯 Your First A/B Test (3 minutes)

### Step 1: Prepare Your Data

```python
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Create sample data
X, y = make_classification(n_samples=500, n_features=10, random_state=42)
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
df['target'] = y

# Save test data
df.to_csv('test_data.csv', index=False)
```

### Step 2: Train Models

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib

# Train champion
champion = RandomForestClassifier(n_estimators=50, random_state=42)
champion.fit(X, y)
joblib.dump(champion, 'champion.joblib')

# Train challenger
challenger = GradientBoostingClassifier(n_estimators=50, random_state=42)
challenger.fit(X, y)
joblib.dump(challenger, 'challenger.joblib')
```

### Step 3: Run A/B Test

```python
from mlops_ab_testing import ABTestFramework

# Initialize framework
framework = ABTestFramework(
    champion_model='champion.joblib',
    challenger_models=['challenger.joblib'],
    test_data='test_data.csv',
    target_column='target'
)

# Run test (1 line!)
results = framework.run_test()
```

### Step 4: Analyze Results

```python
# Get metrics
analysis = framework.analyze_results()
print(analysis['metrics']['summary'])

# Get winner
winner = analysis['metrics']['evaluator'].get_winner('accuracy')
print(f"🏆 Winner: {winner}")
```

---

## 🚀 Next Steps

### Add Explainability

```python
# Generate SHAP explanations
explanations = framework.explain_models(method='shap', sample_size=100)

# View top features
for model_name, result in explanations.items():
    if hasattr(result, 'get_top_features'):
        print(f"\n{model_name} Top Features:")
        print(result.get_top_features(5))
```

### Create Visualizations

```python
from mlops_ab_testing.visualization import plot_metrics_comparison

# Get metrics dict
metrics_dict = {
    name: res['metrics'] 
    for name, res in analysis['metrics']['evaluator'].results.items()
}

# Plot comparison
fig = plot_metrics_comparison(metrics_dict)
fig.savefig('metrics_comparison.png')
```

### Generate Report

```python
from mlops_ab_testing.visualization import generate_ab_test_report

report_path = generate_ab_test_report(
    {'metrics': analysis['metrics']},
    'ab_test_report.html'
)
print(f"Report saved: {report_path}")
```

### Log to MLflow

```python
from mlops_ab_testing.tracking import MLflowTracker

tracker = MLflowTracker(experiment_name="my_test")
tracker.log_ab_test(
    config={'experiment': {'name': 'my_test'}},
    metrics=metrics_dict,
    winner=winner
)
```

---

## 📚 Learn More

- **Full Documentation**: See [README.md](README.md)
- **Examples**: Check `examples/` folder
- **Advanced Usage**: See `examples/complete_workflow.py`

---

## 🆘 Troubleshooting

### SHAP Import Error

```bash
pip install shap
```

### LIME Import Error

```bash
pip install lime
```

### MLflow Import Error

```bash
pip install mlflow
```

### Windows Pickle Issues

The framework handles Windows compatibility automatically with fallback mechanisms.

---

## ✅ Verify Installation

```python
# Test import
from mlops_ab_testing import ABTestFramework
print("✓ Framework imported successfully!")

# Check version
import mlops_ab_testing
print(f"Version: {mlops_ab_testing.__version__}")
```

---

## 🎉 You're Ready!

You now know how to:
- ✅ Run A/B tests
- ✅ Calculate metrics
- ✅ Analyze results
- ✅ Generate explanations
- ✅ Create visualizations
- ✅ Log experiments

**Dive deeper with the full documentation!** 📖