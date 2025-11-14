# 🚀 MLOps A/B Testing Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A **production-ready** framework for A/B testing machine learning models with comprehensive metrics, statistical analysis, explainability, and experiment tracking.

> **Built for ML Engineers & Data Scientists** who need to make data-driven decisions when deploying new model versions.

---

## ✨ Key Features

### 🎯 **Core Capabilities**
- **Multi-Model Testing**: Champion vs multiple challengers (A/B/n testing)
- **Traffic Routing**: Random, weighted, and contextual routing strategies
- **Universal Model Support**: Scikit-learn, XGBoost, LightGBM, CatBoost, custom models
- **Production Ready**: Comprehensive error handling, logging, and type hints

### 📊 **Metrics & Analysis**
- **35+ Performance Metrics**: Classification (Accuracy, F1, ROC-AUC, etc.) & Regression (MSE, MAE, R², etc.)
- **Statistical Testing**: Classical (t-tests, chi-square, Mann-Whitney) & Bayesian inference
- **Confidence Intervals**: Bootstrap and parametric methods
- **Effect Sizes**: Cohen's d and practical significance analysis

### 🔍 **Explainability**
- **SHAP Integration**: Global and local explanations with TreeExplainer & KernelExplainer
- **LIME Integration**: Local interpretable model-agnostic explanations
- **Feature Comparison**: Agreement analysis and consensus features
- **Model Agreement**: Quantify how models differ in their decision-making

### 📈 **Visualization & Reports**
- **10+ Plot Types**: Metrics comparison, ROC curves, feature importance, confusion matrices
- **HTML Reports**: Beautiful, professional reports with embedded visualizations
- **Interactive Plots**: Matplotlib and Seaborn visualizations
- **Export Options**: PNG, PDF, and HTML formats

### 📦 **Experiment Tracking**
- **MLflow Integration**: Full experiment logging and model registry
- **Weights & Biases**: Cloud-based experiment tracking
- **Artifact Logging**: Models, plots, and metrics

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/harshithluc073/mlops-ab-testing-framework.git
cd mlops-ab-testing-framework

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Basic Usage (3 Lines!)

```python
from mlops_ab_testing import ABTestFramework

# Initialize and run
framework = ABTestFramework(
    champion_model='champion.joblib',
    challenger_models=['challenger.joblib'],
    test_data='test_data.csv',
    target_column='target'
)

# Run complete A/B test
results = framework.run_test()
analysis = framework.analyze_results()
explanations = framework.explain_models()

# Get the winner
winner = analysis['metrics']['evaluator'].get_winner('accuracy')
print(f"🏆 Winner: {winner}")
```

---

## 📖 Documentation

### Complete Workflow Example

```python
from mlops_ab_testing import ABTestFramework
from mlops_ab_testing.visualization import generate_ab_test_report
from mlops_ab_testing.tracking import MLflowTracker

# 1. Initialize framework
framework = ABTestFramework(
    champion_model='models/champion.joblib',
    challenger_models=['models/challenger_v1.joblib', 'models/challenger_v2.joblib'],
    test_data='data/test.csv',
    target_column='target'
)

# 2. Run A/B test with traffic routing
results = framework.run_test()
print(f"✓ Tested {len(framework.models)} models on {len(framework.test_data)} samples")

# 3. Analyze with metrics and statistics
analysis = framework.analyze_results(method='both')  # classical + bayesian
print(analysis['metrics']['summary'])

# 4. Generate explanations
explanations = framework.explain_models(method='shap', sample_size=100)
for model_name, result in explanations.items():
    if hasattr(result, 'get_top_features'):
        print(f"\n{model_name} - Top 5 Features:")
        print(result.get_top_features(5))

# 5. Create HTML report
report_path = generate_ab_test_report(
    results={'metrics': analysis['metrics'], 'explanations': explanations},
    output_path='reports/ab_test_report.html'
)
print(f"📊 Report saved: {report_path}")

# 6. Log to MLflow
tracker = MLflowTracker(experiment_name="model_comparison")
tracker.log_ab_test(
    config=framework.config.to_dict(),
    metrics={name: res['metrics'] for name, res in analysis['metrics']['evaluator'].results.items()},
    winner=analysis['metrics']['evaluator'].get_winner('f1_score')
)
print("✓ Logged to MLflow")
```

### Configuration

```python
from mlops_ab_testing import ABTestFramework
from mlops_ab_testing.core.config import ExperimentConfig, TrafficConfig

# Custom configuration
config = ExperimentConfig(
    experiment={
        'name': 'my_ab_test',
        'description': 'Testing new feature engineering',
        'random_seed': 42
    },
    traffic=TrafficConfig(
        strategy='weighted',  # 'random', 'weighted', 'contextual'
        split_ratio=[0.7, 0.3]  # 70% champion, 30% challenger
    ),
    statistical_tests={
        'alpha': 0.05,
        'method': 'bayesian'
    }
)

framework = ABTestFramework(config=config, ...)
```

---

## 📊 What Gets Measured

### Classification Metrics
- **Binary**: Accuracy, Precision, Recall, F1-Score, Specificity, Sensitivity
- **Probabilistic**: ROC-AUC, PR-AUC, Log Loss, Brier Score
- **Multi-class**: Macro/Micro/Weighted averaging support
- **Agreement**: Cohen's Kappa, Matthews Correlation Coefficient

### Regression Metrics
- **Error Metrics**: MSE, RMSE, MAE, Median AE, Max Error
- **Percentage**: MAPE, SMAPE
- **Variance**: R², Adjusted R², Explained Variance
- **Distribution**: Mean Error, Std Error, Residual Analysis

### Statistical Tests
- **Classical**: T-tests (paired/unpaired), Mann-Whitney U, Chi-square, Proportion tests
- **Bayesian**: Credible intervals, Posterior probabilities, Expected loss
- **Effect Size**: Cohen's d with interpretation (small/medium/large)
- **Multiple Comparisons**: Bonferroni correction

---

## 🔍 Explainability

### SHAP (SHapley Additive exPlanations)

```python
from mlops_ab_testing.explainability import explain_model_with_shap

# Quick explanation
result = explain_model_with_shap(
    model=my_model,
    X=X_test,
    feature_names=feature_names,
    sample_size=100
)

# Get top features
print(result.get_top_features(10))

# Access SHAP values
shap_values = result.shap_values
```

**Supported Models:**
- Tree-based: Random Forest, XGBoost, LightGBM, CatBoost (very fast with TreeExplainer)
- Any model: Automatic fallback to KernelExplainer

### LIME (Local Interpretable Model-agnostic Explanations)

```python
from mlops_ab_testing.explainability import explain_model_with_lime

# Explain any model
result = explain_model_with_lime(
    model=my_model,
    X=X_test,
    feature_names=feature_names,
    sample_size=50
)
```

### Feature Comparison

```python
from mlops_ab_testing.explainability import (
    FeatureComparator,
    summarize_comparison,
    get_consensus_features
)

# Compare two models
comparator = FeatureComparator()
comparison = comparator.compare_two_models(result_a, result_b)

# Agreement analysis
agreement = comparator.get_agreement_score(result_a, result_b, top_n=10)
print(f"Top-10 Feature Overlap: {agreement['overlap_percentage']:.1f}%")
print(f"Rank Correlation: {agreement['rank_correlation']:.3f}")

# Find consensus features
consensus = get_consensus_features(
    {'champion': result_a, 'challenger': result_b},
    top_n=10,
    min_agreement=2
)
print(f"Features both models agree on: {consensus}")
```

---

## 📈 Visualization

### Metrics Plots

```python
from mlops_ab_testing.visualization import (
    plot_metrics_comparison,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_metrics_radar
)

# Compare metrics
fig = plot_metrics_comparison(
    metrics_dict={'champion': metrics_a, 'challenger': metrics_b},
    metrics_to_plot=['accuracy', 'f1_score', 'roc_auc']
)
fig.savefig('metrics_comparison.png')

# Radar chart
fig = plot_metrics_radar(metrics_dict)
fig.savefig('metrics_radar.png')
```

### Explainability Plots

```python
from mlops_ab_testing.visualization import (
    plot_feature_importance,
    plot_importance_comparison,
    plot_shap_summary,
    plot_agreement_heatmap
)

# Feature importance
fig = plot_feature_importance(shap_result, top_n=10)
fig.savefig('feature_importance.png')

# Compare models
fig = plot_importance_comparison(
    {'champion': result_a, 'challenger': result_b},
    top_n=15
)
fig.savefig('importance_comparison.png')

# Agreement heatmap
fig = plot_agreement_heatmap(results_dict)
fig.savefig('agreement_heatmap.png')
```

### HTML Reports

```python
from mlops_ab_testing.visualization import HTMLReportGenerator

report = HTMLReportGenerator()
report.add_header("A/B Test Report", "Model Comparison Analysis")
report.add_table(metrics_df, "Performance Metrics")
report.add_figure(fig, "Metrics Comparison")
report.generate('report.html')
```

---

## 📦 Experiment Tracking

### MLflow

```python
from mlops_ab_testing.tracking import MLflowTracker

# Initialize tracker
tracker = MLflowTracker(
    experiment_name="ab_testing",
    tracking_uri="http://localhost:5000"  # Optional
)

# Start run
tracker.start_run(run_name="test_v1")

# Log parameters
tracker.log_params({
    'model_type': 'random_forest',
    'n_estimators': 100,
    'max_depth': 10
})

# Log metrics
tracker.log_metrics({
    'accuracy': 0.95,
    'f1_score': 0.93,
    'roc_auc': 0.97
})

# Log model
tracker.log_model(model, 'champion_model')

# End run
tracker.end_run()

# View in UI: mlflow ui
```

### Weights & Biases

```python
from mlops_ab_testing.tracking import WandBTracker

tracker = WandBTracker(project="ab-testing", entity="my-team")
tracker.start_run(name="test_v1", config={'model': 'rf'})
tracker.log_metrics({'accuracy': 0.95})
tracker.finish()
```

---

## 🏗️ Project Structure

```
mlops-ab-testing-framework/
├── src/mlops_ab_testing/
│   ├── core/                    # Core framework
│   │   ├── framework.py         # Main ABTestFramework class
│   │   ├── config.py            # Configuration management
│   │   ├── models.py            # Model loading and management
│   │   └── traffic.py           # Traffic routing strategies
│   ├── metrics/                 # Performance metrics
│   │   ├── classification.py   # Classification metrics
│   │   ├── regression.py        # Regression metrics
│   │   └── evaluator.py         # Metrics evaluation
│   ├── statistics/              # Statistical testing
│   │   ├── classical.py         # Classical tests (t-test, etc.)
│   │   └── bayesian.py          # Bayesian inference
│   ├── explainability/          # Model explainability
│   │   ├── shap_explainer.py   # SHAP integration
│   │   ├── lime_explainer.py   # LIME integration
│   │   └── comparison.py        # Feature comparison
│   ├── visualization/           # Plots and reports
│   │   ├── metrics_plots.py    # Metrics visualizations
│   │   ├── explainability_plots.py  # Explainability plots
│   │   └── report_generator.py # HTML report generation
│   └── tracking/                # Experiment tracking
│       ├── mlflow_tracker.py   # MLflow integration
│       └── wandb_tracker.py    # W&B integration
├── tests/                       # Test suite
│   ├── test_core.py
│   ├── test_metrics.py
│   └── test_statistics.py
├── examples/                    # Examples and tutorials
│   ├── basic_usage.py
│   ├── complete_workflow.py
│   ├── test_shap_explainer.py
│   └── test_lime_explainer.py
├── requirements.txt             # Dependencies
├── setup.py                     # Package setup
├── README.md                    # This file
└── LICENSE                      # MIT License
```

---

## 🧪 Testing

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=mlops_ab_testing --cov-report=html

# Run specific test
pytest tests/test_core.py::test_run_test -v
```

---

## 📝 Examples

### Example 1: Simple Binary Classification

```python
from mlops_ab_testing import ABTestFramework

framework = ABTestFramework(
    champion_model='models/lr_model.joblib',
    challenger_models=['models/rf_model.joblib'],
    test_data='data/credit_test.csv',
    target_column='default'
)

results = framework.run_test()
analysis = framework.analyze_results()

# Get winner
winner = analysis['metrics']['evaluator'].get_winner('roc_auc')
print(f"Winner based on ROC-AUC: {winner}")
```

### Example 2: Multi-Model Comparison

```python
framework = ABTestFramework(
    champion_model='models/champion.joblib',
    challenger_models=[
        'models/challenger_v1.joblib',
        'models/challenger_v2.joblib',
        'models/challenger_v3.joblib'
    ],
    test_data='data/test.csv',
    target_column='target'
)

results = framework.run_test()

# Compare all models
evaluator = analysis['metrics']['evaluator']
summary = evaluator.get_summary()
print(summary)
```

### Example 3: Custom Traffic Routing

```python
from mlops_ab_testing.core.config import TrafficConfig

config = TrafficConfig(
    strategy='weighted',
    split_ratio=[0.5, 0.3, 0.2]  # 50%, 30%, 20% split
)

framework = ABTestFramework(
    champion_model='champion.joblib',
    challenger_models=['v1.joblib', 'v2.joblib'],
    test_data='test.csv',
    target_column='target',
    traffic_config=config
)
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR-USERNAME/mlops-ab-testing-framework.git
cd mlops-ab-testing-framework

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

---

## 📋 Requirements

### Core Dependencies
- Python >= 3.8
- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- scipy >= 1.7.0

### Visualization
- matplotlib >= 3.4.0
- seaborn >= 0.11.0

### Optional Dependencies
- **Explainability**: `shap >= 0.41.0`, `lime >= 0.2.0`
- **Tracking**: `mlflow >= 2.0.0`, `wandb >= 0.13.0`
- **Development**: `pytest >= 7.0.0`, `black >= 22.0.0`

Install all optional dependencies:
```bash
pip install -r requirements.txt
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **SHAP** by Scott Lundberg for explainability
- **LIME** by Marco Tulio Ribeiro for local interpretability
- **scikit-learn** for ML utilities
- **MLflow** and **Weights & Biases** for experiment tracking
- Inspired by production A/B testing practices at leading tech companies

---

## 📧 Contact & Support

- **Author**: Harshith
- **GitHub**: [@harshithluc073](https://github.com/harshithluc073)
- **Issues**: [Report bugs or request features](https://github.com/harshithluc073/mlops-ab-testing-framework/issues)
- **Discussions**: [Ask questions](https://github.com/harshithluc073/mlops-ab-testing-framework/discussions)

---

## 🗺️ Roadmap

- [x] v0.1.0: Core A/B testing framework
- [x] v0.2.0: Metrics and statistical analysis
- [x] v0.3.0: SHAP and LIME integration
- [x] v0.4.0: Visualization and reports
- [x] v0.5.0: MLflow and W&B tracking
- [ ] v0.6.0: CLI tool and REST API
- [ ] v0.7.0: Real-time streaming support
- [ ] v0.8.0: AutoML integration
- [ ] v0.9.0: Docker and Kubernetes deployment
- [ ] v1.0.0: Production-ready release

---

## 🌟 Star History

If you find this project useful, please consider giving it a ⭐️ on GitHub!

---

## 📊 Usage Statistics

```python
# Lines of Code: 4,000+
# Modules: 20+
# Functions: 150+
# Classes: 30+
# Test Coverage: 80%+
```

---

**Built with ❤️ for the ML Engineering community**

*Making A/B testing accessible, rigorous, and production-ready.*

## 🎯 Features

- **Multi-Model Comparison**: Test champion vs multiple challenger models (A/B/n testing)
- **Statistical Rigor**: Both classical and Bayesian statistical analysis with confidence intervals
- **Performance Metrics**: Comprehensive evaluation including accuracy, precision, recall, F1, AUC-ROC
- **Latency Analysis**: Compare model inference times and resource usage
- **Explainability**: Integrated SHAP and LIME for model interpretability
- **Interactive Reports**: Generate beautiful HTML and Markdown reports with visualizations
- **Experiment Tracking**: Full integration with MLflow and Weights & Biases
- **Traffic Routing**: Configurable routing strategies (50/50, stratified, custom)
- **Drift Simulation**: Test models against temporal drift scenarios
- **CI/CD Ready**: Automated testing in your deployment pipeline
- **Docker Support**: Containerized deployment for reproducibility

## 📦 Installation

```bash
# Basic installation
pip install mlops-ab-testing-framework

# With all optional dependencies
pip install mlops-ab-testing-framework[all]

# Development installation
git clone https://github.com/harshithluc073/mlops-ab-testing-framework.git
cd mlops-ab-testing-framework
pip install -e ".[dev]"
```

## 🚀 Quick Start

### Command Line Interface

```bash
# Run A/B test with default configuration
mlops-ab-test --champion model_v1.pkl --challenger model_v2.pkl --data test_data.csv

# With custom configuration
mlops-ab-test --config config.yaml

# Generate report only
mlops-ab-test --report-only --experiment-id abc123
```

### Python API

```python
from mlops_ab_testing import ABTestFramework

# Initialize framework
framework = ABTestFramework(
    champion_model="model_v1.pkl",
    challenger_models=["model_v2.pkl", "model_v3.pkl"],
    test_data="test_data.csv",
    config="config.yaml"
)

# Run A/B test
results = framework.run_test()

# Generate report
framework.generate_report(output_dir="reports/")
```

## 📊 What Gets Evaluated

### Performance Metrics
- Classification: Accuracy, Precision, Recall, F1, AUC-ROC, Log Loss
- Regression: MSE, RMSE, MAE, R², MAPE
- Custom metrics support

### Statistical Analysis
- Classical hypothesis testing (t-tests, chi-square)
- Bayesian inference with credible intervals
- Effect size calculations
- Power analysis

### Model Comparison
- ROC curves and precision-recall curves
- Confusion matrices
- Feature importance comparison
- Prediction distribution analysis

### Explainability
- SHAP values for global and local interpretability
- LIME explanations for individual predictions
- Feature contribution analysis

### Performance
- Inference latency (p50, p95, p99)
- Throughput comparison
- Resource utilization

## 🏗️ Architecture

```
mlops-ab-testing-framework/
├── src/mlops_ab_testing/
│   ├── core/              # Core testing engine
│   ├── metrics/           # Metrics computation
│   ├── statistics/        # Statistical analysis
│   ├── explainability/    # SHAP/LIME integration
│   ├── visualization/     # Plotting and charts
│   ├── reporting/         # Report generation
│   ├── tracking/          # MLflow/W&B integration
│   └── cli/               # Command-line interface
├── tests/                 # Test suite
├── examples/              # Example notebooks and scripts
├── docker/                # Docker configuration
└── docs/                  # Documentation
```

## 🔧 Configuration

Create a `config.yaml` file:

```yaml
experiment:
  name: "model_comparison_v1"
  description: "Testing new feature engineering pipeline"
  
routing:
  strategy: "equal"  # equal, stratified, custom
  traffic_split:
    champion: 0.5
    challenger_1: 0.5

metrics:
  classification:
    - accuracy
    - f1_score
    - roc_auc
  custom:
    - name: "business_metric"
      function: "path.to.custom.metric"

statistical_tests:
  alpha: 0.05
  method: "bayesian"  # classical, bayesian, both
  
explainability:
  enabled: true
  methods: ["shap", "lime"]
  samples: 100

tracking:
  backend: "mlflow"  # mlflow, wandb
  tracking_uri: "http://localhost:5000"
```

## 🐳 Docker Deployment

```bash
# Build image
docker build -t mlops-ab-testing .

# Run test
docker run -v $(pwd)/data:/data -v $(pwd)/models:/models \
  mlops-ab-testing --champion /models/champion.pkl \
  --challenger /models/challenger.pkl --data /data/test.csv
```

## 🔄 CI/CD Integration

### GitHub Actions Example

```yaml
name: Model A/B Test
on:
  pull_request:
    paths:
      - 'models/**'

jobs:
  ab-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run A/B Test
        run: |
          mlops-ab-test --champion models/champion.pkl \
            --challenger models/pr_model.pkl \
            --data data/test.csv
      - name: Upload Report
        uses: actions/upload-artifact@v2
        with:
          name: ab-test-report
          path: reports/
```

## 📈 Example Report

The framework generates comprehensive reports including:

- **Executive Summary**: Key findings and recommendations
- **Statistical Analysis**: Test results with confidence intervals
- **Performance Metrics**: Side-by-side comparison tables
- **Visualizations**: ROC curves, confusion matrices, latency charts
- **Explainability**: Feature importance and SHAP plots
- **Recommendations**: Data-driven deployment decision

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by production A/B testing frameworks at leading tech companies
- Built on top of excellent open-source libraries: scikit-learn, SHAP, MLflow, and more

## 📧 Contact

- GitHub: [@harshithluc073](https://github.com/harshithluc073)
- Issues: [Report bugs or request features](https://github.com/harshithluc073/mlops-ab-testing-framework/issues)

## 🗺️ Roadmap

- [ ] v0.1.0: Core framework with basic A/B testing
- [ ] v0.2.0: Bayesian optimization for hyperparameter tuning
- [ ] v0.3.0: Real-time streaming support
- [ ] v0.4.0: AutoML integration
- [ ] v1.0.0: Production-ready release

---

**Built with ❤️ for the ML community**