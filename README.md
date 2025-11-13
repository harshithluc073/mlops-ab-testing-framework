# MLOps A/B Testing Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An offline and automated framework for simulating A/B tests between different ML model versions, enabling data-driven and statistically sound evaluation before deployment.

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
