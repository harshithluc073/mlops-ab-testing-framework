# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-01-XX

### Added
- **Core Framework**
  - ABTestFramework class for running A/B tests
  - Support for champion vs multiple challenger models
  - Traffic routing strategies (random, weighted, contextual)
  - Model loading for scikit-learn, joblib, and pickle formats
  
- **Metrics & Statistics**
  - 35+ performance metrics (classification and regression)
  - Classical statistical tests (t-tests, chi-square, Mann-Whitney)
  - Bayesian inference with credible intervals
  - Confidence intervals and effect size calculations
  
- **Explainability**
  - SHAP integration with TreeExplainer and KernelExplainer
  - LIME integration for local explanations
  - Feature importance comparison between models
  - Model agreement analysis
  - Consensus feature identification
  
- **Visualization**
  - 10+ plot types for metrics and explainability
  - HTML report generation with embedded visualizations
  - Support for matplotlib and seaborn
  
- **Experiment Tracking**
  - MLflow integration for experiment logging
  - Weights & Biases integration
  - Artifact and metric tracking
  
- **Testing**
  - Unit tests for core functionality
  - Unit tests for metrics calculation
  - Example scripts and complete workflow demo
  
- **Documentation**
  - Comprehensive README with examples
  - API documentation in docstrings
  - Contributing guidelines

### Fixed
- Numpy indexing issues in SHAP explainer for binary classification
- Boolean evaluation errors in LIME feature importance
- Pickle compatibility issues on Windows
- Unicode encoding handling in file operations

### Security
- No known security issues

---

## [Unreleased]

### Planned Features
- [ ] CLI tool for command-line usage
- [ ] REST API for remote model testing
- [ ] Interactive Plotly dashboards
- [ ] Permutation importance explainer
- [ ] Deep learning model support
- [ ] Docker containerization
- [ ] Real-time streaming support
- [ ] AutoML integration
- [ ] Kubernetes deployment support

---

## Version History

- **0.1.0** - Initial release with core features

---

**Legend:**
- `Added` - New features
- `Changed` - Changes in existing functionality
- `Deprecated` - Soon-to-be removed features
- `Removed` - Removed features
- `Fixed` - Bug fixes
- `Security` - Security fixes