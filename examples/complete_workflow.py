"""
Complete end-to-end example using all framework features.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


def create_complete_example():
    """Run complete A/B testing workflow."""
    print("="*70)
    print(" "*15 + "COMPLETE A/B TESTING WORKFLOW")
    print("="*70)
    
    # Step 1: Create sample data
    print("\n📊 Step 1: Creating sample data...")
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        random_state=42
    )
    
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    
    print(f"   ✓ Train: {len(train_df)}, Test: {len(test_df)}")
    
    # Create directories
    os.makedirs('data/complete', exist_ok=True)
    os.makedirs('models/complete', exist_ok=True)
    os.makedirs('reports/complete', exist_ok=True)
    
    # Save data
    train_df.to_csv('data/complete/train.csv', index=False)
    test_df.to_csv('data/complete/test.csv', index=False)
    
    # Step 2: Train models
    print("\n🤖 Step 2: Training models...")
    X_train, y_train = train_df[feature_names], train_df['target']
    
    # Champion model
    champion = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    champion.fit(X_train, y_train)
    joblib.dump(champion, 'models/complete/champion.joblib')
    print(f"   ✓ Champion trained (accuracy: {champion.score(X_train, y_train):.3f})")
    
    # Challenger model
    challenger = GradientBoostingClassifier(n_estimators=50, max_depth=8, random_state=42)
    challenger.fit(X_train, y_train)
    joblib.dump(challenger, 'models/complete/challenger.joblib')
    print(f"   ✓ Challenger trained (accuracy: {challenger.score(X_train, y_train):.3f})")
    
    # Step 3: Run A/B test
    print("\n🧪 Step 3: Running A/B test...")
    from mlops_ab_testing import ABTestFramework
    
    framework = ABTestFramework(
        champion_model='models/complete/champion.joblib',
        challenger_models=['models/complete/challenger.joblib'],
        test_data='data/complete/test.csv',
        target_column='target'
    )
    
    results = framework.run_test()
    print("   ✓ A/B test complete")
    
    # Step 4: Analyze with metrics & statistics
    print("\n📈 Step 4: Analyzing results...")
    analysis = framework.analyze_results(method='both')
    
    # Show metrics summary
    summary = analysis['metrics']['summary']
    print("\n   Metrics Summary:")
    print(summary.to_string(index=False))
    
    # Step 5: Generate explanations
    print("\n🔍 Step 5: Generating model explanations...")
    try:
        explanations = framework.explain_models(method='shap', sample_size=50)
        print("   ✓ SHAP explanations generated")
    except Exception as e:
        print(f"   ⚠ Explanation failed: {e}")
        explanations = {}
    
    # Step 6: Create visualizations
    print("\n📊 Step 6: Creating visualizations...")
    try:
        from mlops_ab_testing.visualization import plot_metrics_comparison
        
        metrics_dict = {
            name: result['metrics']
            for name, result in analysis['metrics']['evaluator'].results.items()
        }
        
        fig = plot_metrics_comparison(
            metrics_dict,
            metrics_to_plot=['accuracy', 'precision', 'recall', 'f1_score'],
            title="Champion vs Challenger Metrics"
        )
        fig.savefig('reports/complete/metrics_comparison.png', dpi=150)
        print("   ✓ Metrics plot saved")
    except Exception as e:
        print(f"   ⚠ Visualization failed: {e}")
    
    # Step 7: Generate HTML report
    print("\n📝 Step 7: Generating HTML report...")
    try:
        from mlops_ab_testing.visualization import generate_ab_test_report
        
        report_path = generate_ab_test_report(
            {'metrics': analysis['metrics']},
            'reports/complete/ab_test_report.html'
        )
        print(f"   ✓ Report saved: {report_path}")
    except Exception as e:
        print(f"   ⚠ Report generation failed: {e}")
    
    # Step 8: Log to MLflow (optional)
    print("\n📦 Step 8: Logging to MLflow...")
    try:
        from mlops_ab_testing.tracking import MLflowTracker
        
        tracker = MLflowTracker(experiment_name="complete_example")
        tracker.log_ab_test(
            config={'experiment': {'name': 'complete_test'}},
            metrics=metrics_dict,
            winner=analysis['metrics']['evaluator'].get_winner('accuracy')
        )
        print("   ✓ Logged to MLflow")
    except Exception as e:
        print(f"   ⚠ MLflow logging failed: {e}")
    
    # Final summary
    print("\n" + "="*70)
    print("✅ COMPLETE WORKFLOW FINISHED!")
    print("="*70)
    print("\n📂 Output Files:")
    print("   - Data: data/complete/")
    print("   - Models: models/complete/")
    print("   - Reports: reports/complete/")
    print("\n🏆 Winner: ", end="")
    try:
        winner = analysis['metrics']['evaluator'].get_winner('accuracy')
        print(f"{winner}")
    except:
        print("Unable to determine winner")
    
    print("\n💡 Next Steps:")
    print("   1. Review the HTML report: reports/complete/ab_test_report.html")
    print("   2. Check metrics plot: reports/complete/metrics_comparison.png")
    print("   3. View MLflow UI: mlflow ui")
    print("\n")


if __name__ == '__main__':
    create_complete_example()