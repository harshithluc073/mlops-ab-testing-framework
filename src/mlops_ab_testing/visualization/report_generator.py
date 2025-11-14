"""
HTML report generation module.

This module generates comprehensive HTML reports for A/B tests.
"""

import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
import base64
from io import BytesIO
import logging

logger = logging.getLogger(__name__)


class HTMLReportGenerator:
    """Generate HTML reports for A/B test results."""
    
    def __init__(self):
        """Initialize report generator."""
        self.sections = []
    
    def add_header(self, title: str, subtitle: Optional[str] = None):
        """Add report header."""
        html = f"""
        <div class="header">
            <h1>{title}</h1>
            {f'<p class="subtitle">{subtitle}</p>' if subtitle else ''}
            <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        """
        self.sections.append(html)
    
    def add_section(self, title: str, content: str):
        """Add a section to the report."""
        html = f"""
        <div class="section">
            <h2>{title}</h2>
            <div class="content">
                {content}
            </div>
        </div>
        """
        self.sections.append(html)
    
    def add_table(self, df: pd.DataFrame, title: Optional[str] = None):
        """Add a table to the report."""
        table_html = df.to_html(index=False, classes='data-table')
        
        if title:
            content = f"<h3>{title}</h3>{table_html}"
        else:
            content = table_html
        
        self.sections.append(content)
    
    def add_figure(self, fig, title: Optional[str] = None):
        """Add a matplotlib figure to the report."""
        # Convert figure to base64
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        buffer.close()
        
        html = f"""
        <div class="figure">
            {f'<h3>{title}</h3>' if title else ''}
            <img src="data:image/png;base64,{image_base64}" alt="{title or 'Figure'}">
        </div>
        """
        self.sections.append(html)
    
    def add_metrics_summary(self, metrics_dict: Dict[str, Dict[str, float]]):
        """Add metrics summary section."""
        df = pd.DataFrame(metrics_dict).T
        
        html = """
        <div class="metrics-summary">
            <h3>Metrics Summary</h3>
        """
        
        for model_name, metrics in metrics_dict.items():
            html += f"""
            <div class="model-card">
                <h4>{model_name}</h4>
                <ul>
            """
            for metric_name, value in list(metrics.items())[:5]:
                html += f"<li><strong>{metric_name}:</strong> {value:.4f}</li>"
            html += """
                </ul>
            </div>
            """
        
        html += "</div>"
        self.sections.append(html)
    
    def generate(self, output_path: str):
        """Generate and save HTML report."""
        css = """
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                margin-bottom: 30px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            .header h1 {
                margin: 0;
                font-size: 2.5em;
            }
            .subtitle {
                font-size: 1.2em;
                opacity: 0.9;
                margin: 10px 0;
            }
            .timestamp {
                font-size: 0.9em;
                opacity: 0.8;
            }
            .section {
                background: white;
                padding: 25px;
                margin-bottom: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .section h2 {
                color: #667eea;
                border-bottom: 2px solid #667eea;
                padding-bottom: 10px;
                margin-top: 0;
            }
            .section h3 {
                color: #555;
                margin-top: 20px;
            }
            .data-table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }
            .data-table th {
                background-color: #667eea;
                color: white;
                padding: 12px;
                text-align: left;
            }
            .data-table td {
                padding: 10px;
                border-bottom: 1px solid #ddd;
            }
            .data-table tr:hover {
                background-color: #f5f5f5;
            }
            .figure {
                text-align: center;
                margin: 30px 0;
            }
            .figure img {
                max-width: 100%;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            .metrics-summary {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }
            .model-card {
                background: #f9f9f9;
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid #667eea;
            }
            .model-card h4 {
                margin-top: 0;
                color: #667eea;
            }
            .model-card ul {
                list-style: none;
                padding: 0;
            }
            .model-card li {
                padding: 5px 0;
            }
        </style>
        """
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>A/B Test Report</title>
            {css}
        </head>
        <body>
            {''.join(self.sections)}
        </body>
        </html>
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report generated: {output_path}")
        return output_path


def generate_ab_test_report(
    results: Dict[str, Any],
    output_path: str = 'ab_test_report.html'
) -> str:
    """
    Generate comprehensive A/B test report.
    
    Args:
        results: Dictionary with test results, metrics, and analysis
        output_path: Path for output HTML file
        
    Returns:
        Path to generated report
    """
    report = HTMLReportGenerator()
    
    # Header
    report.add_header(
        "A/B Test Report",
        "Model Comparison Analysis"
    )
    
    # Executive Summary
    summary_content = """
    <p>This report provides a comprehensive analysis of the A/B test comparing multiple models.</p>
    <ul>
        <li>Statistical significance testing</li>
        <li>Performance metrics comparison</li>
        <li>Feature importance analysis</li>
    </ul>
    """
    report.add_section("Executive Summary", summary_content)
    
    # Metrics
    if 'metrics' in results:
        metrics = results['metrics']
        if 'summary' in metrics:
            report.add_section("Metrics Summary", "")
            report.add_table(metrics['summary'], "Model Performance Metrics")
    
    # Generate report
    output_path = report.generate(output_path)
    
    return output_path