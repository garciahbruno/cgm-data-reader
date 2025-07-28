import pandas as pd
import requests
from datetime import datetime
import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64


def chat_with_ollama(prompt, model="llama3.2"):
    """Send prompt to Ollama and get response"""
    try:
        response = requests.post('http://localhost:11434/api/generate',
                               json={
                                   'model': model,
                                   'prompt': prompt,
                                   'stream': False
                               })
        return response.json()['response']
    except Exception as e:
        return f"Error communicating with Ollama: {e}"

def read_txt_file(filepath):
    """Read text file content"""
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        return f"Error reading text file: {e}"

def read_csv_file(filepath):
    """Read and parse CSV file"""
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        return f"Error reading CSV file: {e}"

def generate_cgm_report(txt_file, daily_csv, summary_7_csv, summary_30_csv):
    """Generate comprehensive CGM analysis report"""
    
    # Read all files
    print("Reading CGM data files...")
    
    # Try to read text file, but make it optional
    if txt_file and os.path.exists(txt_file):
        txt_content = read_txt_file(txt_file)
        if isinstance(txt_content, str) and "Error" in txt_content:
            print(f"Warning: Could not read text file: {txt_content}")
            txt_content = "Text report not available - analyzing CSV data only."
    else:
        print("Text report not found - will analyze CSV data only.")
        txt_content = "Text report not available - analyzing CSV data only."
    
    daily_data = read_csv_file(daily_csv)
    summary_7_data = read_csv_file(summary_7_csv)
    summary_30_data = read_csv_file(summary_30_csv)
    
    # Check for errors in required files
    if isinstance(daily_data, str):
        return daily_data
    if isinstance(summary_7_data, str):
        return summary_7_data
    if isinstance(summary_30_data, str):
        return summary_30_data
    
    # Convert DataFrames to readable format for the LLM
    daily_summary = daily_data.to_string(index=False)
    summary_7_summary = summary_7_data.to_string(index=False)
    summary_30_summary = summary_30_data.to_string(index=False)
    
    # Create the comprehensive prompt for better readability
    endocrinologist_prompt = f"""
You are an experienced endocrinologist writing a patient-friendly CGM report. Create a comprehensive but easy-to-understand analysis.

**PATIENT DATA:**
{txt_content}

**DAILY SUMMARIES:**
{daily_summary}

**7-DAY SUMMARY:**
{summary_7_summary}

**30-DAY SUMMARY:**
{summary_30_summary}

**INSTRUCTIONS:** Write a clear, professional report using these sections. When referencing specific days, use just the date format "July 15" without day names:

# GLUCOSE CONTROL OVERVIEW
Start with a simple 2-3 sentence summary of how well the patient's glucose is controlled overall. Use plain language like "excellent," "good," "needs improvement," or "concerning."

# KEY FINDINGS
List the 3 most important discoveries from the data:
- Finding 1: [Explain what you found and why it matters, using dates like "July 15" when relevant]
- Finding 2: [Explain what you found and why it matters, using dates like "July 16" when relevant] 
- Finding 3: [Explain what you found and why it matters, using dates like "July 17" when relevant]

# TIME IN RANGE ANALYSIS
Explain what "time in range" means and how this patient is doing:
- Target range is 70-180 mg/dL
- Current performance: X% in range
- What this means for health
- How it compares to diabetes management goals (>70% is excellent, 50-70% is good)

# GLUCOSE PATTERNS
Describe when glucose tends to be high or low using specific dates:
- **High glucose times:** When does it happen most? (reference specific dates like "July 15")
- **Low glucose concerns:** Any dangerous low episodes? (reference specific dates)
- **Daily trends:** Are there consistent patterns? (mention specific dates)

# SPECIFIC RECOMMENDATIONS

## Immediate Actions (This Week)
- Action 1: [Specific, actionable advice]
- Action 2: [Specific, actionable advice]

## Medication Adjustments
- [If needed, suggest specific changes with reasoning]

## Lifestyle Modifications  
- Diet: [Specific meal timing or food suggestions]
- Exercise: [Specific activity recommendations]
- Monitoring: [What to watch for, mention specific dates]

# PROGRESS TRACKING
What should the patient monitor over the next 2 weeks to measure improvement? Include specific dates for follow-up.

**WRITING STYLE:**
- Use simple, clear language
- Explain medical terms when first mentioned
- Be encouraging but honest
- Focus on actionable advice
- Use dates in format "July 15" (no day names)
- Write like you're talking to the patient directly

Generate the complete report now:
"""
    
    print("Analyzing CGM data...")
    print("This may take a moment...")
    
    # Get analysis from Ollama
    report = chat_with_ollama(endocrinologist_prompt)
    
    return report

def create_glucose_charts(daily_data, summary_7_data, summary_30_data):
    """Create glucose monitoring charts"""
    
    # Set style for better looking plots
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Convert date strings to datetime for proper formatting
    daily_data_copy = daily_data.copy()
    daily_data_copy['Date'] = pd.to_datetime(daily_data_copy['Date'])
    daily_data_copy = daily_data_copy.sort_values('Date')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('CGM Analysis Dashboard', fontsize=20, fontweight='bold')
    
    # Chart 1: Daily Mean Glucose Trend
    axes[0,0].plot(daily_data_copy['Date'], daily_data_copy['MeanGlucose'], 
                   marker='o', linewidth=2, markersize=6, color='#2E86C1')
    axes[0,0].axhline(y=180, color='red', linestyle='--', alpha=0.7, label='High (>180)')
    axes[0,0].axhline(y=70, color='orange', linestyle='--', alpha=0.7, label='Low (<70)')
    axes[0,0].fill_between(daily_data_copy['Date'], 70, 180, alpha=0.2, color='green', label='Target Range')
    axes[0,0].set_title('Daily Average Glucose Trend', fontweight='bold')
    axes[0,0].set_ylabel('Glucose (mg/dL)')
    axes[0,0].set_xlabel('Date')
    
    # Format x-axis to show just month/day
    import matplotlib.dates as mdates
    axes[0,0].xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    axes[0,0].xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(daily_data_copy)//7)))
    plt.setp(axes[0,0].xaxis.get_majorticklabels(), rotation=45)
    
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Chart 2: Time in Range Comparison
    periods = ['7-Day', '30-Day']
    tir_values = [summary_7_data['AvgPercentInRange'].iloc[0], 
                  summary_30_data['AvgPercentInRange'].iloc[0]]
    high_values = [summary_7_data['AvgPercentHigh'].iloc[0], 
                   summary_30_data['AvgPercentHigh'].iloc[0]]
    low_values = [summary_7_data['AvgPercentLow'].iloc[0], 
                  summary_30_data['AvgPercentLow'].iloc[0]]
    
    x = range(len(periods))
    width = 0.25
    
    axes[0,1].bar([i - width for i in x], tir_values, width, label='In Range (70-180)', color='#28B463')
    axes[0,1].bar(x, high_values, width, label='High (>180)', color='#E74C3C')
    axes[0,1].bar([i + width for i in x], low_values, width, label='Low (<70)', color='#F39C12')
    
    axes[0,1].set_title('Time in Range Comparison', fontweight='bold')
    axes[0,1].set_ylabel('Percentage (%)')
    axes[0,1].set_xticks(x)
    axes[0,1].set_xticklabels(periods)
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Chart 3: Daily Glucose Variability
    axes[1,0].bar(daily_data_copy['Date'], daily_data_copy['StdDev'], 
                  color='#8E44AD', alpha=0.7)
    axes[1,0].axhline(y=30, color='red', linestyle='--', alpha=0.7, label='High Variability (>30)')
    axes[1,0].set_title('Daily Glucose Variability (Standard Deviation)', fontweight='bold')
    axes[1,0].set_ylabel('Standard Deviation (mg/dL)')
    axes[1,0].set_xlabel('Date')
    
    # Format x-axis
    axes[1,0].xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    axes[1,0].xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(daily_data_copy)//7)))
    plt.setp(axes[1,0].xaxis.get_majorticklabels(), rotation=45)
    
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Chart 4: Summary Metrics Gauge
    metrics = ['Mean Glucose', 'Time in Range', 'Glucose Variability']
    values_7d = [summary_7_data['MeanGlucose'].iloc[0], 
                 summary_7_data['AvgPercentInRange'].iloc[0],
                 summary_7_data['GlucoseVariability'].iloc[0]]
    values_30d = [summary_30_data['MeanGlucose'].iloc[0], 
                  summary_30_data['AvgPercentInRange'].iloc[0],
                  summary_30_data['GlucoseVariability'].iloc[0]]
    
    x = range(len(metrics))
    width = 0.35
    
    axes[1,1].bar([i - width/2 for i in x], values_7d, width, label='7-Day Average', color='#3498DB')
    axes[1,1].bar([i + width/2 for i in x], values_30d, width, label='30-Day Average', color='#E67E22')
    
    axes[1,1].set_title('Key Metrics Comparison', fontweight='bold')
    axes[1,1].set_ylabel('Values')
    axes[1,1].set_xticks(x)
    axes[1,1].set_xticklabels(metrics, rotation=45)
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save chart to bytes
    chart_buffer = BytesIO()
    plt.savefig(chart_buffer, format='png', dpi=300, bbox_inches='tight')
    chart_buffer.seek(0)
    plt.close()
    
    return chart_buffer

def save_report_as_html(report, daily_data, summary_7_data, summary_30_data, output_file="cgm_clinical_report.html"):
    """Save the generated report as a beautiful HTML file"""
    try:
        # Create charts first
        chart_buffer = create_glucose_charts(daily_data, summary_7_data, summary_30_data)
        
        # Save chart as base64 for embedding
        import base64
        chart_base64 = base64.b64encode(chart_buffer.getvalue()).decode()
        
        # Create beautiful HTML
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CGM Clinical Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1000px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            margin: 0;
            font-size: 2.5rem;
            font-weight: 300;
        }}
        
        .header .subtitle {{
            margin-top: 10px;
            font-size: 1.1rem;
            opacity: 0.9;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        
        .metric-card {{
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            border-left: 5px solid #667eea;
        }}
        
        .metric-value {{
            font-size: 2rem;
            font-weight: bold;
            color: #333;
            margin-bottom: 5px;
        }}
        
        .metric-label {{
            color: #666;
            font-size: 0.9rem;
        }}
        
        .metric-target {{
            color: #28a745;
            font-size: 0.8rem;
            margin-top: 5px;
        }}
        
        .chart-section {{
            margin: 40px 0;
            text-align: center;
        }}
        
        .chart-image {{
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        }}
        
        .report-section {{
            margin: 30px 0;
        }}
        
        .report-section h1 {{
            color: #333;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            font-size: 1.8rem;
        }}
        
        .report-section h2 {{
            color: #555;
            margin-top: 25px;
            font-size: 1.4rem;
        }}
        
        .report-section h3 {{
            color: #666;
            margin-top: 20px;
            font-size: 1.2rem;
        }}
        
        .report-content {{
            background: #f8f9fa;
            padding: 25px;
            border-radius: 10px;
            margin: 20px 0;
            border-left: 5px solid #28a745;
        }}
        
        .key-findings {{
            background: #e3f2fd;
            border-left: 5px solid #2196f3;
        }}
        
        .recommendations {{
            background: #f3e5f5;
            border-left: 5px solid #9c27b0;
        }}
        
        .important {{
            background: #fff3e0;
            border-left: 5px solid #ff9800;
        }}
        
        ul, ol {{
            padding-left: 20px;
        }}
        
        li {{
            margin: 8px 0;
        }}
        
        .footer {{
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #666;
            border-top: 1px solid #dee2e6;
        }}
        
        .status-excellent {{ color: #28a745; }}
        .status-good {{ color: #17a2b8; }}
        .status-needs-improvement {{ color: #ffc107; }}
        .status-concerning {{ color: #dc3545; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🩺 CGM Clinical Report</h1>
            <div class="subtitle">
                Continuous Glucose Monitoring Analysis<br>
                Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
            </div>
        </div>
        
        <div class="content">
            <div class="summary-grid">
                <div class="metric-card">
                    <div class="metric-value">{summary_7_data['MeanGlucose'].iloc[0]}</div>
                    <div class="metric-label">7-Day Avg Glucose</div>
                    <div class="metric-target">Target: 70-180 mg/dL</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{summary_7_data['AvgPercentInRange'].iloc[0]}%</div>
                    <div class="metric-label">Time in Range</div>
                    <div class="metric-target">Goal: >70%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{summary_7_data['AvgPercentHigh'].iloc[0]}%</div>
                    <div class="metric-label">Time Above Range</div>
                    <div class="metric-target">Goal: <25%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{summary_7_data['AvgPercentLow'].iloc[0]}%</div>
                    <div class="metric-label">Time Below Range</div>
                    <div class="metric-target">Goal: <4%</div>
                </div>
            </div>
            
            <div class="chart-section">
                <h2>📊 Glucose Monitoring Dashboard</h2>
                <img src="data:image/png;base64,{chart_base64}" alt="CGM Dashboard" class="chart-image">
            </div>
            
            <div class="report-section">
                <div class="report-content">
"""

        # Process the report content with better formatting
        report_lines = report.split('\n')
        in_section = False
        section_class = "report-content"
        
        for line in report_lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect section types for styling
            if "KEY FINDINGS" in line.upper() or "FINDINGS" in line.upper():
                section_class = "key-findings"
            elif "RECOMMENDATION" in line.upper() or "ACTION" in line.upper():
                section_class = "recommendations"
            elif "IMPORTANT" in line.upper() or "URGENT" in line.upper() or "RISK" in line.upper():
                section_class = "important"
            else:
                section_class = "report-content"
            
            # Process different line types
            if line.startswith('# '):
                if in_section:
                    html_content += "</div>\n"
                title = line.replace('# ', '').strip()
                html_content += f'<h1>{title}</h1>\n<div class="{section_class}">\n'
                in_section = True
            elif line.startswith('## '):
                title = line.replace('## ', '').strip()
                html_content += f'<h2>{title}</h2>\n'
            elif line.startswith('### '):
                title = line.replace('### ', '').strip()
                html_content += f'<h3>{title}</h3>\n'
            elif line.startswith('- '):
                bullet_text = line.replace('- ', '')
                html_content += f'<li>{bullet_text}</li>\n'
            elif line.startswith('**') and line.endswith('**'):
                bold_text = line.replace('**', '')
                html_content += f'<p><strong>{bold_text}</strong></p>\n'
            else:
                if len(line) > 0:
                    html_content += f'<p>{line}</p>\n'
        
        if in_section:
            html_content += "</div>\n"
        
        # Close the HTML
        html_content += f"""
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p><em>Analysis Period: {summary_30_data['StartDate'].iloc[0]} to {summary_30_data['EndDate'].iloc[0]} ({summary_30_data['TotalDays'].iloc[0]} days)</em></p>
        </div>
    </div>
</body>
</html>
"""
        
        # Save the HTML file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Beautiful HTML report saved to: {output_file}")
        print("Open this file in your web browser to view the formatted report!")
        
    except Exception as e:
        print(f"Error creating HTML report: {e}")
        # Fallback to simple text
        with open(output_file.replace('.html', '.txt'), 'w', encoding='utf-8') as file:
            file.write(report)
        print(f"Fallback text report saved to: {output_file.replace('.html', '.txt')}")

def find_files():
    """Find CGM analysis files in current directory and data subdirectory"""
    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir, 'data')
    
    print(f"Looking for files in: {current_dir}")
    print(f"Also checking data directory: {data_dir}")
    
    # Define expected file names and possible locations
    files = {
        'txt_file': 'cgm_analysis_report.txt',  # Optional
        'daily_csv': 'daily_cgm_summaries.csv',
        'summary_7_csv': '7_day_summary.csv',
        'summary_30_csv': '30_day_summary.csv'
    }
    
    # Required files (CSV files)
    required_files = ['daily_csv', 'summary_7_csv', 'summary_30_csv']
    
    # Check if files exist in multiple locations
    found_files = {}
    missing_files = []
    
    for file_type, filename in files.items():
        # Check current directory first
        current_path = filename
        data_path = os.path.join('data', filename)
        
        if os.path.exists(current_path):
            found_files[file_type] = current_path
            print(f"✓ Found: {current_path}")
        elif os.path.exists(data_path):
            found_files[file_type] = data_path
            print(f"✓ Found: {data_path}")
        else:
            if file_type in required_files:
                missing_files.append(filename)
                print(f"✗ Missing: {filename} (checked both . and ./data/)")
            else:
                print(f"⚠ Optional file not found: {filename}")
                found_files[file_type] = None  # Mark as optional
    
    return found_files, missing_files

def main():
    """Main function to run the CGM analysis"""
    print("CGM Data Clinical Analysis Tool")
    print("=" * 40)
    
    # Find files automatically
    found_files, missing_files = find_files()
    
    if missing_files:
        print(f"\nMissing required files: {missing_files}")
        print("Please make sure you've run the CGM analysis notebook first to generate these files.")
        return
    
    print(f"\nAll required files found! Proceeding with analysis...")
    
    # Generate the report
    report = generate_cgm_report(
        found_files['txt_file'],
        found_files['daily_csv'], 
        found_files['summary_7_csv'],
        found_files['summary_30_csv']
    )
    
    # Read the data for charts
    daily_data = read_csv_file(found_files['daily_csv'])
    summary_7_data = read_csv_file(found_files['summary_7_csv'])
    summary_30_data = read_csv_file(found_files['summary_30_csv'])
    
    # Print the report
    print("\n" + "=" * 50)
    print("CLINICAL ANALYSIS REPORT")
    print("=" * 50)
    print(report)
    
    # Save to beautiful HTML report
    save_report_as_html(report, daily_data, summary_7_data, summary_30_data, "cgm_clinical_report.html")
    
    print("\n" + "=" * 50)
    print("Analysis Complete!")

if __name__ == "__main__":
    main()