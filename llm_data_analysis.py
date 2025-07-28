import pandas as pd
import requests
from datetime import datetime
import os
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import tempfile
import numpy as np
import shutil
import base64
from io import BytesIO

def check_ollama_model(model_name="mistral-nemo"):
    """Check if the specified model is available in Ollama"""
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=10)
        if response.status_code == 200:
            models = response.json()
            available_models = [model['name'] for model in models.get('models', [])]
            
            model_available = any(model_name in model for model in available_models)
            
            if not model_available:
                print(f"\nWARNING: {model_name} not found in Ollama.")
                print("Available models:", available_models)
                print(f"\nTo install {model_name}, run:")
                print(f"ollama pull {model_name}")
                return False
            else:
                print(f"✓ {model_name} is available in Ollama")
                return True
        else:
            print("Could not connect to Ollama API. Make sure Ollama is running.")
            return False
    except Exception as e:
        print(f"Error checking Ollama models: {e}")
        print("Make sure Ollama is installed and running (ollama serve)")
        return False

def chat_with_ollama(prompt, model="mistral-nemo"):
    """Send prompt to Ollama using Mistral Nemo for better clinical accuracy"""
    try:
        response = requests.post('http://localhost:11434/api/generate',
                               json={
                                   'model': model,
                                   'prompt': prompt,
                                   'stream': False,
                                   'options': {
                                       'temperature': 0.1,
                                       'top_p': 0.9,
                                       'num_predict': 4000,
                                       'repeat_penalty': 1.1,
                                       'top_k': 50,
                                       'num_ctx': 8192
                                   }
                               }, timeout=120)
        
        if response.status_code != 200:
            return f"Error: Ollama returned status {response.status_code}. Response: {response.text}"
        
        result = response.json()
        if 'response' not in result:
            return f"Error: Unexpected response format from Ollama: {result}"
        
        return result['response']
        
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to Ollama. Make sure Ollama is running (ollama serve)"
    except Exception as e:
        return f"Error communicating with Ollama: {e}"

def format_data_for_llm_strict(daily_data, summary_7_data, summary_30_data):
    """Format data with explicit clinical target violations highlighted"""
    
    if daily_data.empty or summary_7_data.empty or summary_30_data.empty:
        raise ValueError("One or more datasets are empty")
    
    s7 = summary_7_data.iloc[0]
    s30 = summary_30_data.iloc[0]
    
    def evaluate_metric(value, target_min=None, target_max=None, metric_name=""):
        try:
            value = float(value)
        except (ValueError, TypeError):
            return f"{value} - INVALID VALUE"
            
        if target_min is not None and target_max is not None:
            if target_min <= value <= target_max:
                return f"{value:.1f} ✓ MEETS TARGET"
            else:
                return f"{value:.1f} ✗ OUTSIDE TARGET ({target_min}-{target_max})"
        elif target_min is not None:
            if value >= target_min:
                return f"{value:.1f} ✓ MEETS TARGET"
            else:
                return f"{value:.1f} ✗ BELOW TARGET (should be ≥{target_min})"
        elif target_max is not None:
            if value <= target_max:
                return f"{value:.1f} ✓ MEETS TARGET"
            else:
                return f"{value:.1f} ✗ ABOVE TARGET (should be ≤{target_max})"
    
    summary_7_text = f"""7-DAY SUMMARY:
- Mean Glucose: {evaluate_metric(s7['MeanGlucose'], 70, 180)} mg/dL
- Time in Range (70-180): {evaluate_metric(s7['AvgPercentInRange'], target_min=70)}%
- Time Above Range (>180): {evaluate_metric(s7['AvgPercentHigh'], target_max=25)}%
- Time Below Range (<70): {evaluate_metric(s7['AvgPercentLow'], target_max=4)}%
- Period: {s7.get('StartDate', 'N/A')} to {s7.get('EndDate', 'N/A')}"""

    summary_30_text = f"""30-DAY SUMMARY:
- Mean Glucose: {evaluate_metric(s30['MeanGlucose'], 70, 180)} mg/dL
- Time in Range (70-180): {evaluate_metric(s30['AvgPercentInRange'], target_min=70)}%
- Time Above Range (>180): {evaluate_metric(s30['AvgPercentHigh'], target_max=25)}%
- Time Below Range (<70): {evaluate_metric(s30['AvgPercentLow'], target_max=4)}%
- Period: {s30.get('StartDate', 'N/A')} to {s30.get('EndDate', 'N/A')}"""

    daily_text = "DAILY BREAKDOWN WITH CLINICAL FLAGS:\n"
    daily_sorted = daily_data.sort_values('Date').tail(14)
    
    days_with_concerns = set()  # Use set to avoid double-counting same day
    days_with_extreme_issues = set()
    
    for idx, day in daily_sorted.iterrows():
        date = str(day['Date'])
        try:
            mean_glucose = float(day['MeanGlucose'])
            time_in_range = float(day.get('PercentInRange', 0))
            time_high = float(day.get('PercentHigh', 0))
            time_low = float(day.get('PercentLow', 0))
        except (ValueError, TypeError):
            continue
        
        clinical_flags = []
        day_has_concern = False
        day_has_extreme = False
        
        # Check glucose levels
        if mean_glucose > 250:
            clinical_flags.append("SEVERELY HIGH AVERAGE")
            day_has_extreme = True
        elif mean_glucose > 200:
            clinical_flags.append("HIGH AVERAGE")
            day_has_concern = True
        elif mean_glucose > 180:
            clinical_flags.append("ABOVE TARGET")
            day_has_concern = True
        elif mean_glucose < 70:
            clinical_flags.append("DANGEROUSLY LOW AVERAGE")
            day_has_extreme = True
        
        # Check time in range
        if time_in_range < 50:
            clinical_flags.append("POOR TIME IN RANGE")
            day_has_concern = True
        elif time_in_range < 70:
            clinical_flags.append("SUBOPTIMAL TIME IN RANGE")
            day_has_concern = True
        
        # Check hypoglycemia
        if time_low > 4:
            clinical_flags.append("EXCESSIVE HYPOGLYCEMIA")
            day_has_extreme = True
        
        # Check hyperglycemia
        if time_high > 50:
            clinical_flags.append("SEVERE HYPERGLYCEMIA")
            day_has_extreme = True
        elif time_high > 25:
            clinical_flags.append("EXCESSIVE HYPERGLYCEMIA")
            day_has_concern = True
        
        # Add to sets (automatically handles duplicates)
        if day_has_extreme:
            days_with_extreme_issues.add(date)
            days_with_concerns.add(date)  # Extreme issues are also concerning
        elif day_has_concern:
            days_with_concerns.add(date)
        
        flag_text = f" [{', '.join(clinical_flags)}]" if clinical_flags else " [ACCEPTABLE]"
        daily_text += f"- {date}: Avg={mean_glucose:.0f} mg/dL, InRange={time_in_range:.0f}%, High={time_high:.0f}%, Low={time_low:.1f}%{flag_text}\n"
    
    total_days_analyzed = len(daily_sorted)
    concerning_days_count = len(days_with_concerns)
    extreme_days_count = len(days_with_extreme_issues)
    
    daily_text += f"\nPATTERN ANALYSIS:\n"
    daily_text += f"- Days with concerning metrics: {concerning_days_count}/{total_days_analyzed} ({concerning_days_count/total_days_analyzed*100:.0f}%)\n"
    daily_text += f"- Days with extreme glucose issues: {extreme_days_count}/{total_days_analyzed} ({extreme_days_count/total_days_analyzed*100:.0f}%)\n"
    
    return summary_7_text, summary_30_text, daily_text

def create_clinical_assessment_prompt(summary_7_text, summary_30_text, daily_text):
    """Create a strict clinical assessment prompt optimized for Mistral Nemo"""
    
    prompt = f"""You are a board-certified endocrinologist analyzing CGM data for a pediatric patient with diabetes. Your analysis must be clinically accurate, evidence-based, and identify ALL target violations.

<clinical_data>
{summary_7_text}

{summary_30_text}

{daily_text}
</clinical_data>

<clinical_guidelines>
PEDIATRIC DIABETES CLINICAL TARGETS (ADA/ISPAD 2024):
- Time in Range (70-180 mg/dL): >70% (PRIMARY OUTCOME MEASURE)
- Time Above Range (>180 mg/dL): <25% (HYPERGLYCEMIA THRESHOLD) 
- Time Below Range (<70 mg/dL): <4% (HYPOGLYCEMIA SAFETY LIMIT)
- Mean Glucose: 70-180 mg/dL (OPTIMAL GLYCEMIC CONTROL)
</clinical_guidelines>

<analysis_instructions>
CRITICAL REQUIREMENTS:
1. Identify EVERY metric marked with "✗ OUTSIDE TARGET", "✗ ABOVE TARGET", or "✗ BELOW TARGET"
2. If ANY target is not met, classify as clinical concern requiring attention
3. Use EXACT numbers from the data - do not approximate or round
4. Compare 7-day vs 30-day trends to assess improvement or deterioration
5. Analyze daily patterns for concerning glucose excursions
6. Provide specific, actionable clinical recommendations based on target violations
</analysis_instructions>

Provide your clinical assessment in this structured format:

**CLINICAL STATUS:** [Select one: OPTIMAL/ACCEPTABLE/SUBOPTIMAL/POOR/CONCERNING]

**OVERALL ASSESSMENT:**
[Provide a brief summary comparing 7-day vs 30-day metrics, noting specific target violations]

**KEY CLINICAL CONCERNS:**
[List each metric that violates clinical targets with exact values and clinical significance]

**GLUCOSE PATTERNS ANALYSIS:**
[Analyze daily patterns, clinical flags, and glucose excursions from the daily data]

**CLINICAL RECOMMENDATIONS:**
[Provide specific, evidence-based recommendations for each identified target violation]

**FOLLOW-UP REQUIREMENTS:**
[Specify monitoring frequency and clinical actions needed based on the severity of glucose control issues]

Begin your analysis now, ensuring you address every target violation identified in the data."""

    return prompt

def validate_clinical_response(response, summary_7_data, summary_30_data):
    """Validate that the LLM response correctly identifies target violations"""
    
    s7 = summary_7_data.iloc[0]
    s30 = summary_30_data.iloc[0]
    
    validation_issues = []
    
    try:
        if float(s7['AvgPercentInRange']) < 70 or float(s30['AvgPercentInRange']) < 70:
            if "time in range" not in response.lower() or "suboptimal" not in response.lower():
                validation_issues.append("Failed to identify suboptimal Time in Range")
        
        if float(s7['AvgPercentHigh']) > 25 or float(s30['AvgPercentHigh']) > 25:
            if "above range" not in response.lower() or ("poor" not in response.lower() and "concerning" not in response.lower()):
                validation_issues.append("Failed to identify excessive time above range")
        
        if float(s7['MeanGlucose']) > 180 or float(s30['MeanGlucose']) > 180:
            if "glucose" not in response.lower() or "target" not in response.lower():
                validation_issues.append("Failed to identify elevated glucose averages")
        
        concerning_phrases = ["no concerns", "no significant concerns", "appears effective", "excellent management"]
        if any(phrase in response.lower() for phrase in concerning_phrases):
            if float(s7['AvgPercentInRange']) < 70 or float(s30['AvgPercentInRange']) < 70:
                validation_issues.append("Provided inappropriate reassurance despite poor glucose control")
    except (ValueError, TypeError, KeyError) as e:
        validation_issues.append(f"Data validation error: {e}")
    
    return validation_issues

def assess_clinical_status(summary_7_data, summary_30_data):
    """Provide objective clinical status assessment"""
    s7 = summary_7_data.iloc[0]
    s30 = summary_30_data.iloc[0]
    
    violations = 0
    critical_violations = 0
    
    try:
        if float(s7['AvgPercentInRange']) < 70: violations += 1; critical_violations += 1
        if float(s30['AvgPercentInRange']) < 70: violations += 1; critical_violations += 1
        if float(s7['AvgPercentHigh']) > 25: violations += 1; critical_violations += 1
        if float(s30['AvgPercentHigh']) > 25: violations += 1; critical_violations += 1
        if float(s7['AvgPercentLow']) > 4: violations += 1; critical_violations += 1
        if float(s30['AvgPercentLow']) > 4: violations += 1; critical_violations += 1
        if float(s7['MeanGlucose']) > 180: violations += 1
        if float(s30['MeanGlucose']) > 180: violations += 1
    except (ValueError, TypeError, KeyError):
        return "UNKNOWN", "Unable to assess due to data issues"
    
    if critical_violations >= 2:
        return "POOR", "Multiple critical targets not met - requires immediate attention"
    elif critical_violations >= 1:
        return "SUBOPTIMAL", "Key clinical targets not met - requires intervention"
    elif violations >= 2:
        return "SUBOPTIMAL", "Several targets not met - needs improvement"
    elif violations == 1:
        return "ACCEPTABLE", "Most targets met with minor areas for improvement"
    else:
        return "OPTIMAL", "All clinical targets met"

def create_chart_base64(chart_func, *args):
    """Create a chart and return it as base64 string for HTML embedding"""
    try:
        # Set matplotlib to use non-interactive backend
        plt.style.use('default')
        
        # Create chart
        fig = chart_func(*args)
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=200, bbox_inches='tight', 
                   facecolor='white', edgecolor='none', pad_inches=0.1)
        buffer.seek(0)
        
        chart_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        buffer.close()
        
        return f"data:image/png;base64,{chart_base64}"
        
    except Exception as e:
        print(f"Warning: Error creating chart: {e}")
        return None

def create_glucose_trend_chart_fig(daily_data):
    """Create glucose trend chart figure"""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('white')
    
    # Apple-inspired colors
    primary_blue = '#007AFF'
    success_green = '#34C759'
    warning_orange = '#FF9500'
    danger_red = '#FF3B30'
    
    # Clean data
    daily_data = daily_data.copy()
    daily_data['Date'] = pd.to_datetime(daily_data['Date'], errors='coerce')
    daily_data = daily_data.dropna(subset=['Date', 'MeanGlucose'])
    daily_data['MeanGlucose'] = pd.to_numeric(daily_data['MeanGlucose'], errors='coerce')
    daily_data = daily_data.dropna(subset=['MeanGlucose'])
    
    if daily_data.empty:
        ax.text(0.5, 0.5, 'No valid glucose data available', 
               ha='center', va='center', transform=ax.transAxes, fontsize=16,
               color='#8E8E93', fontweight='500')
        return fig
    
    daily_sorted = daily_data.sort_values('Date')
    
    # Background gradient for target ranges
    ax.axhspan(70, 180, alpha=0.05, color=success_green, zorder=0)
    ax.axhspan(180, 400, alpha=0.03, color=danger_red, zorder=0)
    ax.axhspan(0, 70, alpha=0.03, color=warning_orange, zorder=0)
    
    # Main trend line with Apple-style styling
    ax.plot(daily_sorted['Date'], daily_sorted['MeanGlucose'], 
           linewidth=3, color=primary_blue, zorder=3, alpha=0.9,
           marker='o', markersize=6, markerfacecolor='white', 
           markeredgecolor=primary_blue, markeredgewidth=2)
    
    # Target lines
    ax.axhline(y=180, color=danger_red, linestyle='-', alpha=0.4, linewidth=2)
    ax.axhline(y=70, color=warning_orange, linestyle='-', alpha=0.4, linewidth=2)
    
    # Clean Apple-style axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#E5E5EA')
    ax.spines['bottom'].set_color('#E5E5EA')
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    
    # Grid
    ax.grid(True, alpha=0.15, color='#E5E5EA', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Labels with Apple typography
    ax.set_xlabel('Date', fontsize=14, color='#1D1D1F', fontweight='600', labelpad=15)
    ax.set_ylabel('Average Glucose (mg/dL)', fontsize=14, color='#1D1D1F', fontweight='600', labelpad=15)
    
    # Tick styling
    ax.tick_params(colors='#1D1D1F', labelsize=12, pad=8)
    
    # Format dates
    if len(daily_sorted) > 1:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(daily_sorted)//8)))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
    
    plt.tight_layout()
    return fig

def create_time_in_range_chart_fig(daily_data):
    """Create time in range chart figure"""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('white')
    
    # Apple colors
    success_green = '#34C759'
    warning_orange = '#FF9500'
    danger_red = '#FF3B30'
    
    # Clean data
    daily_data = daily_data.copy()
    daily_data['Date'] = pd.to_datetime(daily_data['Date'], errors='coerce')
    daily_data = daily_data.dropna(subset=['Date'])
    
    for col in ['PercentLow', 'PercentInRange', 'PercentHigh']:
        if col in daily_data.columns:
            daily_data[col] = pd.to_numeric(daily_data[col], errors='coerce')
        else:
            daily_data[col] = 0
    
    daily_data = daily_data.fillna(0)
    daily_sorted = daily_data.sort_values('Date').tail(14)
    
    if daily_sorted.empty:
        ax.text(0.5, 0.5, 'No valid time range data available', 
               ha='center', va='center', transform=ax.transAxes, fontsize=16,
               color='#8E8E93', fontweight='500')
        return fig
    
    dates = daily_sorted['Date']
    low = daily_sorted['PercentLow']
    in_range = daily_sorted['PercentInRange']
    high = daily_sorted['PercentHigh']
    
    # Stacked bars with rounded appearance
    width = 0.8
    bars_low = ax.bar(dates, low, width, label='Below Range (<70 mg/dL)', 
                      color=warning_orange, alpha=0.9, edgecolor='white', linewidth=2)
    bars_in_range = ax.bar(dates, in_range, width, bottom=low, 
                          label='In Range (70-180 mg/dL)', color=success_green, 
                          alpha=0.9, edgecolor='white', linewidth=2)
    bars_high = ax.bar(dates, high, width, bottom=low+in_range, 
                       label='Above Range (>180 mg/dL)', color=danger_red, 
                       alpha=0.9, edgecolor='white', linewidth=2)
    
    # Target line
    ax.axhline(y=70, color=success_green, linestyle='-', alpha=0.4, linewidth=2)
    
    # Clean Apple styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#E5E5EA')
    ax.spines['bottom'].set_color('#E5E5EA')
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    
    ax.grid(True, alpha=0.15, axis='y', color='#E5E5EA', linewidth=0.5)
    ax.set_axisbelow(True)
    
    ax.set_xlabel('Date', fontsize=14, color='#1D1D1F', fontweight='600', labelpad=15)
    ax.set_ylabel('Time (%)', fontsize=14, color='#1D1D1F', fontweight='600', labelpad=15)
    
    ax.tick_params(colors='#1D1D1F', labelsize=12, pad=8)
    ax.set_ylim(0, 105)
    
    # Apple-style legend
    legend = ax.legend(frameon=False, loc='upper right', fontsize=11,
                      bbox_to_anchor=(0.98, 0.98), ncol=1)
    for text in legend.get_texts():
        text.set_color('#1D1D1F')
        text.set_fontweight('500')
    
    # Format dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    if len(daily_sorted) > 7:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
    
    plt.tight_layout()
    return fig

def create_summary_comparison_chart_fig(summary_7_data, summary_30_data):
    """Create summary comparison chart figure"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    fig.patch.set_facecolor('white')
    
    # Apple colors
    primary_blue = '#007AFF'
    secondary_blue = '#5856D6'
    success_green = '#34C759'
    success_mint = '#00C7BE'
    warning_orange = '#FF9500'
    warning_yellow = '#FFCC00'
    danger_red = '#FF3B30'
    danger_pink = '#FF2D92'
    
    s7 = summary_7_data.iloc[0]
    s30 = summary_30_data.iloc[0]
    periods = ['7 Days', '30 Days']
    
    # Style all subplots
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#E5E5EA')
        ax.spines['bottom'].set_color('#E5E5EA')
        ax.spines['left'].set_linewidth(1)
        ax.spines['bottom'].set_linewidth(1)
        ax.grid(True, alpha=0.15, axis='y', color='#E5E5EA', linewidth=0.5)
        ax.tick_params(colors='#1D1D1F', labelsize=11, pad=6)
        ax.set_axisbelow(True)
    
    # Mean Glucose
    try:
        mean_glucose = [float(s7['MeanGlucose']), float(s30['MeanGlucose'])]
        bars1 = ax1.bar(periods, mean_glucose, color=[primary_blue, secondary_blue], 
                        alpha=0.9, width=0.6, edgecolor='white', linewidth=2)
        ax1.axhline(y=180, color=danger_red, linestyle='-', alpha=0.4, linewidth=2)
        ax1.set_ylabel('mg/dL', fontsize=12, color='#1D1D1F', fontweight='600')
        ax1.set_title('Mean Glucose', fontsize=14, fontweight='700', color='#1D1D1F', pad=15)
        
        for bar, value in zip(bars1, mean_glucose):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                    f'{value:.0f}', ha='center', va='bottom', fontweight='700', 
                    fontsize=12, color='#1D1D1F')
    except (ValueError, KeyError):
        ax1.text(0.5, 0.5, 'Data not available', ha='center', va='center', transform=ax1.transAxes)
    
    # Time in Range
    try:
        tir = [float(s7['AvgPercentInRange']), float(s30['AvgPercentInRange'])]
        bars2 = ax2.bar(periods, tir, color=[success_green, success_mint], 
                        alpha=0.9, width=0.6, edgecolor='white', linewidth=2)
        ax2.axhline(y=70, color=success_green, linestyle='-', alpha=0.4, linewidth=2)
        ax2.set_ylabel('Time in Range (%)', fontsize=12, color='#1D1D1F', fontweight='600')
        ax2.set_title('Time in Range', fontsize=14, fontweight='700', color='#1D1D1F', pad=15)
        ax2.set_ylim(0, 100)
        
        for bar, value in zip(bars2, tir):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                    f'{value:.0f}%', ha='center', va='bottom', fontweight='700', 
                    fontsize=12, color='#1D1D1F')
    except (ValueError, KeyError):
        ax2.text(0.5, 0.5, 'Data not available', ha='center', va='center', transform=ax2.transAxes)
    
    # Time Above Range
    try:
        tar = [float(s7['AvgPercentHigh']), float(s30['AvgPercentHigh'])]
        bars3 = ax3.bar(periods, tar, color=[danger_red, danger_pink], 
                        alpha=0.9, width=0.6, edgecolor='white', linewidth=2)
        ax3.axhline(y=25, color=danger_red, linestyle='-', alpha=0.4, linewidth=2)
        ax3.set_ylabel('Time Above Range (%)', fontsize=12, color='#1D1D1F', fontweight='600')
        ax3.set_title('Time Above Range', fontsize=14, fontweight='700', color='#1D1D1F', pad=15)
        
        for bar, value in zip(bars3, tar):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{value:.0f}%', ha='center', va='bottom', fontweight='700', 
                    fontsize=12, color='#1D1D1F')
    except (ValueError, KeyError):
        ax3.text(0.5, 0.5, 'Data not available', ha='center', va='center', transform=ax3.transAxes)
    
    # Time Below Range
    try:
        tlr = [float(s7['AvgPercentLow']), float(s30['AvgPercentLow'])]
        bars4 = ax4.bar(periods, tlr, color=[warning_orange, warning_yellow], 
                        alpha=0.9, width=0.6, edgecolor='white', linewidth=2)
        ax4.axhline(y=4, color=warning_orange, linestyle='-', alpha=0.4, linewidth=2)
        ax4.set_ylabel('Time Below Range (%)', fontsize=12, color='#1D1D1F', fontweight='600')
        ax4.set_title('Time Below Range', fontsize=14, fontweight='700', color='#1D1D1F', pad=15)
        
        for bar, value in zip(bars4, tlr):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='700', 
                    fontsize=12, color='#1D1D1F')
    except (ValueError, KeyError):
        ax4.text(0.5, 0.5, 'Data not available', ha='center', va='center', transform=ax4.transAxes)
    
    plt.tight_layout(pad=3.0)
    return fig

def create_apple_style_html(analysis_text, daily_data, summary_7_data, summary_30_data, output_filename):
    """Create beautiful Apple-style HTML website"""
    
    print(f"Creating Apple-style HTML website: {output_filename}")
    
    try:
        # Generate charts as base64
        print("Generating charts...")
        glucose_chart = create_chart_base64(create_glucose_trend_chart_fig, daily_data)
        time_range_chart = create_chart_base64(create_time_in_range_chart_fig, daily_data)
        summary_chart = create_chart_base64(create_summary_comparison_chart_fig, summary_7_data, summary_30_data)
        
        # Get clinical status
        clinical_status, status_description = assess_clinical_status(summary_7_data, summary_30_data)
        
        # Get metrics
        s7 = summary_7_data.iloc[0]
        s30 = summary_30_data.iloc[0]
        
        # Status color mapping
        status_colors = {
            "OPTIMAL": "#34C759",
            "ACCEPTABLE": "#007AFF", 
            "SUBOPTIMAL": "#FF9500",
            "POOR": "#FF3B30",
            "CONCERNING": "#FF3B30"
        }
        
        status_color = status_colors.get(clinical_status, "#007AFF")
        
        # Parse analysis text into sections and clean formatting
        analysis_sections = {}
        current_section = ""
        current_content = []
        
        if analysis_text:
            lines = analysis_text.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('**') and line.endswith('**'):
                    if current_section:
                        analysis_sections[current_section] = '\n'.join(current_content)
                    current_section = line[2:-2]  # Remove ** from section headers
                    current_content = []
                elif line:
                    # Remove all markdown formatting from content lines
                    cleaned_line = line.replace('**', '').replace('*', '')
                    current_content.append(cleaned_line)
            
            if current_section:
                analysis_sections[current_section] = '\n'.join(current_content)
        
        # Create HTML with Apple design system
        html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CGM Clinical Analysis</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: #1D1D1F;
            background: #FBFBFD;
            scroll-behavior: smooth;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px;
        }}
        
        /* Header */
        .header {{
            background: rgba(255, 255, 255, 0.8);
            backdrop-filter: blur(20px);
            border-bottom: 1px solid rgba(0, 0, 0, 0.05);
            position: sticky;
            top: 0;
            z-index: 100;
            padding: 20px 0;
        }}
        
        .header h1 {{
            font-size: 28px;
            font-weight: 700;
            text-align: center;
            color: #1D1D1F;
        }}
        
        .header .subtitle {{
            text-align: center;
            color: #86868B;
            font-size: 16px;
            margin-top: 4px;
        }}
        
        /* Hero Section */
        .hero {{
            padding: 60px 0;
            text-align: center;
            background: linear-gradient(135deg, #F5F5F7 0%, #FFFFFF 100%);
        }}
        
        .hero h2 {{
            font-size: 48px;
            font-weight: 800;
            color: #1D1D1F;
            margin-bottom: 16px;
            letter-spacing: -0.02em;
        }}
        
        .hero .date {{
            color: #86868B;
            font-size: 18px;
            margin-bottom: 40px;
        }}
        
        /* Status Badge */
        .status-badge {{
            display: inline-block;
            padding: 16px 32px;
            background: {status_color};
            color: white;
            border-radius: 24px;
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
        }}
        
        .status-description {{
            font-size: 18px;
            color: #515154;
            max-width: 600px;
            margin: 0 auto;
        }}
        
        /* Cards */
        .card {{
            background: white;
            border-radius: 20px;
            padding: 32px;
            box-shadow: 0 4px 24px rgba(0, 0, 0, 0.06);
            margin-bottom: 32px;
            border: 1px solid rgba(0, 0, 0, 0.05);
        }}
        
        .card h3 {{
            font-size: 24px;
            font-weight: 700;
            margin-bottom: 24px;
            color: #1D1D1F;
        }}
        
        /* Metrics Grid */
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 24px;
            margin: 40px 0;
        }}
        
        .metric-card {{
            background: white;
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 2px 16px rgba(0, 0, 0, 0.06);
            border: 1px solid rgba(0, 0, 0, 0.05);
            text-align: center;
        }}
        
        .metric-title {{
            font-size: 14px;
            color: #86868B;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
            font-weight: 600;
        }}
        
        .metric-value {{
            font-size: 28px;
            font-weight: 700;
            color: #1D1D1F;
            margin-bottom: 4px;
        }}
        
        .metric-period {{
            font-size: 24px;
            font-weight: 600;
            margin-bottom: 8px;
        }}
        
        .metric-target {{
            font-size: 12px;
            color: #86868B;
            margin-bottom: 12px;
        }}
        
        .metric-status {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }}
        
        .status-good {{
            background: #E8F7E8;
            color: #2D7A2D;
        }}
        
        .status-warning {{
            background: #FFF3E0;
            color: #E65100;
        }}
        
        .status-danger {{
            background: #FFEBEE;
            color: #C62828;
        }}
        
        /* Charts */
        .chart-container {{
            margin: 32px 0;
            text-align: center;
        }}
        
        .chart-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        }}
        
        /* Analysis Section */
        .analysis-section {{
            margin: 24px 0;
        }}
        
        .analysis-section h4 {{
            font-size: 20px;
            font-weight: 600;
            color: #007AFF;
            margin-bottom: 16px;
            padding-bottom: 8px;
            border-bottom: 2px solid #F2F2F7;
        }}
        
        .analysis-content {{
            font-size: 16px;
            line-height: 1.7;
            color: #515154;
        }}
        
        .analysis-content p {{
            margin-bottom: 12px;
        }}
        
        /* Responsive Design */
        @media (max-width: 768px) {{
            .hero h2 {{
                font-size: 36px;
            }}
            
            .card {{
                padding: 24px;
                margin-bottom: 24px;
            }}
            
            .metrics-grid {{
                grid-template-columns: 1fr;
                gap: 16px;
            }}
            
            .container {{
                padding: 0 16px;
            }}
        }}
        
        /* Smooth Animations */
        .card, .metric-card {{
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        
        .card:hover, .metric-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
        }}
        
        /* Loading animation for charts */
        @keyframes fade-in {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .chart-container {{
            animation: fade-in 0.6s ease-out;
        }}
    </style>
</head>
<body>
    <!-- Header -->
    <header class="header">
        <div class="container">
            <h1>CGM Clinical Analysis</h1>
            <p class="subtitle">Continuous Glucose Monitoring Report</p>
        </div>
    </header>

    <!-- Hero Section -->
    <section class="hero">
        <div class="container">
            <h2>Clinical Assessment</h2>
            <p class="date">Generated {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
            
            <div class="status-badge">{clinical_status}</div>
            <p class="status-description">{status_description}</p>
        </div>
    </section>

    <!-- Key Metrics -->
    <section class="container">
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-title">Mean Glucose</div>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div class="metric-period">7-Day</div>
                        <div class="metric-value">{float(s7['MeanGlucose']):.0f}</div>
                        <div class="metric-target">mg/dL</div>
                    </div>
                    <div>
                        <div class="metric-period">30-Day</div>
                        <div class="metric-value">{float(s30['MeanGlucose']):.0f}</div>
                        <div class="metric-target">mg/dL</div>
                    </div>
                </div>
                <div class="metric-status {'status-good' if 70 <= float(s7['MeanGlucose']) <= 180 and 70 <= float(s30['MeanGlucose']) <= 180 else 'status-danger'}">
                    Target: 70-180 mg/dL
                </div>
            </div>
            
            <div class="metric-card">
                <div class="metric-title">Time in Range</div>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div class="metric-period">7-Day</div>
                        <div class="metric-value">{float(s7['AvgPercentInRange']):.0f}%</div>
                    </div>
                    <div>
                        <div class="metric-period">30-Day</div>
                        <div class="metric-value">{float(s30['AvgPercentInRange']):.0f}%</div>
                    </div>
                </div>
                <div class="metric-status {'status-good' if float(s7['AvgPercentInRange']) >= 70 and float(s30['AvgPercentInRange']) >= 70 else 'status-danger'}">
                    Target: >70%
                </div>
            </div>
            
            <div class="metric-card">
                <div class="metric-title">Time Above Range</div>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div class="metric-period">7-Day</div>
                        <div class="metric-value">{float(s7['AvgPercentHigh']):.0f}%</div>
                    </div>
                    <div>
                        <div class="metric-period">30-Day</div>
                        <div class="metric-value">{float(s30['AvgPercentHigh']):.0f}%</div>
                    </div>
                </div>
                <div class="metric-status {'status-good' if float(s7['AvgPercentHigh']) <= 25 and float(s30['AvgPercentHigh']) <= 25 else 'status-danger'}">
                    Target: &lt;25%
                </div>
            </div>
            
            <div class="metric-card">
                <div class="metric-title">Time Below Range</div>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div class="metric-period">7-Day</div>
                        <div class="metric-value">{float(s7['AvgPercentLow']):.1f}%</div>
                    </div>
                    <div>
                        <div class="metric-period">30-Day</div>
                        <div class="metric-value">{float(s30['AvgPercentLow']):.1f}%</div>
                    </div>
                </div>
                <div class="metric-status {'status-good' if float(s7['AvgPercentLow']) <= 4 and float(s30['AvgPercentLow']) <= 4 else 'status-warning'}">
                    Target: &lt;4%
                </div>
            </div>
        </div>
    </section>

    <!-- Charts Section -->
    <section class="container">'''
        
        # Add glucose trend chart
        html_content += f'''
        <div class="card">
            <h3>Glucose Trend Analysis</h3>
            <div class="chart-container">'''
        
        if glucose_chart:
            html_content += f'<img src="{glucose_chart}" alt="Glucose Trend Chart" />'
        else:
            html_content += '<p style="color: #86868B; text-align: center; padding: 40px;">Chart not available</p>'
        
        html_content += '''
            </div>
        </div>'''
        
        # Add time in range chart
        html_content += f'''
        <div class="card">
            <h3>Daily Time in Range Distribution</h3>
            <div class="chart-container">'''
        
        if time_range_chart:
            html_content += f'<img src="{time_range_chart}" alt="Time in Range Chart" />'
        else:
            html_content += '<p style="color: #86868B; text-align: center; padding: 40px;">Chart not available</p>'
        
        html_content += '''
            </div>
        </div>'''
        
        # Add summary comparison chart
        html_content += f'''
        <div class="card">
            <h3>Period Comparison</h3>
            <div class="chart-container">'''
        
        if summary_chart:
            html_content += f'<img src="{summary_chart}" alt="Summary Comparison Chart" />'
        else:
            html_content += '<p style="color: #86868B; text-align: center; padding: 40px;">Chart not available</p>'
        
        html_content += '''
            </div>
        </div>'''
        
        # Add clinical analysis
        html_content += '''
        <div class="card">
            <h3>Clinical Analysis</h3>'''
        
        if analysis_sections:
            for section_name, content in analysis_sections.items():
                html_content += f'''
            <div class="analysis-section">
                <h4>{section_name}</h4>
                <div class="analysis-content">'''
                
                # Convert content to paragraphs
                paragraphs = content.split('\n')
                for paragraph in paragraphs:
                    if paragraph.strip():
                        html_content += f'<p>{paragraph.strip()}</p>'
                
                html_content += '''
                </div>
            </div>'''
        else:
            html_content += '''
            <div class="analysis-content">
                <p>Clinical analysis not available</p>
            </div>'''
        
        html_content += '''
        </div>
    </section>

    <!-- Footer -->
    <footer style="padding: 40px 0; text-align: center; color: #86868B; border-top: 1px solid rgba(0, 0, 0, 0.05); margin-top: 60px;">
        <div class="container">
            <p>CGM Clinical Analysis Report • Generated with Mistral Nemo</p>
            <p style="font-size: 14px; margin-top: 8px;">This report is for clinical review and should be interpreted by qualified healthcare professionals.</p>
        </div>
    </footer>

    <script>
        // Smooth scroll animations
        document.addEventListener('DOMContentLoaded', function() {
            const cards = document.querySelectorAll('.card, .metric-card');
            
            const observer = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        entry.target.style.opacity = '1';
                        entry.target.style.transform = 'translateY(0)';
                    }
                });
            }, { threshold: 0.1 });
            
            cards.forEach(card => {
                card.style.opacity = '0';
                card.style.transform = 'translateY(20px)';
                card.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
                observer.observe(card);
            });
        });
    </script>
</body>
</html>'''
        
        # Write HTML file
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Verify file was created
        if os.path.exists(output_filename):
            file_size = os.path.getsize(output_filename)
            print(f"✓ Apple-style HTML website successfully created: {output_filename}")
            print(f"✓ File size: {file_size:,} bytes")
            return True
        else:
            print("✗ HTML file was not created")
            return False
            
    except Exception as e:
        print(f"Error creating HTML website: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_cgm_report_clinical(txt_file, daily_csv, summary_7_csv, summary_30_csv):
    """Generate comprehensive CGM analysis with Apple-style HTML website"""
    
    print("CGM Clinical Analysis with Apple-Style Website")
    print("=" * 50)
    
    # Check if Mistral Nemo is available
    if not check_ollama_model("mistral-nemo"):
        print("\nFalling back to available models...")
        return "Error: Mistral Nemo model not available. Please install it with: ollama pull mistral-nemo"
    
    print("\nReading CGM data files...")
    
    try:
        daily_data = pd.read_csv(daily_csv)
        summary_7_data = pd.read_csv(summary_7_csv)
        summary_30_data = pd.read_csv(summary_30_csv)
        
        print(f"✓ Daily data: {len(daily_data)} records")
        print(f"✓ 7-day summary: {len(summary_7_data)} records")
        print(f"✓ 30-day summary: {len(summary_30_data)} records")
        
    except Exception as e:
        print(f"✗ Error reading CSV files: {e}")
        return f"Error reading CSV files: {e}"
    
    if daily_data.empty or summary_7_data.empty or summary_30_data.empty:
        return "Error: One or more CSV files are empty"
    
    # Show preview
    try:
        s7 = summary_7_data.iloc[0]
        s30 = summary_30_data.iloc[0]
        print(f"\nKey Metrics Preview:")
        print(f"7-day Time in Range: {float(s7['AvgPercentInRange']):.1f}% (Target: >70%)")
        print(f"30-day Time in Range: {float(s30['AvgPercentInRange']):.1f}% (Target: >70%)")
        print(f"7-day Time Above Range: {float(s7['AvgPercentHigh']):.1f}% (Target: <25%)")
        print(f"30-day Time Above Range: {float(s30['AvgPercentHigh']):.1f}% (Target: <25%)")
    except Exception as e:
        print(f"Warning: Could not display preview: {e}")
    
    try:
        # Format data with clinical flags
        print("\nFormatting clinical data...")
        summary_7_text, summary_30_text, daily_text = format_data_for_llm_strict(
            daily_data, summary_7_data, summary_30_data
        )
        
        # Create clinical assessment prompt
        prompt = create_clinical_assessment_prompt(summary_7_text, summary_30_text, daily_text)
        
        print("\nSending clinical data to Mistral Nemo for analysis...")
        print("This may take a moment for thorough clinical assessment...")
        
        # Get analysis from Ollama with Mistral Nemo
        response = chat_with_ollama(prompt, model="mistral-nemo")
        
        # Check for errors in response
        if response.startswith("Error:"):
            print(f"✗ LLM Analysis failed: {response}")
            return response
        
        print("✓ Clinical analysis completed")
        
        # Validate clinical response
        validation_issues = validate_clinical_response(response, summary_7_data, summary_30_data)
        
        if validation_issues:
            print("\nCLINICAL VALIDATION WARNINGS:")
            for issue in validation_issues:
                print(f"- {issue}")
            print("\nNote: Review the analysis for accuracy.")
        else:
            print("\n✓ Clinical analysis appears accurate and comprehensive")
        
        # Generate Apple-style HTML website with fixed filename
        html_filename = "CGM_Clinical_Analysis.html"
        
        print(f"\nGenerating Apple-style HTML website: {html_filename}")
        
        html_success = create_apple_style_html(response, daily_data, summary_7_data, summary_30_data, html_filename)
        
        if html_success:
            print(f"✓ HTML website successfully generated: {html_filename}")
            print(f"✓ Open the file in your web browser to view the report")
            print(f"✓ File will be overwritten on subsequent runs")
        else:
            print(f"✗ HTML generation failed")
            
        return response
        
    except Exception as e:
        print(f"✗ Error processing clinical data: {e}")
        import traceback
        traceback.print_exc()
        return f"Error processing clinical data: {e}"

def find_files():
    """Find CGM analysis files in current directory and data subdirectory"""
    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir, 'data')
    
    print(f"Looking for files in: {current_dir}")
    if os.path.exists(data_dir):
        print(f"Also checking data directory: {data_dir}")
    
    files = {
        'txt_file': 'cgm_analysis_report.txt',
        'daily_csv': 'daily_cgm_summaries.csv',
        'summary_7_csv': '7_day_summary.csv',
        'summary_30_csv': '30_day_summary.csv'
    }
    
    required_files = ['daily_csv', 'summary_7_csv', 'summary_30_csv']
    
    found_files = {}
    missing_files = []
    
    for file_type, filename in files.items():
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
                print(f"? Optional file not found: {filename}")
                found_files[file_type] = None
    
    return found_files, missing_files

def debug_data_structure(csv_file):
    """Debug function to inspect CSV structure"""
    try:
        df = pd.read_csv(csv_file)
        print(f"\nDEBUG - {csv_file}:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("First few rows:")
        print(df.head())
        print("Data types:")
        print(df.dtypes)
        if not df.empty:
            print("Sample values from first row:")
            for col in df.columns:
                print(f"  {col}: {df.iloc[0][col]}")
        print("\n" + "="*50)
        return df
    except Exception as e:
        print(f"✗ Error reading {csv_file}: {e}")
        return None

def main():
    """Main function with Apple-style HTML website generation"""
    print("CGM Clinical Analysis Tool - Apple-Style Website")
    print("=" * 60)
    print("Beautiful HTML website with Apple's design system")
    print("=" * 60)
    
    # Find files
    found_files, missing_files = find_files()
    
    if missing_files:
        print(f"\n✗ Missing required files: {missing_files}")
        print("Please ensure the following files are present:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nFiles should be in current directory or ./data/ subdirectory")
        return
    
    # Validate data structure
    print("\nValidating data structure:")
    data_valid = True
    
    for file_type, filepath in found_files.items():
        if filepath and file_type.endswith('_csv'):
            try:
                df = pd.read_csv(filepath)
                if df.empty:
                    print(f"✗ {filepath}: File is empty")
                    data_valid = False
                else:
                    print(f"✓ {filepath}: {df.shape[0]} rows, {df.shape[1]} columns")
                    
                    # Check for required columns
                    if file_type == 'daily_csv':
                        required_cols = ['Date', 'MeanGlucose']
                        missing_cols = [col for col in required_cols if col not in df.columns]
                        if missing_cols:
                            print(f"  ⚠ Missing columns: {missing_cols}")
                    elif file_type in ['summary_7_csv', 'summary_30_csv']:
                        required_cols = ['MeanGlucose', 'AvgPercentInRange', 'AvgPercentHigh', 'AvgPercentLow']
                        missing_cols = [col for col in required_cols if col not in df.columns]
                        if missing_cols:
                            print(f"  ⚠ Missing columns: {missing_cols}")
                            
            except Exception as e:
                print(f"✗ {filepath}: Error - {e}")
                data_valid = False
    
    if not data_valid:
        print("\n✗ Data validation failed. Please check your CSV files.")
        return
    
    # Generate clinical report
    print("\nInitiating clinical analysis with Apple-style website generation...")
    print("This process will:")
    print("1. Format clinical data with target violations")
    print("2. Send data to Mistral Nemo for analysis") 
    print("3. Validate the clinical assessment")
    print("4. Generate beautiful charts with Apple design")
    print("5. Create responsive HTML website")
    print("\nStarting analysis...")
    
    report = generate_cgm_report_clinical(
        found_files['txt_file'],
        found_files['daily_csv'], 
        found_files['summary_7_csv'],
        found_files['summary_30_csv']
    )
    
    print("\n" + "=" * 60)
    print("CLINICAL ANALYSIS REPORT")
    print("=" * 60)
    print(report)
    print("\n" + "=" * 60)
    
    if not report.startswith("Error:"):
        print("✓ Analysis complete. Open the generated HTML file in your web browser!")
        print("✓ The website features Apple's design system with:")
        print("  • Clean, modern interface")
        print("  • Interactive animations")
        print("  • Responsive design for all devices")
        print("  • Beautiful data visualizations")
        print("  • Professional clinical presentation")
    else:
        print("✗ Analysis failed. Check error messages above.")

if __name__ == "__main__":
    main()