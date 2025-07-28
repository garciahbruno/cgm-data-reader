import pandas as pd
import requests
from datetime import datetime
import os
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import tempfile
import numpy as np

def chat_with_ollama(prompt, model="llama3.2"):
    """Send prompt to Ollama and get response"""
    try:
        response = requests.post('http://localhost:11434/api/generate',
                               json={
                                   'model': model,
                                   'prompt': prompt,
                                   'stream': False,
                                   'options': {
                                       'temperature': 0.1,
                                       'top_p': 0.9,
                                       'num_predict': 2000
                                   }
                               })
        return response.json()['response']
    except Exception as e:
        return f"Error communicating with Ollama: {e}"

def format_data_for_llm(daily_data, summary_7_data, summary_30_data):
    """Convert DataFrames to clear, structured text that LLMs can understand better"""
    
    if daily_data.empty or summary_7_data.empty or summary_30_data.empty:
        raise ValueError("One or more datasets are empty")
    
    s7 = summary_7_data.iloc[0]
    summary_7_text = f"""7-DAY SUMMARY:
- Mean Glucose: {s7['MeanGlucose']} mg/dL
- Time in Range (70-180): {s7['AvgPercentInRange']}%
- Time Above Range (>180): {s7['AvgPercentHigh']}%
- Time Below Range (<70): {s7['AvgPercentLow']}%
- Glucose Variability (StdDev): {s7.get('GlucoseVariability', 'N/A')}
- Period: {s7.get('StartDate', 'N/A')} to {s7.get('EndDate', 'N/A')}"""

    s30 = summary_30_data.iloc[0]
    summary_30_text = f"""30-DAY SUMMARY:
- Mean Glucose: {s30['MeanGlucose']} mg/dL
- Time in Range (70-180): {s30['AvgPercentInRange']}%
- Time Above Range (>180): {s30['AvgPercentHigh']}%
- Time Below Range (<70): {s30['AvgPercentLow']}%
- Glucose Variability (StdDev): {s30.get('GlucoseVariability', 'N/A')}
- Period: {s30.get('StartDate', 'N/A')} to {s30.get('EndDate', 'N/A')}"""

    daily_text = "DAILY BREAKDOWN (Recent Days):\n"
    daily_sorted = daily_data.sort_values('Date').tail(10)
    
    for _, day in daily_sorted.iterrows():
        date = day['Date']
        mean_glucose = day['MeanGlucose']
        time_in_range = day.get('PercentInRange', 'N/A')
        time_high = day.get('PercentHigh', 'N/A')
        time_low = day.get('PercentLow', 'N/A')
        
        flags = []
        if isinstance(mean_glucose, (int, float)):
            if mean_glucose > 200:
                flags.append("HIGH AVG")
            elif mean_glucose < 80:
                flags.append("LOW AVG")
        
        if isinstance(time_low, (int, float)) and time_low > 4:
            flags.append("EXCESS LOW")
        
        if isinstance(time_high, (int, float)) and time_high > 25:
            flags.append("EXCESS HIGH")
        
        flag_text = f" [{', '.join(flags)}]" if flags else ""
        
        daily_text += f"- {date}: Avg={mean_glucose} mg/dL, InRange={time_in_range}%, High={time_high}%, Low={time_low}%{flag_text}\n"
    
    return summary_7_text, summary_30_text, daily_text

def create_focused_prompt(summary_7_text, summary_30_text, daily_text):
    """Create a focused, clear prompt for the LLM"""
    
    prompt = f"""You are analyzing CGM data for a pediatric patient. Provide a clinical assessment based on ONLY the data provided below. Do not make up numbers or contradict the given data.

{summary_7_text}

{summary_30_text}

{daily_text}

CLINICAL TARGETS (Pediatric):
- Target Range: 70-180 mg/dL
- Time in Range Goal: >70%
- Time Below Range: <4%
- Time Above Range: <25%

Please provide:

1. OVERALL ASSESSMENT: Compare 7-day vs 30-day trends
2. KEY CONCERNS: Any values outside targets (if none, state "No significant concerns identified")
3. PATTERNS NOTED: Daily trends or recurring issues (if none, state "No concerning patterns observed")
4. RECOMMENDATIONS: Only provide recommendations if there are actual concerns or values outside targets. If all metrics meet clinical targets, state "Current management appears effective. Continue current diabetes care plan."

Keep your response concise and use ONLY the exact numbers provided above."""

    return prompt

def create_glucose_trend_chart_apple(daily_data, temp_dir):
    """Create Apple-style glucose trend chart"""
    # Set Apple-style chart appearance
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('white')
    
    # Apple color palette
    apple_blue = '#007AFF'
    apple_green = '#34C759'
    apple_orange = '#FF9500'
    apple_red = '#FF3B30'
    apple_gray = '#8E8E93'
    
    daily_data['Date'] = pd.to_datetime(daily_data['Date'])
    daily_sorted = daily_data.sort_values('Date')
    
    # Clean background
    ax.set_facecolor('white')
    
    # Add subtle target range shading
    ax.axhspan(70, 180, alpha=0.08, color=apple_green, zorder=0)
    ax.axhspan(180, 400, alpha=0.05, color=apple_red, zorder=0)
    ax.axhspan(0, 70, alpha=0.05, color=apple_orange, zorder=0)
    
    # Plot with Apple-style line
    ax.plot(daily_sorted['Date'], daily_sorted['MeanGlucose'], 
            linewidth=3, color=apple_blue, zorder=3)
    
    # Add subtle dots
    ax.scatter(daily_sorted['Date'], daily_sorted['MeanGlucose'], 
               s=40, color=apple_blue, zorder=4, alpha=0.8)
    
    # Clean styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#E5E5EA')
    ax.spines['bottom'].set_color('#E5E5EA')
    
    # Subtle grid
    ax.grid(True, alpha=0.2, color='#C7C7CC', linewidth=0.5)
    
    # Apple-style labels
    ax.set_xlabel('')
    ax.set_ylabel('mg/dL', fontsize=12, color='#3C3C43', fontweight='500')
    ax.tick_params(colors='#3C3C43', labelsize=10)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    chart_path = os.path.join(temp_dir, 'glucose_trend.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return chart_path

def create_time_in_range_chart_apple(daily_data, temp_dir):
    """Create Apple-style time in range chart"""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('white')
    
    # Apple colors
    apple_green = '#34C759'
    apple_orange = '#FF9500'
    apple_red = '#FF3B30'
    
    daily_data['Date'] = pd.to_datetime(daily_data['Date'])
    daily_sorted = daily_data.sort_values('Date').tail(14)
    
    ax.set_facecolor('white')
    
    dates = daily_sorted['Date']
    low = daily_sorted.get('PercentLow', 0)
    in_range = daily_sorted.get('PercentInRange', 0)
    high = daily_sorted.get('PercentHigh', 0)
    
    # Apple-style bars with rounded appearance effect
    width = 0.7
    ax.bar(dates, low, width, label='Below Range', color=apple_orange, alpha=0.9)
    ax.bar(dates, in_range, width, bottom=low, label='In Range', color=apple_green, alpha=0.9)
    ax.bar(dates, high, width, bottom=low+in_range, label='Above Range', color=apple_red, alpha=0.9)
    
    # Clean styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#E5E5EA')
    ax.spines['bottom'].set_color('#E5E5EA')
    
    ax.set_xlabel('')
    ax.set_ylabel('Time (%)', fontsize=12, color='#3C3C43', fontweight='500')
    ax.tick_params(colors='#3C3C43', labelsize=10)
    
    # Subtle grid
    ax.grid(True, alpha=0.2, axis='y', color='#C7C7CC', linewidth=0.5)
    
    # Apple-style legend
    ax.legend(frameon=False, loc='upper right', fontsize=10)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    chart_path = os.path.join(temp_dir, 'time_in_range.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return chart_path

def create_summary_comparison_chart_apple(summary_7_data, summary_30_data, temp_dir):
    """Create Apple-style comparison chart"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    fig.patch.set_facecolor('white')
    
    # Apple colors
    apple_blue = '#007AFF'
    apple_purple = '#5856D6'
    apple_green = '#34C759'
    apple_mint = '#00C7BE'
    apple_orange = '#FF9500'
    apple_yellow = '#FFCC00'
    apple_red = '#FF3B30'
    apple_pink = '#FF2D92'
    
    s7 = summary_7_data.iloc[0]
    s30 = summary_30_data.iloc[0]
    
    periods = ['7 Days', '30 Days']
    
    # Clean background for all subplots
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#E5E5EA')
        ax.spines['bottom'].set_color('#E5E5EA')
        ax.grid(True, alpha=0.15, axis='y', color='#C7C7CC', linewidth=0.5)
        ax.tick_params(colors='#3C3C43', labelsize=9)
    
    # Mean Glucose Comparison
    mean_glucose = [s7['MeanGlucose'], s30['MeanGlucose']]
    bars1 = ax1.bar(periods, mean_glucose, color=[apple_blue, apple_purple], alpha=0.9, width=0.6)
    ax1.axhline(y=180, color=apple_red, linestyle='-', alpha=0.3, linewidth=2)
    ax1.axhline(y=70, color=apple_orange, linestyle='-', alpha=0.3, linewidth=2)
    ax1.set_ylabel('mg/dL', fontsize=10, color='#3C3C43', fontweight='500')
    ax1.set_ylim(0, max(mean_glucose) * 1.2)
    
    for bar, value in zip(bars1, mean_glucose):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3, 
                f'{value:.0f}', ha='center', va='bottom', fontweight='600', fontsize=11, color='#3C3C43')
    
    # Time in Range Comparison
    tir = [s7['AvgPercentInRange'], s30['AvgPercentInRange']]
    bars2 = ax2.bar(periods, tir, color=[apple_green, apple_mint], alpha=0.9, width=0.6)
    ax2.axhline(y=70, color=apple_green, linestyle='-', alpha=0.3, linewidth=2)
    ax2.set_ylabel('Time in Range (%)', fontsize=10, color='#3C3C43', fontweight='500')
    ax2.set_ylim(0, 100)
    
    for bar, value in zip(bars2, tir):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{value:.0f}%', ha='center', va='bottom', fontweight='600', fontsize=11, color='#3C3C43')
    
    # Time Above Range
    thr = [s7['AvgPercentHigh'], s30['AvgPercentHigh']]
    bars3 = ax3.bar(periods, thr, color=[apple_red, apple_pink], alpha=0.9, width=0.6)
    ax3.axhline(y=25, color=apple_red, linestyle='-', alpha=0.3, linewidth=2)
    ax3.set_ylabel('Time Above Range (%)', fontsize=10, color='#3C3C43', fontweight='500')
    ax3.set_ylim(0, max(50, max(thr) * 1.2))
    
    for bar, value in zip(bars3, thr):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{value:.0f}%', ha='center', va='bottom', fontweight='600', fontsize=11, color='#3C3C43')
    
    # Time Below Range
    tlr = [s7['AvgPercentLow'], s30['AvgPercentLow']]
    bars4 = ax4.bar(periods, tlr, color=[apple_orange, apple_yellow], alpha=0.9, width=0.6)
    ax4.axhline(y=4, color=apple_orange, linestyle='-', alpha=0.3, linewidth=2)
    ax4.set_ylabel('Time Below Range (%)', fontsize=10, color='#3C3C43', fontweight='500')
    ax4.set_ylim(0, max(10, max(tlr) * 1.5))
    
    for bar, value in zip(bars4, tlr):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
                f'{value:.1f}%', ha='center', va='bottom', fontweight='600', fontsize=11, color='#3C3C43')
    
    # Remove x-axis labels for cleaner look
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlabel('')
    
    plt.tight_layout()
    chart_path = os.path.join(temp_dir, 'summary_comparison.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return chart_path

def assess_clinical_targets(summary_7_data, summary_30_data):
    """Assess if patient meets clinical targets"""
    s7 = summary_7_data.iloc[0]
    s30 = summary_30_data.iloc[0]
    
    targets_met = {
        'tir_7day': s7['AvgPercentInRange'] > 70,
        'tir_30day': s30['AvgPercentInRange'] > 70,
        'high_7day': s7['AvgPercentHigh'] < 25,
        'high_30day': s30['AvgPercentHigh'] < 25,
        'low_7day': s7['AvgPercentLow'] < 4,
        'low_30day': s30['AvgPercentLow'] < 4,
        'glucose_7day': 70 <= s7['MeanGlucose'] <= 180,
        'glucose_30day': 70 <= s30['MeanGlucose'] <= 180
    }
    
    all_targets_met = all(targets_met.values())
    critical_targets_met = (
        targets_met['tir_7day'] and targets_met['tir_30day'] and
        targets_met['low_7day'] and targets_met['low_30day']
    )
    
    return all_targets_met, critical_targets_met, targets_met

def parse_analysis_text(analysis_text):
    """Parse the analysis text into structured sections"""
    sections = {
        'overall_assessment': '',
        'key_concerns': [],
        'patterns_noted': [],
        'recommendations': []
    }
    
    current_section = None
    lines = analysis_text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Identify section headers
        if 'OVERALL ASSESSMENT' in line.upper():
            current_section = 'overall_assessment'
            continue
        elif 'KEY CONCERNS' in line.upper():
            current_section = 'key_concerns'
            continue
        elif 'PATTERNS NOTED' in line.upper():
            current_section = 'patterns_noted'
            continue
        elif 'RECOMMENDATIONS' in line.upper():
            current_section = 'recommendations'
            continue
        
        # Add content to appropriate section
        if current_section == 'overall_assessment':
            sections['overall_assessment'] += line + ' '
        elif current_section in ['key_concerns', 'patterns_noted', 'recommendations']:
            # Remove numbering and formatting artifacts
            clean_line = line
            if line.startswith(('1.', '2.', '3.', '4.', '5.')):
                clean_line = line[2:].strip()
            if clean_line.startswith('**') and clean_line.endswith('**'):
                clean_line = clean_line[2:-2]
            if clean_line:
                sections[current_section].append(clean_line)
    
    return sections

def create_pdf_report(analysis_text, daily_data, summary_7_data, summary_30_data, output_filename):
    """Create an Apple-inspired PDF report with clean design"""
    
    # Create temporary directory for charts
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Generate charts with Apple-style colors
        print("Generating charts...")
        glucose_trend_path = create_glucose_trend_chart_apple(daily_data, temp_dir)
        time_range_path = create_time_in_range_chart_apple(daily_data, temp_dir)
        summary_comparison_path = create_summary_comparison_chart_apple(summary_7_data, summary_30_data, temp_dir)
        
        # Create PDF with Apple-inspired styling
        print("Creating PDF report...")
        doc = SimpleDocTemplate(
            output_filename, 
            pagesize=letter, 
            topMargin=0.8*inch,
            bottomMargin=0.8*inch,
            leftMargin=0.8*inch,
            rightMargin=0.8*inch
        )
        story = []
        
        # Apple-inspired color palette
        apple_blue = colors.Color(0.0, 0.48, 1.0)  # SF Blue
        apple_gray = colors.Color(0.56, 0.56, 0.58)  # SF Gray
        apple_light_gray = colors.Color(0.96, 0.96, 0.96)  # Light gray background
        apple_dark_gray = colors.Color(0.17, 0.17, 0.18)  # Dark gray text
        
        # Define Apple-inspired styles
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'AppleTitle',
            parent=styles['Heading1'],
            fontSize=32,
            spaceAfter=8,
            spaceBefore=0,
            alignment=TA_LEFT,
            textColor=apple_dark_gray,
            fontName='Helvetica-Bold'
        )
        
        subtitle_style = ParagraphStyle(
            'AppleSubtitle',
            parent=styles['Normal'],
            fontSize=16,
            spaceAfter=40,
            spaceBefore=0,
            alignment=TA_LEFT,
            textColor=apple_gray,
            fontName='Helvetica'
        )
        
        section_heading_style = ParagraphStyle(
            'AppleSectionHeading',
            parent=styles['Heading2'],
            fontSize=20,
            spaceAfter=16,
            spaceBefore=32,
            textColor=apple_dark_gray,
            fontName='Helvetica-Bold'
        )
        
        subsection_style = ParagraphStyle(
            'AppleSubsection',
            parent=styles['Heading3'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=20,
            textColor=apple_blue,
            fontName='Helvetica-Bold'
        )
        
        body_style = ParagraphStyle(
            'AppleBody',
            parent=styles['Normal'],
            fontSize=12,
            spaceAfter=8,
            spaceBefore=0,
            textColor=apple_dark_gray,
            fontName='Helvetica',
            leading=18
        )
        
        bullet_style = ParagraphStyle(
            'AppleBullet',
            parent=styles['Normal'],
            fontSize=12,
            spaceAfter=6,
            spaceBefore=0,
            leftIndent=20,
            textColor=apple_dark_gray,
            fontName='Helvetica',
            leading=16
        )
        
        # Title section
        story.append(Paragraph("CGM Analysis", title_style))
        story.append(Paragraph(f"Report generated {datetime.now().strftime('%B %d, %Y')}", subtitle_style))
        
        # Key metrics card
        s7 = summary_7_data.iloc[0]
        s30 = summary_30_data.iloc[0]
        
        story.append(Paragraph("Key Metrics", section_heading_style))
        
        # Create a clean metrics table
        metrics_data = [
            ['', '7 Days', '30 Days', 'Target'],
            ['Average Glucose', f"{s7['MeanGlucose']:.0f} mg/dL", f"{s30['MeanGlucose']:.0f} mg/dL", "70-180 mg/dL"],
            ['Time in Range', f"{s7['AvgPercentInRange']:.0f}%", f"{s30['AvgPercentInRange']:.0f}%", "> 70%"],
            ['Time Above Range', f"{s7['AvgPercentHigh']:.0f}%", f"{s30['AvgPercentHigh']:.0f}%", "< 25%"],
            ['Time Below Range', f"{s7['AvgPercentLow']:.0f}%", f"{s30['AvgPercentLow']:.0f}%", "< 4%"]
        ]
        
        metrics_table = Table(metrics_data, colWidths=[2*inch, 1.2*inch, 1.2*inch, 1.4*inch])
        metrics_table.setStyle(TableStyle([
            # Header row
            ('BACKGROUND', (0, 0), (-1, 0), apple_light_gray),
            ('TEXTCOLOR', (0, 0), (-1, 0), apple_dark_gray),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            
            # Data rows
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 11),
            ('TEXTCOLOR', (0, 1), (-1, -1), apple_dark_gray),
            
            # Remove grid lines for cleaner look
            ('LINEBELOW', (0, 0), (-1, 0), 0.5, apple_gray),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ]))
        
        story.append(metrics_table)
        story.append(Spacer(1, 30))
        
        # Charts section
        story.append(Paragraph("Glucose Trends", section_heading_style))
        story.append(Image(glucose_trend_path, width=6*inch, height=3*inch))
        story.append(Spacer(1, 20))
        
        story.append(Paragraph("Daily Distribution", section_heading_style))
        story.append(Image(time_range_path, width=6*inch, height=3*inch))
        story.append(PageBreak())
        
        story.append(Paragraph("Period Comparison", section_heading_style))
        story.append(Image(summary_comparison_path, width=6*inch, height=3.6*inch))
        story.append(Spacer(1, 30))
        
        # Parse and format clinical analysis
        parsed_analysis = parse_analysis_text(analysis_text)
        
        # Assess if targets are met
        all_targets_met, critical_targets_met, targets_breakdown = assess_clinical_targets(summary_7_data, summary_30_data)
        
        story.append(Paragraph("Clinical Assessment", section_heading_style))
        
        # Add status indicator at the top
        if all_targets_met:
            status_text = "Clinical Status: All targets met - Excellent glucose management"
            status_color = colors.Color(0.2, 0.7, 0.3)  # Green
        elif critical_targets_met:
            status_text = "Clinical Status: Key safety targets met - Good glucose management"
            status_color = colors.Color(1.0, 0.6, 0.0)  # Orange
        else:
            status_text = "Clinical Status: Some targets not met - Requires attention"
            status_color = colors.Color(0.9, 0.3, 0.3)  # Red
        
        status_style = ParagraphStyle(
            'StatusStyle',
            parent=body_style,
            fontSize=13,
            spaceAfter=20,
            textColor=status_color,
            fontName='Helvetica-Bold'
        )
        story.append(Paragraph(status_text, status_style))
        
        # Overall Assessment
        if parsed_analysis['overall_assessment']:
            story.append(Paragraph("Overview", subsection_style))
            story.append(Paragraph(parsed_analysis['overall_assessment'].strip(), body_style))
            story.append(Spacer(1, 16))
        
        # Key Concerns - only show if there are actual concerns
        if parsed_analysis['key_concerns']:
            has_real_concerns = not any(
                'no significant concerns' in concern.lower() or 
                'no concerns identified' in concern.lower()
                for concern in parsed_analysis['key_concerns']
            )
            
            if has_real_concerns:
                story.append(Paragraph("Key Concerns", subsection_style))
                for concern in parsed_analysis['key_concerns']:
                    story.append(Paragraph(f"• {concern}", bullet_style))
                story.append(Spacer(1, 16))
            else:
                story.append(Paragraph("Key Concerns", subsection_style))
                story.append(Paragraph("No significant concerns identified with current glucose management.", body_style))
                story.append(Spacer(1, 16))
        
        # Patterns - only show if there are concerning patterns
        if parsed_analysis['patterns_noted']:
            has_concerning_patterns = not any(
                'no concerning patterns' in pattern.lower() or 
                'no patterns observed' in pattern.lower()
                for pattern in parsed_analysis['patterns_noted']
            )
            
            if has_concerning_patterns:
                story.append(Paragraph("Patterns Observed", subsection_style))
                for pattern in parsed_analysis['patterns_noted']:
                    story.append(Paragraph(f"• {pattern}", bullet_style))
                story.append(Spacer(1, 16))
        
        # Recommendations - conditional based on clinical status
        if parsed_analysis['recommendations']:
            needs_recommendations = not any(
                'current management appears effective' in rec.lower() or
                'continue current' in rec.lower()
                for rec in parsed_analysis['recommendations']
            )
            
            if needs_recommendations and not all_targets_met:
                story.append(Paragraph("Recommendations", subsection_style))
                for i, rec in enumerate(parsed_analysis['recommendations'], 1):
                    story.append(Paragraph(f"{i}. {rec}", bullet_style))
            else:
                story.append(Paragraph("Recommendations", subsection_style))
                if all_targets_met:
                    story.append(Paragraph("Excellent glucose management. Continue current diabetes care plan and monitoring schedule.", body_style))
                elif critical_targets_met:
                    story.append(Paragraph("Good glucose control with key safety targets met. Continue current management with regular monitoring.", body_style))
                else:
                    # Show specific recommendations if targets not met
                    for i, rec in enumerate(parsed_analysis['recommendations'], 1):
                        story.append(Paragraph(f"{i}. {rec}", bullet_style))
        
        # Build PDF
        doc.build(story)
        print(f"PDF report saved as: {output_filename}")
        
    except Exception as e:
        print(f"Error creating PDF: {e}")
    
    finally:
        # Clean up temporary files
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

def validate_llm_response(response, daily_data, summary_7_data, summary_30_data):
    """Check if LLM response contains hallucinated numbers"""
    
    actual_7day_mean = summary_7_data['MeanGlucose'].iloc[0]
    actual_7day_tir = summary_7_data['AvgPercentInRange'].iloc[0]
    actual_30day_mean = summary_30_data['MeanGlucose'].iloc[0]
    actual_30day_tir = summary_30_data['AvgPercentInRange'].iloc[0]
    
    issues = []
    
    if str(actual_7day_mean) not in response:
        issues.append(f"Missing 7-day mean glucose ({actual_7day_mean})")
    
    if str(actual_7day_tir) not in response:
        issues.append(f"Missing 7-day time-in-range ({actual_7day_tir}%)")
    
    import re
    numbers_in_response = re.findall(r'\b\d+\.?\d*\b', response)
    glucose_values = [float(n) for n in numbers_in_response if 50 <= float(n) <= 400]
    
    actual_glucose_values = [
        actual_7day_mean, actual_30day_mean,
        summary_7_data['AvgPercentInRange'].iloc[0],
        summary_30_data['AvgPercentInRange'].iloc[0]
    ]
    
    hallucinated_values = [v for v in glucose_values if not any(abs(v - actual) < 1 for actual in actual_glucose_values)]
    
    if hallucinated_values:
        issues.append(f"Potential hallucinated values: {hallucinated_values}")
    
    return issues

def generate_cgm_report(txt_file, daily_csv, summary_7_csv, summary_30_csv):
    """Generate comprehensive CGM analysis report with PDF export"""
    
    print("Reading CGM data files...")
    
    try:
        daily_data = pd.read_csv(daily_csv)
        summary_7_data = pd.read_csv(summary_7_csv)
        summary_30_data = pd.read_csv(summary_30_csv)
    except Exception as e:
        return f"Error reading CSV files: {e}"
    
    if daily_data.empty or summary_7_data.empty or summary_30_data.empty:
        return "Error: One or more CSV files are empty"
    
    print(f"Data loaded: {len(daily_data)} daily records, 7-day summary, 30-day summary")
    
    try:
        # Format data for LLM
        summary_7_text, summary_30_text, daily_text = format_data_for_llm(
            daily_data, summary_7_data, summary_30_data
        )
        
        # Create focused prompt
        prompt = create_focused_prompt(summary_7_text, summary_30_text, daily_text)
        
        print("Sending to Ollama...")
        
        # Get analysis from Ollama
        response = chat_with_ollama(prompt)
        
        # Validate response
        issues = validate_llm_response(response, daily_data, summary_7_data, summary_30_data)
        
        if issues:
            print("\nWARNING - Potential issues detected:")
            for issue in issues:
                print(f"- {issue}")
            print("\nYou may want to regenerate the report.")
        
        # Generate PDF report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"CGM_Analysis_Report_{timestamp}.pdf"
        
        print(f"Generating PDF report: {pdf_filename}")
        create_pdf_report(response, daily_data, summary_7_data, summary_30_data, pdf_filename)
        
        return response
        
    except Exception as e:
        return f"Error processing data: {e}"

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
        print("\n" + "="*50)
        return df
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
        return None

def find_files():
    """Find CGM analysis files in current directory and data subdirectory"""
    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir, 'data')
    
    print(f"Looking for files in: {current_dir}")
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
            print(f"Found: {current_path}")
        elif os.path.exists(data_path):
            found_files[file_type] = data_path
            print(f"Found: {data_path}")
        else:
            if file_type in required_files:
                missing_files.append(filename)
                print(f"Missing: {filename} (checked both . and ./data/)")
            else:
                print(f"Optional file not found: {filename}")
                found_files[file_type] = None
    
    return found_files, missing_files

def main():
    """Main function with PDF export capability"""
    print("CGM Data Clinical Analysis Tool with PDF Export")
    print("=" * 50)
    
    # Find files
    found_files, missing_files = find_files()
    
    if missing_files:
        print(f"\nMissing required files: {missing_files}")
        return
    
    # Debug: Inspect data structure first
    print("\nDEBUGGING DATA STRUCTURE:")
    for file_type, filepath in found_files.items():
        if filepath and file_type.endswith('_csv'):
            debug_data_structure(filepath)
    
    # Generate report
    print("Generating analysis...")
    report = generate_cgm_report(
        found_files['txt_file'],
        found_files['daily_csv'], 
        found_files['summary_7_csv'],
        found_files['summary_30_csv']
    )
    
    print("\n" + "=" * 50)
    print("CLINICAL ANALYSIS REPORT")
    print("=" * 50)
    print(report)

if __name__ == "__main__":
    main()