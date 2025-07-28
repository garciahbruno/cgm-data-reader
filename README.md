# 🩺 CGM Data Analysis Tool

A comprehensive Python toolkit for analyzing Continuous Glucose Monitoring (CGM) data and generating beautiful, patient-friendly clinical reports using AI.

## 📋 Overview

This tool takes raw CGM data from any sensor format and creates:
- 📊 **Standardized data processing** for different CGM file formats
- 📈 **Visual dashboards** with glucose trends and analytics
- 🤖 **AI-powered clinical reports** using Ollama (local LLM)
- 🌐 **Beautiful HTML reports** that look professional

## 🎯 Features

### 🔄 Data Standardization
- **Universal file support**: Works with CSV files from any CGM sensor
- **Smart column detection**: Automatically identifies time and glucose columns
- **Multiple formats**: Handles different date/time formats and separators
- **Data validation**: Filters out unrealistic glucose values and missing data

### 📊 Analytics & Visualization
- **Daily summaries**: Mean glucose, time in range, variability metrics
- **Period analysis**: 7-day and 30-day trend comparisons
- **Interactive charts**: Professional glucose trend visualizations
- **Key metrics**: Time above/below range, peak/low times, standard deviation

### 🤖 AI Clinical Reports
- **Patient-friendly language**: Easy-to-understand explanations
- **Actionable recommendations**: Specific diet, exercise, and monitoring advice
- **Pattern recognition**: Identifies glucose trends and concerning patterns
- **Clinical insights**: Professional endocrinologist-level analysis

### 🌐 Beautiful Output
- **Modern HTML reports**: Professional medical report styling
- **Embedded charts**: Visual dashboards directly in the report
- **Mobile responsive**: Works on desktop and mobile devices
- **Print-friendly**: Clean formatting for physical copies

## 🚀 Quick Start

### Prerequisites

```bash
# Install required packages
pip install pandas numpy matplotlib seaborn requests
```

### Setup Ollama (for AI analysis)

1. **Install Ollama**: Download from [ollama.ai](https://ollama.ai)
2. **Install a model**:
   ```bash
   ollama pull llama3.2
   ```
3. **Start Ollama**: The service should run automatically

### Usage

1. **Prepare your CGM data**:
   - Place your CGM CSV file in the project directory
   - File should contain time and glucose columns (any names work)

2. **Run the data processing notebook**:
   ```python
   # Update the file path in the notebook
   cgm_file_path = "your_cgm_data.csv"
   
   # Run all cells to generate:
   # - data/daily_cgm_summaries.csv
   # - data/7_day_summary.csv  
   # - data/30_day_summary.csv
   ```

3. **Generate AI clinical report**:
   ```bash
   python llm_data_analysis.py
   ```

4. **View your report**:
   - Open `cgm_clinical_report.html` in any web browser
   - Professional medical report with charts and AI analysis

## 📁 Project Structure

```
cgm-data-reader/
├── data/                          # Generated data files
│   ├── standardized_cgm_data.csv  # Cleaned raw data
│   ├── daily_cgm_summaries.csv    # Daily metrics
│   ├── 7_day_summary.csv          # 7-day analysis
│   └── 30_day_summary.csv         # 30-day analysis
├── data_processing.ipynb          # Main analysis notebook
├── llm_data_analysis.py           # AI report generator
├── cgm_clinical_report.html       # Generated report
└── README.md                      # This file
```

## 📊 Sample Output

### Key Metrics Dashboard
- **Time in Range**: 85% (Goal: >70%)
- **Average Glucose**: 142 mg/dL (Target: 70-180)
- **Time Above Range**: 12% (Goal: <25%)
- **Time Below Range**: 3% (Goal: <4%)

### AI Clinical Insights
> "Your glucose control is **good** overall. The data shows excellent time in range at 85%, which exceeds the clinical target of 70%. However, there were some post-dinner spikes on July 16 and July 18 that we should address..."

### Visual Analytics
- Daily glucose trend charts
- Time-in-range comparisons
- Glucose variability analysis
- Pattern recognition graphs

## 🛠️ Customization

### Modify Glucose Targets
Edit the target ranges in `llm_data_analysis.py`:
```python
# Current targets (mg/dL)
TARGET_LOW = 70
TARGET_HIGH = 180
EXCELLENT_TIR = 70  # Time in range percentage
```

### Change AI Model
Update the model in `llm_data_analysis.py`:
```python
def chat_with_ollama(prompt, model="llama3.2"):
    # Try: "llama3.1", "mistral", "codellama"
```

### Customize Report Styling
Modify the CSS in the `save_report_as_html()` function for different colors, fonts, or layout.

## 🔧 Troubleshooting

### Common Issues

**"No files found"**
- Ensure you've run the data processing notebook first
- Check that CSV files are generated in the `data/` folder

**"Error communicating with Ollama"**
- Verify Ollama is installed and running
- Check that you have a model downloaded: `ollama list`
- Try: `ollama serve` to start the service manually

**"Could not identify columns"**
- Check your CSV file has time and glucose columns
- Ensure data is in proper format (no extra headers)
- Manually specify columns in the `identify_columns()` function if needed

**Charts not displaying**
- Install missing packages: `pip install matplotlib seaborn`
- Ensure your data has valid dates and numeric glucose values

### File Format Requirements

Your CGM data should be a CSV file with:
- **Time column**: Any format (timestamps, dates, etc.)
- **Glucose column**: Numeric values in mg/dL
- **Headers**: Column names (can be anything descriptive)

Example formats that work:
```csv
timestamp,glucose_level
2024-07-15 08:30:00,145
2024-07-15 08:35:00,152
```

```csv
DateTime,SensorGlucose
15Jul24:08:30:00,145
15Jul24:08:35:00,152
```

## 📈 Advanced Features

### Batch Processing
Process multiple patient files by modifying the main script to loop through a directory of CSV files.

### Custom Time Periods
Adjust analysis windows by changing the date range selection:
```python
# Analyze last 14 days instead of 30
start_date = latest_date - pd.Timedelta(days=14)
```

### Additional Metrics
Add custom calculations in the `create_daily_summaries_csv()` function:
- Coefficient of variation
- Dawn phenomenon detection
- Post-meal spike analysis
- Nocturnal glucose patterns

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request with your improvements

## 📄 License

This project is open source. Use responsibly for educational and research purposes.

## ⚠️ Medical Disclaimer

This tool is for educational and informational purposes only. Always consult with healthcare professionals for medical decisions. The AI-generated reports should supplement, not replace, professional medical advice.

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review your file formats and data quality
3. Ensure all dependencies are properly installed

---

**Made with ❤️ for better diabetes management**