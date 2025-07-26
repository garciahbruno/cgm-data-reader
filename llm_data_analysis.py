import json
import requests
from datetime import datetime


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

def read_json_file(filepath):
    """Read and parse JSON file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception as e:
        return f"Error reading JSON file: {e}"

def generate_cgm_report(txt_file, json_file):
    """Generate comprehensive CGM analysis report"""
    
    # Read both files
    print("Reading CGM data files...")
    txt_content = read_txt_file(txt_file)
    json_data = read_json_file(json_file)
    
    if isinstance(txt_content, str) and "Error" in txt_content:
        return txt_content
    if isinstance(json_data, str) and "Error" in json_data:
        return json_data
    
    # Create the comprehensive prompt
    endocrinologist_prompt = f"""
Act as an endocrinologist analyzing a patient's Continuous Glucose Monitoring (CGM) data. Generate a comprehensive clinical report by following these steps:

### **CGM DATA PROVIDED:**

**TEXT REPORT:**
{txt_content}

**STRUCTURED JSON DATA:**
{json.dumps(json_data, indent=2)}

### **ANALYSIS INSTRUCTIONS:**

### **1. Data Extraction & Validation**
- **Verify Inputs**:  
  "Confirm the dataset includes:  
  - Time range (start/end dates)  
  - Average glucose, time-in-range (TIR), hyper/hypo percentages  
  - Daily breakdowns with peaks/lows and variability metrics  
  - Any missing data? Flag gaps with: [Incomplete: {{metric}}]"

### **2. Dynamic Summary**
- **Period Identification**:  
  "Analyzing data from {{analysis_period.start}} to {{analysis_period.end}} ({{total_days}} days):"  
- **Key Metrics**:  
  "Glucose control is {{optimal/moderate/poor}} based on:  
  - Avg Glucose: {{mean_glucose}} mg/dL (Target: 70-180)  
  - TIR: {{time_in_range}}% (Goal >70%)  
  - Hyperglycemia (>180): {{percent_high}}%  
  - Hypoglycemia (<70): {{percent_low}}%  
  - Variability (SD): {{glucose_variability}} mg/dL (Goal <30%)"

### **3. Pattern Detection**
- **Hyperglycemia**:  
  "Top 3 recurring high-glucose patterns:  
  1. {{Time/context}} (e.g., 'Post-dinner spikes averaging {{value}} mg/dL')  
  2. {{Time/context}}  
  3. {{Time/context}}"  
- **Hypoglycemia**:  
  "Most frequent lows: {{time_of_day}} (e.g., '3 AM lows <70 mg/dL on {{X}} days')"  
- **Variability Drivers**:  
  "Highest swings correlate with: {{possible_cause}} (e.g., 'Inconsistent meal timing')"

### **4. Actionable Insights**
- **Medication**:  
  "{{Adjustment}} (e.g., 'Increase basal insulin by 10% if fasting glucose >140')"  
- **Lifestyle**:  
  "{{Recommendation}} (e.g., '15-min post-meal walks to reduce spikes')"  
- **Monitoring**:  
  "Focus on {{time_range}} for next {{days}} (e.g., '2 AM checks for nocturnal lows')"

### **5. Risk Prioritization**
- **Urgent Flags**:  
  "Immediate action needed for:  
  - Glucose <54 mg/dL on {{dates}}  
  - Consecutive highs >250 mg/dL"  
- **Watchlist**:  
  "Monitor: {{trend}} (e.g., 'Weekly rising fasting glucose')"

### **Output Rules**
- **Adapt to Data**: Skip sections if metrics are missing.  
- **Plain Language**: Explain terms like "CV%" or "TIR".  
- **Format in Markdown with clear sections and bullet points**
- **Bold key findings**
- **Use specific data from the provided files**

Generate the complete clinical report now:
"""
    
    print("Analyzing CGM data...")
    print("This may take a moment...")
    
    # Get analysis from Ollama
    report = chat_with_ollama(endocrinologist_prompt)
    
    return report

def save_report(report, output_file="cgm_clinical_report.md"):
    """Save the generated report to a file"""
    try:
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(report)
        print(f"Report saved to: {output_file}")
    except Exception as e:
        print(f"Error saving report: {e}")

def main():
    """Main function to run the CGM analysis"""
    print("CGM Data Clinical Analysis Tool")
    print("=" * 40)
    
    # File paths (update these to match your file names)
    txt_file = "/Users/venyo/ProDev/cgm-data-reader/cgm_analysis_report.txt"
    json_file = "/Users/venyo/ProDev/cgm-data-reader/cgm_structured_data.json"
    
    # Generate the report
    report = generate_cgm_report(txt_file, json_file)
    
    # Print the report
    print("\n" + "=" * 50)
    print("CLINICAL ANALYSIS REPORT")
    print("=" * 50)
    print(report)
    
    # Save to file
    save_report(report)
    
    print("\n" + "=" * 50)
    print("Analysis Complete!")

if __name__ == "__main__":
    main()