# Prerequisites
Ini khusus bagian development metode, model AI
*   **Python:** `3.11` (Recommended: **3.11.9**)
*   **Package Manager:** `pip`

# Pipeline
![Flowchart](image/Algorithm.png)

# Training

finBERT : https://colab.research.google.com/drive/1BCSbqvqGkYqZ0-Gwt9d_JLBNY6-i2UUS?usp=sharing

XGBoost : https://colab.research.google.com/drive/1kOsba0z-BVFObEBp0HQi4TPQxfHWrvVs?usp=sharing

# Models

https://drive.google.com/drive/folders/1q6NxtH5zROvEvAEWO0ihy320rxWmSiUn?usp=sharing

# How to run?

## Full 
1. Install all requirements
```bash
pip install -r requirements.txt
```
2. run in command prompt
```
cd code/pipeline
python pipeline.py prepare
cd ../..
```
1. Upload the `MCS_features.csv` into a the google collab. (Use the XGBoost Google collab link)
2. Download all the models into `models/XGBoost`
3. Open the link for the finBERT training in google collab
4. Run all the cell in the google collab
5. Download and unzip the `model_finetuned.zip` into `models/finBERT` (Make sure it's not in a nested folder)
6. Run in command prompt
```
cd code/pipeline
python pipeline.py score
cd ../..
```
Note : I recommend to just download the models in the models GDrive link
## After models download
1. Install all requirements
```bash
pip install -r requirements.txt
```
2. Download all models
3. run in command prompt
```
cd code/pipeline
python pipeline.py score
cd ../..
```



# Outputs

## Output
1. `MCS_raw.csv` : File Master Calender Spline Raw
2. `MCS_report.csv` : File Master Calender Spline that have a health score from real data.
3. `MCS_prdedict.csv` : File Master Calender Spline with predicted data with XGBoost (All Company)
4. `MCS_health.csv` : File Master Calender Spline that have a health score from real and predicted data. 

# Additional

## BERT testing
To test BERT model
```
cd code/test
python finBERT.py
cd ../..
```

