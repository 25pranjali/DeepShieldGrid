import os
import zipfile
import io
import pandas as pd
import numpy as np

def analyze_df(name, df, label_col):
    print("=" * 60)
    print(f"DATASET: {name}")
    print("=" * 60)
    print(f"Rows: {df.shape[0]:,}, Columns: {df.shape[1]}")
    print(f"Columns: {list(df.columns)}")
    print(f"Target column: {label_col}")
    if label_col in df.columns:
        print(f"Class distribution:\n{df[label_col].value_counts(dropna=False)}")
    else:
        print(f"Target column '{label_col}' not found!")
    
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    print(f"Missing values: {missing_cols.to_dict() if len(missing_cols) > 0 else 'None'}")
    
    dup_count = df.duplicated().sum()
    print(f"Duplicates: {dup_count:,} ({dup_count/len(df)*100:.2f}%)")
    
    const_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    print(f"Constant/single-value columns: {const_cols}")
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    print(f"Numerical cols count: {len(num_cols)}")
    print(f"Categorical cols count: {len(cat_cols)}, names: {cat_cols}")

# 1. Clean_FDI_TSA_Combined.csv
p1 = 'datasets/Clean_FDI_TSA_Combined.csv'
if os.path.exists(p1):
    df1 = pd.read_csv(p1, low_memory=False)
    analyze_df('PMU FDI/TSA Dataset (Clean_FDI_TSA_Combined.csv)', df1, 'attack_type')

# 2. archive (1).zip Train.csv
p2 = 'datasets/archive (1).zip'
if os.path.exists(p2):
    with zipfile.ZipFile(p2, 'r') as z:
        df_train = pd.read_csv(io.BytesIO(z.read('Train.csv')))
        analyze_df('IEC 61850 Substation GOOSE/SV Dataset (Train.csv)', df_train, 'class')
        df_test = pd.read_csv(io.BytesIO(z.read('Test.csv')))
        analyze_df('IEC 61850 Substation GOOSE/SV Dataset (Test.csv)', df_test, 'class')

# 3. MSU/ORNL
p3 = 'datasets/archive.zip'
if os.path.exists(p3):
    with zipfile.ZipFile(p3, 'r') as z:
        df_msu = pd.read_csv(io.BytesIO(z.read('binaryAllNaturalPlusNormalVsAttacks/data1.csv')))
        analyze_df('MSU/ORNL Power System Dataset (Sample data1.csv)', df_msu, 'marker')

# 4. IEC-104 with IOA & without IOA
for fn in ['datasets/data_with_ioa.csv', 'datasets/data_without_ioa.csv']:
    if os.path.exists(fn):
        df_iec = pd.read_csv(fn, nrows=100000, low_memory=False)
        analyze_df(f'IEC-104 SCADA Dataset ({os.path.basename(fn)}) [First 100k rows sample]', df_iec, 'label')
