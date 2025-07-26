import pandas as pd

def load_dataset():
    return pd.read_csv('data/raw/housing.csv')

def initialDataExploration(df):
    dictData = {}
    dictData["Row Count"] = df.shape[0]
    dictData["Column Count"] = df.shape[1]
    dictData["Column Name"] = df.columns
    dictData["Data Types"] = df.dtypes
    return dictData

def statisticData(df):
    dictData = {}
    dictData["Mean"] = df.mean(numeric_only=True)
    dictData["Variance"] = df.var(numeric_only=True)
    dictData["Min"] = df.min(numeric_only=True)
    dictData["Max"] = df.max(numeric_only=True)
    dictData["Mode"] = df.select_dtypes(include=["object"]).mode()
    dictData["Count"] = df.select_dtypes(include=["object"]).value_counts()
    return dictData