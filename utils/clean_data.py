import pandas as pd
import numpy as np

def dataVerification(df):
    dictData = {}
    dictData["Data Types Count"] = df.dtypes.value_counts().to_dict()
    dictData["Null Data Count"] = df.isnull().sum()
    return dictData

def getBadData(df):
    bad_data = df[df.total_bedrooms.isnull()]
    return bad_data

def getCorrelation(df,*col):
    """ if len(col) > 1:
        for i in col:
            cor = df.corr()[i]
            return cor
    else:
        cor = df.corr()[col]
        return cor """
    cor = df.corr()['total_bedrooms']
    return cor
    
def updateMissingValue(df,bad_data,numerical_columns):
    for i,j in zip(bad_data.values,bad_data.index):
        bad_data.loc[j,'total_bedrooms'] = ((numerical_columns[df["households"] == i[6]].mean()['total_bedrooms']).round())
    bad_data.total_bedrooms = bad_data.total_bedrooms.ffill()
    df.total_bedrooms = df.total_bedrooms.fillna(bad_data.total_bedrooms)

def convertToCategory(df,col):
    df[col] = df[col].str.lower()
    df[col] = df[col].str.replace(' ','_')

    # special case because I am lazy
    df[col] = df[col].str.replace('<','less_than_')
    df[col] = df[col].astype('category')

def getZscore(df):
    zscore = (df - df.mean(numeric_only=True)) / df.std()
    return zscore

def getOutlierFromZscore(df,zscore):
    return df[(np.abs(zscore) > 3).any(axis=1)]

def getIQRoutlier(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    # Define the outlier bounds
    iqr_factor = 3 # Common factor
    lower_bound = Q1 - (iqr_factor * IQR)
    upper_bound = Q3 + (iqr_factor * IQR)

    iqrDetectHigh = df[(df > upper_bound).any(axis=1)]
    iqrDetectLow = df[(df < lower_bound).any(axis=1)]
    return iqrDetectHigh,iqrDetectLow

def getDuplicateValue(df):
    df.duplicated().sum()