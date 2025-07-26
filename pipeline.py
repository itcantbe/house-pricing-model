import pandas as pd

from data.load_data import load_dataset,initialDataExploration,statisticData
from utils.clean_data import getBadData,getCorrelation,getIQRoutlier,getOutlierFromZscore,getZscore, convertToCategory, updateMissingValue, dataVerification,getDuplicateValue
from utils.visualize_data import histogramPlot,boxPlot, scatterWRTvar, boxCompWRTvar, makeScatterMatrix, scatterLogPlot, histroLogPlot


class MLWorkflow:
    def __init__(self):
        self.df = None
        self.initialDataExploration = None
        self.statisticData = None
        self.verification = None
        self.bad_data = None
        self.zOutl = None
        self.iqrOut = None
        self.cor = None

    def load_data(self):
        self.df = load_dataset()
        self.initialDataExploration = initialDataExploration(self.df)
        self.statisticData = statisticData(self.df)

    def clean_data(self):
        self.verification = dataVerification(self.df)
        self.bad_data = getBadData(self.df)
        updateMissingValue(self.df,self.bad_data,self.df.select_dtypes(include=["float64"]))
        convertToCategory(self.df,"ocean_proximity")
        zScore = getZscore(self.df.select_dtypes(include=["float64"]))
        self.zOutl = getOutlierFromZscore(self.df,zScore)
        self.iqrOut = getIQRoutlier(self.df.select_dtypes(include=["float64"]))
        self.cor = getCorrelation(self.df.select_dtypes(include=["float64"]),"total_bedrooms")
        print(getDuplicateValue(self.df))
    
    def generate_graphs(self):
        histogramPlot(self.df,"longitude")
        histogramPlot(self.df,"latitude")
        histogramPlot(self.df,"housing_median_age")
        histogramPlot(self.df,"total_rooms")
        histogramPlot(self.df,"total_bedrooms")
        histogramPlot(self.df,"population")
        histogramPlot(self.df,"households")
        histogramPlot(self.df,"median_income")
        histogramPlot(self.df,"median_house_value")
        histogramPlot(self.df,"ocean_proximity")
        boxPlot(self.df,"longitude")
        boxPlot(self.df,"latitude")
        boxPlot(self.df,"housing_median_age")
        boxPlot(self.df,"total_rooms")
        boxPlot(self.df,"total_bedrooms")
        boxPlot(self.df,"population")
        boxPlot(self.df,"households")
        boxPlot(self.df,"median_income")
        boxPlot(self.df,"median_house_value")
        boxPlot(self.df,"ocean_proximity")
        scatterWRTvar(self.df,"longitude")
        scatterWRTvar(self.df,"latitude")
        scatterWRTvar(self.df,"housing_median_age")
        scatterWRTvar(self.df,"total_rooms")
        scatterWRTvar(self.df,"total_bedrooms")
        scatterWRTvar(self.df,"population")
        scatterWRTvar(self.df,"households")
        scatterWRTvar(self.df,"median_income")
        scatterWRTvar(self.df,"ocean_proximity")
        boxCompWRTvar(self.df,"longitude",isContinous=True)
        boxCompWRTvar(self.df,"latitude",isContinous=True)
        boxCompWRTvar(self.df,"housing_median_age",isContinous=True)
        boxCompWRTvar(self.df,"total_rooms",isContinous=True)
        boxCompWRTvar(self.df,"total_bedrooms",isContinous=True)
        boxCompWRTvar(self.df,"population",isContinous=True)
        boxCompWRTvar(self.df,"households",isContinous=True)
        boxCompWRTvar(self.df,"median_income",isContinous=True)
        boxCompWRTvar(self.df,"ocean_proximity")
        makeScatterMatrix(self.df)
        scatterLogPlot(self.df, "median_income", "median_house_value")
        histroLogPlot("median_income")

    def run_pipeline(self):
        self.load_data()
        self.clean_data()
        #self.generate_graphs()