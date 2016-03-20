#!/usr/bin/env python

#import required packages
import pandas as pd
import numpy as np
import statsmodels.api as sm
from bokeh.plotting import figure, show, output_file, ColumnDataSource, gridplot
from bokeh.models import HoverTool, BoxSelectTool, WheelZoomTool, ResetTool, PreviewSaveTool
from scipy import stats
from scipy import spatial
from sklearn import linear_model


#define required functions
def clean_up_name(text):
    try:
        return text.strip().replace('\t', ' ')
    except AttributeError:
        return text
#part 1
#import the Health Profiles Programme csv files as dfs
sti = pd.read_csv(
        "/Users/elektra/Desktop/MSc in Data Science/Principles of Data Science/Report/DataHealthProfiles/AcutesexuallytransmittedinfectionsCSV.csv",
        skiprows=19,
        usecols=[1, 2, 9]
)

#change rate to per 1,000 in population
sti['Indicator value'] = sti['Indicator value'] / 100

#change column names
sti.rename(
            columns={'Indicator value': 'RateOfAcuteSTIsPer1000Population'},
            inplace=True
)

#strip column names from spaces before and after
sti.columns = [clean_up_name(col) for col in sti.columns]

#check first rows
print sti.head(5)

#similarly import the other Health Profiles Programme csv files as dfs
drug = pd.read_csv(
        "/Users/elektra/Desktop/MSc in Data Science/Principles of Data Science/Report/DataHealthProfiles/DrugmisuseCSV.csv",
        skiprows=19,
        usecols=[1, 9]
)
drug.rename(columns={'Indicator value': 'EstimatedCrudeRateOfOpiateAndCrackCocaineUsersPer1000Aged15To64'}, inplace=True)
drug.columns = [clean_up_name(col) for col in drug.columns]
print drug.head(5)

teen_preg = pd.read_csv(
        "/Users/elektra/Desktop/MSc in Data Science/Principles of Data Science/Report/DataHealthProfiles/Teenagepregnancyunder18CSV.csv",
        skiprows=19,
        usecols=[1, 9]
)

teen_preg.rename(
                columns={'Indicator value': 'Under18ConceptionRatePer1000FemalesAged15To17'},
                inplace=True
)
teen_preg.columns = [clean_up_name(col) for col in teen_preg.columns]
print teen_preg.head(5)

crime = pd.read_csv(
        "/Users/elektra/Desktop/MSc in Data Science/Principles of Data Science/Report/DataHealthProfiles/ViolentcrimeCSV.csv",
        skiprows=19,
        usecols=[1, 9]
)
crime.rename(
            columns={'Indicator value': 'RateOfRecordedViolenceAgainstThePersonOffences'},
            inplace=True
)
crime.columns = [clean_up_name(col) for col in crime.columns]
print crime.head(5)

#import London DataStore xlsx Wards sheet as df
hh_inc = pd.read_excel(
        "/Users/elektra/Desktop/MSc in Data Science/Principles of Data Science/Report/gla-household-income-estimates-WardLevel.xlsx",
        sheetname='Wards',
        skiprows=2,
        usecols=[0, 1, 2, 3, 15, 28]
)

#change column names
columnTrans = {
    'Unnamed: 15': 'TotalMeanAnnualHouseholdIncomeEstimateFor2013',
    'Unnamed: 28': 'TotalMedianAnnualHouseholdIncomeEstimateFor2013'
}
hh_inc.rename(columns=columnTrans, inplace=True)
hh_inc.columns = [clean_up_name(col) for col in hh_inc.columns]
#check first rows
print hh_inc.head(5)

#part 2
#merge the dataframes into new dataframe
healthDF = pd.merge(sti, drug, on='ONS Code (new)', suffixes=('_sti', '_drug'))
healthDF = pd.merge(healthDF, teen_preg, on='ONS Code (new)', suffixes=('', '_teen_preg'))
healthDF = pd.merge(healthDF, crime, on='ONS Code (new)', suffixes=('', '_crime'))

#check first rows
print healthDF.head(5)

#group hh_inc entries by LAD code & Borough name
grouped_hh_inc1 = (hh_inc
                    .groupby(['LAD code', 'Borough'])[
                    'TotalMeanAnnualHouseholdIncomeEstimateFor2013']
                    .agg(np.average)
                    .reset_index())

grouped_hh_inc2 = (hh_inc
                    .groupby(['LAD code', 'Borough'])[
                    'TotalMedianAnnualHouseholdIncomeEstimateFor2013']
                    .agg(np.median)
                    .reset_index())

grouped_hh_inc = pd.merge(
                            grouped_hh_inc1,
                            grouped_hh_inc2, on=['LAD code', 'Borough']
)

#merge London DataStore & Health Profiles Programme dataframes
#gets data for London Boroughs only
mergedDF = pd.merge(grouped_hh_inc, healthDF,
                        right_on='ONS Code (new)',
                        left_on='LAD code',
                        how='inner')

#delete unwanted fields
del(mergedDF['ONS Code (new)'], mergedDF['Area Name'])

#check first rows
print mergedDF.head(5)

#part 3
#identify 'outliers' of 1.5 or more standard deviations larger than the mean
#create columns that mark outliers for sti rate and teen conception rate
mergedDF['out_sti'] = None
sti_mean = mergedDF.RateOfAcuteSTIsPer1000Population.mean()
sti_stdev = mergedDF.RateOfAcuteSTIsPer1000Population.std()
mergedDF['out_sti'] = np.where(mergedDF[
                                        'RateOfAcuteSTIsPer1000Population'
                                        ] - sti_mean >= 1.5 * sti_stdev, 1, 0)
mergedDF['out_preg'] = None
preg_mean = mergedDF.Under18ConceptionRatePer1000FemalesAged15To17.mean()
preg_stdev = mergedDF.Under18ConceptionRatePer1000FemalesAged15To17.std()
mergedDF['out_preg'] = np.where(mergedDF[
                                        'Under18ConceptionRatePer1000FemalesAged15To17'
                                        ] - preg_mean >= 1.5 * preg_stdev, 1, 0)
print mergedDF.head(5)

#find boroughs with excessive STI rate
#and without correspondingly high underage conception rate
stiHigh_pregLow = mergedDF['Borough'][(mergedDF.out_sti == 1) & (mergedDF.out_preg != 1)]
print "The Boroughs with high STI rates, but not correspondingly high under-18 conception rates, are: "
print(stiHigh_pregLow)

#compute Mahalanobis distance to properly identify outliers
#create numpy matrix with required data
columnValues = mergedDF.as_matrix(["RateOfAcuteSTIsPer1000Population", "Under18ConceptionRatePer1000FemalesAged15To17"])
#generate a "mean vector" with means as calculated above
meanVector = np.asarray([sti_mean, preg_mean ]).reshape(1,2)
mahalanobisDistances = spatial.distance.cdist(columnValues, meanVector, 'mahalanobis')
def MD_Outliers(x, y, MD):
    threshold = np.mean(MD) * 1
    outliers = []
    for i in range(len(MD)):
        if MD[i] >= threshold:
            outliers.append(i)
    return outliers
outMD = MD_Outliers(
                    mergedDF.RateOfAcuteSTIsPer1000Population,
                    mergedDF.Under18ConceptionRatePer1000FemalesAged15To17,
                    mahalanobisDistances
)
print "Outliers according to Mahalanobis distance: ", mergedDF.loc[outMD]

#Scatterplots using bokeh module
output_file("scatter.html")
#create function for scatterplots with bokeh
def plt(Title, dependentTitle, dependent):
    Source = ColumnDataSource(data=dict(
                i=mergedDF.Borough,
                x=mergedDF.RateOfAcuteSTIsPer1000Population,
                y=dependent,
                z=mergedDF.TotalMedianAnnualHouseholdIncomeEstimateFor2013,
            )
    )
    hover = HoverTool(
            tooltips=[
                ("Borough", "@i"),
                ("Rate Of Acute STIs Per 1000 Population", "$x"),
                (dependentTitle, "$y"),
                ("Total Median Annual Household Income Estimate For 2013", "@z")
            ]
        )
    colorColumn = []
    for i in range(len(mergedDF)):
        #if mergedDF["out_sti"][i] == 1 and mergedDF["out_preg"][i] != 1:
        if i in outMD:
            colorColumn.append("grey")
        else:
            colorColumn.append("teal")

    TOOLS = [
            BoxSelectTool(),
            WheelZoomTool(),
            ResetTool(),
            BoxSelectTool(),
            PreviewSaveTool(),
            hover
    ]
    p = figure(title = Title, tools=TOOLS)
    p.xaxis.axis_label = 'Rate Of Acute STIs Per 1000 Population'
    p.xaxis.axis_label_text_font_size = "12pt"
    p.yaxis.axis_label = dependentTitle
    p.yaxis.axis_label_text_font_size = "12pt"
    p.inverted_triangle(mergedDF.RateOfAcuteSTIsPer1000Population,
                dependent,
                color=colorColumn,
                size=15,
                alpha=0.5,
                source=Source
    )
    return p

#plot dependent vs. independent variables
s1 = plt(
        "Rate of acute STIs vs. Under 18 Conception rate",
        'Under 18 Conception Rate Per 1000 Females Aged 15 To 17',
        mergedDF.Under18ConceptionRatePer1000FemalesAged15To17
)
s2 = plt(
        "Rate of acute STIs vs. Violence rate",
        'Rate Of Recorded Violence Against The Person Offences',
        mergedDF.RateOfRecordedViolenceAgainstThePersonOffences
)
s3 = plt(
        "Rate of acute STIs vs. Narcotics User rate",
        'Estimated Crude Rate Of Opiate And Crack Cocaine Users Per 1000 Aged 15 To 64',
        mergedDF.EstimatedCrudeRateOfOpiateAndCrackCocaineUsersPer1000Aged15To64
)
s4 = plt(
        "Rate of acute STIs vs. Median Houshold Income",
        'Total Median Annual Household Income Estimate For 2013',
        mergedDF.TotalMedianAnnualHouseholdIncomeEstimateFor2013
)
#align all 4 plots into one single figure
vp = gridplot([[s1, s2], [s3, s4]])
show(vp)

#perform a Pearson correlation for dependent vs each independent variable
print "Pearson correlation, p-value between Rate of acute STIs and: "
print "1. Total Median Annual Household Income Estimate For 2013: "
print stats.pearsonr(mergedDF.RateOfAcuteSTIsPer1000Population,
                        mergedDF.TotalMedianAnnualHouseholdIncomeEstimateFor2013)
print "2. Estimated Crude Rate Of Opiate And Crack Cocaine Users Per 1000 Aged 15 To 64: "
print stats.pearsonr(mergedDF.RateOfAcuteSTIsPer1000Population,
                        mergedDF.EstimatedCrudeRateOfOpiateAndCrackCocaineUsersPer1000Aged15To64)
print '3. Rate Of Recorded Violence Against The Person Offences: '
print stats.pearsonr(mergedDF.RateOfAcuteSTIsPer1000Population,
                        mergedDF.RateOfRecordedViolenceAgainstThePersonOffences)
print "4. Under 18 Conception Rate Per 1000 Females Aged 15 To 17: "
print stats.pearsonr(mergedDF.RateOfAcuteSTIsPer1000Population,
                        mergedDF.Under18ConceptionRatePer1000FemalesAged15To17)

#build linear regression model with
#dependent var: 'Rate of acute STIs per 1,000 population'
#independent var: the others
model = sm.formula.ols(formula= ('RateOfAcuteSTIsPer1000Population ~ '
                                 'Under18ConceptionRatePer1000FemalesAged15To17 + '
                                 'RateOfRecordedViolenceAgainstThePersonOffences + '
                                 'EstimatedCrudeRateOfOpiateAndCrackCocaineUsersPer1000Aged15To64 + '
                                 'TotalMedianAnnualHouseholdIncomeEstimateFor2013'),
                                 data= mergedDF
)

results = model.fit()
print results.summary()

#OR
#use alternative linear regression model
x = mergedDF.RateOfAcuteSTIsPer1000Population
y = mergedDF[
            ['Under18ConceptionRatePer1000FemalesAged15To17',
            'RateOfRecordedViolenceAgainstThePersonOffences',
            'EstimatedCrudeRateOfOpiateAndCrackCocaineUsersPer1000Aged15To64',
            'TotalMedianAnnualHouseholdIncomeEstimateFor2013']
]
clf = linear_model.LinearRegression(normalize=True)
clfResults = clf.fit(y, x)
clfResults.intercept_
clfResults.coef_

#build ridge regression model to check for effect size of multicolinearity
#find the best alpha to use
a = linear_model.RidgeCV(alphas=np.arange(0.1,10,.1), normalize=True)
a_results = a.fit(y, x)
a_results.alpha_
#build ridge regression model
ridge_model = linear_model.Ridge(alpha=a_results.alpha_, normalize=True)
ridge_results = ridge_model.fit(y, x)
ridge_results.intercept_
ridge_results.coef_
