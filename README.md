# London-Health-Data
# Analysis Strategy
The “gla-household-income-estimates-WardLevel” excel file from London Datastore along with the following csv files from the Health Profiles 2013 repository were selected for this study: “AcutesexuallytransmittedinfectionsCSV”, “DrugmisuseCSV”, “Teenagepregnancyunder18CSV” & “ViolentcrimeCSV”. Although the Health Profiles data files are marked for year 2013, within the files themselves other years are mentioned as well. In order to avoid confusion all data was assumed to correspond to year 2013 and other date points were ignored.
The dependent variable upon which the following analysis was based is the “Rate Of Acute STIs Per 1000 Population” (”Indicator value”). This variable was selected because monitoring and being able to predict the rate of STIs in a region is crucial for public health, while knowledge on the patterns of STI rates may lead to better understanding of the underlying causes of such events per location. The measures selected as independent variables were the following, as they make intuitive sense and are logically linked with acute STIs: “Estimated Crude Rate Of Opiate And Crack Cocaine Users Per 1000 Aged 15 To 64”, “Under 18 Conception Rate Per 1000 Females Aged 15 To 17”, “Rate Of Recorded Violence Against The Person Offences” and “Total Median Annual Household Income Estimate For 2013”. The latter was preferred to its relevant mean, as the median value is more robust against the unknown underlying samples’ distributions.
During the analysis, the following steps were performed: Initially data was imported and merged. Some boroughs were marked as outliers for further inspection via exploratory data analysis using scatterplots [1]. The correlation between the dependent and independent variables was calculated. Finally, attempts were made to create a linear regression model, in order to derive STI rates as a function of the four independent variables, which will be evaluated accordingly.
Despite the lack of detailed data, expectations are that few boroughs exhibiting high STI rates will show relatively low rates of Under-18 Conception. These boroughs are of interest, as other reasons, such as sexual abuse or the use of narcotics might be causing the elevated STI rates, apart from unprotected consensual intercourse (which is assumed to be the primary reason for Under-18 Conception and STIs).
# Data Preparation
Initially the data identified previously were imported from each file, omitting unwanted columns, so that only borough names and codes, rates of the aforementioned metrics and the median annual household income remained. A new data frame was created with the data merged based on the “ONS Code (new)”/”LAD Code” inner join. This resulted in the restriction of the dataset to regions of Greater London (Boroughs), which is the focus of this analysis. Columns were reassessed and few unwanted ones were discarded. Meaningful names were set for each column and all rates were transformed to ‘per 1000 population’, so that they are interpreted in the same way and are, thus, more comparable. No further normalisation was considered necessary, as most of the data was already comparable and easily interpretable. Thus, additional such manipulations may have reduced the quality of information captured by the dataset.
One of the issues that arose during the data preparation was that the names of some of the initial columns contained hidden whitespace characters, which interfered with the data wrangling process. A function was designed in order to overcome this issue and adjust the format of the column titles appropriately, by removing these characters.
Then, boroughs with excessive STI rates, but not proportionally high Under-18 Conception rates were sought. For this purpose two new columns were created, with Boolean values that indicate whether an observation is more than 1.5 standard deviations larger than the mean of the respective variable (a metric selected due to the restricted dataset size).
# Data Analysis, Results & Discussion
Only a single borough, that of Hackney, was characterized by excessive STI, and low Under-18 Conception rates, as categorized previously. Exploratory data analysis was performed via scatterplot visualisations to further examine outliers and trends (view the interactive scatterplots at: file:///Users/elektra/scatter.html). Outliers were marked according to their Mahalanobis distance, as computed previously, for the variables of STI rate vs. Under-18 Conception rate. It is clear that this distance measure was better at capturing outliers than the standard deviation threshold used earlier, due to the elliptical (and not spherical) structure of the observations in the scatterplot [2]. These outliers were then marked in a different colour on all scatterplots, for them to be inspected more closely. Hackney appeared as an outlier once more.
One of the outliers on the lower end is Richmond, which also exhibited low rates of criminality and narcotics use. This is understandable, given that Richmond has one of the highest median annual household incomes and is considered a family suburb. On the other hand, Hackney, Southwark and Lambeth show high rates of both STIs and Under-18 conceptions. However, as seen above, Hackney has lower under-18 conception rates than the other two, which could imply that high STI rates are caused by something other than “casual” unprotected intercourse. Yet, all three exhibit a similar behaviour when it comes to median annual income (medium-low), narcotics user rates ( high) and rate of violent crimes (medium-high). In Hackney, more specifically, elevated narcotics user rates are higher than the other two boroughs, whereas crime rate is lower, a fact which is counter-intuitive and suggests recreational sex and drug use. Therefore additional social habits may be responsible for
these results, which the authorities in Hackney could investigate further. Additionally, Camden and Kensington & Chelsey are interesting since they exhibit high rates of STIs compared to their low rates of under-18 conceptions. For Kensington & Chelsea in particular, this information, along with the facts that median annual household income is very high and criminality rates are very low, can lead to the conclusion that much recreational drug usage takes place there. This is also a finding which local authorities could act upon.
Naturally, many of the above comments may prove wrong, as correlation does not imply causal relationships among variables. Moreover, many factors were ignored during this analysis, such as the fact that boroughs with universities may appear as outliers due to the behavior of the many students residing there. Finally, Under-18 Conception rate was used as an indicator for the rate of unprotected
intercourse in the general population (regardless of age) to enable hypotheses and discussion. In reality this is a very wrong assumption and probably underestimates the true value of this rate. Before attempting to create a linear regression model, Pearson’s correlation was calculated for the pairs of dependent and independent variables. STI rates showed statistically signifficant high correlation with narcotics’ user rates (>0.7).
After performing ordinary least squares regression on STI rates as a function of the remaining variables, the following results were obtained for the normalised data (normalisation was performed mainly because of the Median Annual Household Income data, whose scale differed from the rest): an adjusted R2 of 0.626 representing goodness-of-fit, an F- statistic of 14 with a p-value of 2.64*10-6 denoting significance under the threshold of alpha=0.05. For the Under-18 Conception rate and Narcotics’ User rate variables, p-values << 0.05, meaning that their coefficients were statistically significant. Both showed a positive relationship with the rate of STIs, as expected, with coefficients of ~0.32 and ~0.96, respectively. The Median Annual Household Income and Violence rate variables on the other hand showed negligible impact on the STI rate (coefficients=~0), results for which no statistical significance was established. The output of the model suggested that multicolinearity might indeed affect the results. Therefore, a ridge regression was performed to assess the magnitude of the problem caused by such relationships among the independent variables [3]. For this, bias was introduced as the ‘alpha’ parameter and was optimised accordingly. The resulting alpha that was chosen for the model was equal to 0.2, not much higher from that of the original model (alpha=0), while the coefficients did not vary much either, compared to the first model. In conclusion the model was considered to perform well enough for the purposes of this study and was not rejected as completely unreliable because of multicolinearity.
# References
[1] R. M. Church, “How To Look At Data: a Review of John W. Tukey’S Exploratory Data Analysis1,” J. Exp. Anal. Behav., vol. 31, no. 3, pp. 433–440, 1979.
[2] H. Bhavsar and A. Ganatra, “Support Vector Machine Classification using Mahalanobis Distance Function,” Int. J. Sci. Eng. Res., vol. 6, no. 1, pp. 618–626, 2015.
[3] N. I. Rashwan and M. El-Dereny, “Solving Multicollinearity Problem Using Ridge Regression Models,” Int. J. Contemp. Math. Sci., vol. 6, no. 12, pp. 585 – 600, 2011.
