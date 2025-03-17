# BME160 Final Projeect

# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import seaborn as sns
from scipy import stats

### Pre-processing
def readCSVs(momData): 
    #initializing a dict to eventually return
    outDataDict = {}

    for momFile in momData: #iterate through list of csv files (names = IDs)
        # retrieving animalID 
        animalID = momFile[:-4] #to use in outDataDict as key for indiv

        # creating data frame of resight history (uncleaned)
        df = pd.read_csv(momFile)
        #select only columns needed --> dfColSel is df w necessary data
        dfColSel = df[['Age','AnimalID','Date','Pups']]

        #formatting the date 
        dfColSel['Date'] = pd.to_datetime(dfColSel['Date'], format='%m/%d/%y')
        dfColSel['Date'] = dfColSel['Date'].dt.strftime('%m-%d')

        ## Cleaning data ##
        # remove special chars
        dfColSel['Pups'] = dfColSel['Pups'].astype(str).str.replace('?','')
        #remove nonnumeric values; assign to na
        dfColSel['Pups'] = pd.to_numeric(dfColSel['Pups'], errors='coerce')
        #drop all na values
        dfColSel = dfColSel.dropna()
        #appending dict w key/value pair: animalID / completed panda df
        outDataDict[animalID] = dfColSel
    
    #return cumulative dict w all mom data
    return outDataDict

### Processing
def breedingAnalysis(CleanDataDict):
    #create a dicitonary for output
    outDict = {}
    out2Dict = {}

    #iterate through mom data
    for mom in CleanDataDict.keys():
        #initializing for each mom 
        momID = mom
        df = CleanDataDict[mom]
        #print('mom:',momID,'df:\n', df)

        # making df with only pup present 
        dfPup = df[df['Pups'] > 0]

        # iterating through df, making dict for every year present
        # getting important age values (trying df.items() )
        ageSet = set()
        ageList = [] #to track number of sig sightings
        for i in dfPup.items(): 
            if i[0] == 'Age':
                for num in i[1]: # the series of ages
                    ageList.append(num) # make list of all years to sort sig years
                    ageSet.add(num) # make set of ages she had a pup w/out dupes 

        # appending ageSet to remove any insignificant years (<5 sightings)
        ageSetFull = set()
        for age in ageSet:
            sightingCount = ageList.count(age) #counts number of significant sightings in year
            if sightingCount > 5:
                ageSetFull.add(age)

        # making dict for significant age --> dict of dicts: ageDict = { {Age: {date: pupStatus, date: pupStatus} } }
        ageDict = {}
        for age in ageSetFull:
            ageDict[int(age)] = {}
        #print(ageDict)
        finalDict = ageDict # just the keys, to finalize LATER

        # populating ageDict dictionary w date / pupStatus keyValue pairs
        for row in dfPup.iterrows():
            data = row[1]
            # within series, get age data
            # match to dict key associated w that age
            for key in ageDict.keys():
                #print(type(key[1:]), type(data['Age']))
                if data['Age'] == key: #want to add data to appropriate year
                    # adding data to dict
                    ageDict[key].update({data['Date']: data['Pups']}) #update dictionary w date and pup keyValue pairs

        #populating finalized dictionary: format is {age: [pupDate, lastSeenDate]}
        for key in finalDict.keys():
            #get min, max date in dict corresponding to age key in ageDict
            #only count years when mom seen w pup for @least 5 days
            pupDate = min(ageDict[key])
            lastSeen = max(ageDict[key])

            #append final dictionary w list of pupDate, lastSeen date
            finalDict[key] = [pupDate, lastSeen]

        #print('here\n',finalDict)
        # transferring into np array to work w plotting: final dict is in workable format: 'a'+age: [pupdate, lastSeen]
        l = []
        for key in finalDict.keys():
            finalDict[key].insert(0, int(key))
            l.append(finalDict[key])
            #np.append(fullData,['here'])

            #append avgDict with age data
            age = int(key)
            #avgDict[age] = 

        l=np.array(l)
        outDict[momID] = l
        out2Dict[momID] = finalDict

    return outDict, out2Dict #return dict of momID and corresponding np array ready to plot

### Colony data investigation with target individual
def avgWork(compDict, targetID=''): 
    workableDict = {}
    targetData = {}
    # extract just pupping date
    for momID, momData in compDict.items():
        #retrieve target mom data
        if momID == targetID:
            targetData[momID] = momData
        
        workableDict[momID] = {}
        currentMom = workableDict[momID]
        for age, ageData in momData.items():
            ageData_n = '2025-'+ageData[1] #remove lsd; make into single value
            currentMom[age] = ageData_n
    
    #making pandas dataframe => age as columns, momID as indexes 
    df = pd.DataFrame(workableDict)
    correctDf = df.T
    correctDf = correctDf.apply(pd.to_datetime,errors='coerce')

    #get avg for each column
    avgDate_annual = {}
    for column in correctDf.columns:
        age_mean = correctDf[column].mean()
        avgDate_annual[column] = pd.to_datetime(age_mean)
    avgDate_df = pd.DataFrame(avgDate_annual,index=['Average_PupDate'])

    return avgDate_df,targetData # => cols=age, row=Average_PupDate ==> momID data
def indivPred(compDict, targetID=''):  
    #get mom data specifically
    targetData = {}
    for momID, momData in compDict.items():
        if momID == targetID:
            targetData = momData
            for age, ageData in targetData.items():
                ageData_n = '2025-'+ageData[1] #remove lsd; make into single value
                targetData[age] = [age, ageData_n]
    targetDf= pd.DataFrame(targetData, index=['Age','Pup Date'])
    targetDf = targetDf.T

    #best fit line work - Pupping date
    #convert datetime to float64
    targetDf['Pup Date'] = mdates.date2num(targetDf['Pup Date'])
    targetDf['Age'] = pd.to_numeric(targetDf['Age'])
    coef = np.polyfit(targetDf['Age'], targetDf['Pup Date'],1) #coef = [slope, int]
    fx = np.poly1d(coef)
    y_fit = fx(targetDf['Age'])

    # equation: y = coef[0]*x + coef[1]
    slope = coef[0]
    targetAge = max(targetDf['Age']) + 1 #gets last recorded age
    next_pup_est = slope*(targetAge) + coef[1]
    return [targetAge, next_pup_est] #returns list of [finalAge, pupDateEstimate]

### Visualization
def avgPlots(targetData, avgDf, targetEst, targetID=''): 
    #template
    fig, ax = plt.subplots()

    #combining data -> 1 df
    compDf = pd.DataFrame(targetData[targetID],index=['Age',f'ID{targetID}_PupDate']).T
    avgDf1 = avgDf.T
    compDf['Population_Avg_PupDate'] = avgDf1['Average_PupDate']
    compDf[f'ID{targetID}_PupDate'] = pd.to_datetime(compDf[f'ID{targetID}_PupDate'])
    compDf['Population_Avg_PupDate'] = pd.to_datetime(compDf['Population_Avg_PupDate'])

    #replacing na vals w mom data where null
    compDf['Population_Avg_PupDate'] = np.where(compDf['Population_Avg_PupDate'].isnull(), compDf[f'ID{targetID}_PupDate'], compDf['Population_Avg_PupDate'])

    #calc best fit line (momData)
    compDf[f'ID{targetID}_PupDate'] = mdates.date2num(compDf[f'ID{targetID}_PupDate'])
    compDf['Age'] = pd.to_numeric(compDf['Age'])
    coef = np.polyfit(compDf['Age'], compDf[f'ID{targetID}_PupDate'],1)
    fx = np.poly1d(coef)
    y_fit = fx(compDf['Age'])
    compDf[f'ID{targetID}_trendline'] = y_fit #add to compDf

    #plotting
    compDf.plot(x='Age',y=f'ID{targetID}_PupDate',ax=ax,color='orange',marker='o',markersize=3)
    compDf.plot(x='Age',y='Population_Avg_PupDate',ax=ax,color='black',marker='o',markersize=5,alpha=0.1)
    compDf.plot(x='Age',y=f'ID{targetID}_trendline',ax=ax,color='orange',marker='o',markersize=0,alpha=0.4)

    #PREDICTIONS
    targetAge = targetEst[0]
    #estimate based on bestfitline
    plt.scatter(targetEst[0],targetEst[1],color='orange',marker='^',s=100,label=f'ID{targetID}_PupDateEstimate')
    #estimate based on cumulative pop avg
    if targetAge in compDf.index.tolist():
        t = compDf.at[targetAge, 'Population_Avg_PupDate']
    else: #seal oldest in dataset!
        t = compDf.at[targetAge-1, 'Population_Avg_PupDate'] #just get last pupping date
    plt.scatter(targetAge,t, color='black',marker='^',s=100,label='Population_PupDateEstimate')

    #annotating prediction dates
    estDate_lbf = str(mdates.num2date(targetEst[1]))
    estDate_avg = str(t)
    plt.annotate(f'{estDate_lbf[5:10]}', (targetEst[0],targetEst[1]))
    plt.annotate(f'{estDate_avg[5:10]}', (targetAge, t))

    #design specs
    #plotting vert line showing last recorded year
    plt.axvline(x=targetAge-1, color='black', linestyle='-', linewidth=1, alpha=0.1)
    #plotting vert line of pup date estimate range
    plt.vlines(x=targetAge, ymin=targetEst[1], ymax=t, color='red', linestyle='dashed', linewidth=1, alpha=0.5)

    #graph specifications
    date_format = mdates.DateFormatter('%m-%d') #format y-axis ticks
    ax.yaxis.set_major_formatter(date_format)
    ax.set_ylabel('Breeding Season Date (MM-DD)',size=12) #create labels
    ax.set_xlabel(f'ID{targetID} Age (year)',size=12)
    ax.set_title(f'Pupping Date Prediction Interval: ID{targetID}',size=14)
    #adding new year to x-ticks
    xticks = plt.xticks()[0].tolist()
    coagtick = sorted(xticks + [targetAge])
    plt.xticks(coagtick)

    plt.legend()

#create a plot that compares individual female to other seals averages across lifetime
def female_seal_compare_plot(seal_file, other_data, output_file=None):

    #load the mother seal data on csv file
    #convert date and pup comlums to numbered values
    #gather the animal ID from file
    mother_data = pd.read_csv(seal_file)
    mother_data['Pups'] = pd.to_numeric(mother_data['Pups'], errors='coerce')
    mother_data['Date'] = pd.to_datetime(mother_data['Date'], format='%m/%d/%y', errors='coerce')
    mother_id = mother_data['AnimalID'].iloc[0]

    #initilize a list to store data of seals
    #similar to mother data, convert the pups and dates of the csv files
    #add the data to empty list
    #print error if the files cannot be read or processed
    comparing_seals_data = []
    for file in other_data:
        try:
            comp_data = pd.read_csv(file)
            comp_data['Date'] = pd.to_datetime(comp_data['Date'], format='%m/%d/%y', errors='coerce')
            comp_data['Pups'] = pd.to_numeric(comp_data['Pups'], errors='coerce')
            comparing_seals_data.append(comp_data)
        except Exception as e:
            print(f"Error {file}: {e}")

    #gather all of the other seals data from other files and combine for the average
    #empty data frame if no data
    if len(comparing_seals_data) > 0:
        comp_seals = pd.concat(comparing_seals_data)
        seal_comparison = True
    else:
        comp_seals = pd.DataFrame()
        seal_comparison = False

    #analysis of mother seal age to determine pupping rate at each age and # of observations of pups with mother
    #calculate standard error for each age of mother
    seals_age = mother_data.groupby('Age')['Pups'].agg([('PuppingRate', lambda x: (x>0).mean()), ('Count','count')]).reset_index()
    seals_age['StdErr']=mother_data.groupby('Age')['Pups'].apply(lambda x: stats.sem(x, nan_policy='omit')).values

    #to compare with other seals averages
    #analyze other seals age to determine pupping rate and # of observations
    #calculate standard error
    if seal_comparison:
        age_compare = comp_seals.groupby('Age')['Pups'].agg(PuppingRate=lambda x: (x>0).mean()).reset_index()

        age_compare['StdErr'] = comp_seals.groupby('Age')['Pups'].apply(lambda x: stats.sem(x, nan_policy='omit')).values

    #create the plot
    #create scatter plot of of mother seal and adjust point size based on observations
    #create errorbars for mother seals points
    plt. figure(figsize=(10, 6))
    plt.errorbar(seals_age['Age'], seals_age['PuppingRate'], yerr=seals_age['StdErr'], fmt='none', ecolor='skyblue', capsize=4, alpha=0.8, zorder=2)
    plt.scatter(seals_age['Age'], seals_age['PuppingRate'], color='steelblue', s=seals_age['Count']*15, alpha=0.9, zorder=3, label=f'Mother Seal #{mother_id}')

    #add the other seals averages to the plot
    #create a shaded region to represent the standerd error of the average and a line
    if seal_comparison:
        plt.fill_between(age_compare['Age'], age_compare['PuppingRate'] - age_compare['StdErr'], age_compare['PuppingRate'] + age_compare['StdErr'], color='orange', alpha=0.2, zorder=0)
        plt.plot(age_compare['Age'], age_compare['PuppingRate'], 'o-', color='darkorange', alpha=0.4, markersize=4, linewidth=1, zorder=1, label='Other Seals Average')

    #create a small box on plot that shows the age range and the pupping rate for individual being analyzed
    min_age = seals_age['Age'].min()
    max_age = seals_age['Age'].max()
    pupping_rate = mother_data['Pups'].apply(lambda x: x > 0).mean()
    stats_analysis = (f"Observations: {seals_age['Count'].sum()}\n"
                      f"Ages: {min_age}-{max_age} years\n"
                      f"Pupping Rate: {pupping_rate:.2f}")
    plt.text(4, 0.95, stats_analysis, fontsize=8, bbox={'facecolor':'white', 'alpha':0.8, 'pad':5})

    #label the plot title, x axis, y axis
    #add a grid to plot to better see point locations
    #add an x limit and y limit
    #create a legend
    plt.title(f'Seal #{mother_id} Vs. Average Seal Pupping Data', fontsize=15)
    plt.ylabel('Pupping Rate', fontsize=11)
    plt.xlabel('Age', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    plt.xlim(3,19)
    plt.legend()
    plt.show()


def main():
    # data is hardcoded into a list of .csv files
    targetMom = '20643'
    momDataList1 = [
        '40047.csv','40502.csv','40338.csv','38241.csv',
        '40062.csv','37988.csv','38369.csv','40932.csv','40307.csv',
        '42653.csv','38463.csv','40287.csv','34244.csv','40618.csv',
        '36358.csv','40336.csv', '20643.csv']
    momDataList = ['40336.csv', '20643.csv']

    #making list of all files except for target
    otherMoms = []
    target = targetMom+'.csv'
    for mom in momDataList:
        if mom != target:
            otherMoms.append(mom)

    #target - population visualization 
    female_seal_compare_plot(targetMom+'.csv', otherMoms)

    #prediction visualization
    inputfile = readCSVs(momDataList)
    readytoPlot, ageDict = breedingAnalysis(inputfile) #return two dicts; one of momReadytoPlot, one for avgData visualization
    indivPredOut = indivPred(ageDict,targetMom) #returns [targetAge, predictedDate]
    avgDf, targetDict = avgWork(ageDict,targetMom) #returns avgData_df, targetData
    avgPlots(targetDict, avgDf, indivPredOut, targetMom)

main()

