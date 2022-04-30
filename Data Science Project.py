# -*- coding: utf-8 -*-
"""
Name:       Yosua Nainggolan
Email:      Yosua.Nainggolan09@myhunter.cuny.edu
Resources:  https://data.cityofnewyork.us/Health/New-York-City-Leading-Causes-of-Death/jb7j-dtam
            https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95
            https://data.cityofnewyork.us/Housing-Development/Housing-Database-by-NTA/kyz5-72x5
Title:      Safety of NYC Based on Mortality Rates of Different Causes
URL:        https://ynainggolan.github.io/DataSci-Project/


@author: yhpna

Data Science Project
"""

import pandas as pd
import seaborn as sns
import numpy as np
import folium
import matplotlib.pyplot as plt
import re

def import_data(file_name) -> pd.DataFrame:
    '''
    Used to import any data into a data frame
    '''
    df = pd.DataFrame(pd.read_csv(file_name))
    return df

def import_vehicle_data(file_name) -> pd.DataFrame:
    '''
    Used to Import the Vehicle Data
    '''

    df = pd.DataFrame(pd.read_csv(file_name, low_memory = False))
    df = df[ ['CRASH DATE', 'BOROUGH', 'ZIP CODE', 'LOCATION',\
              'NUMBER OF PERSONS INJURED', 'NUMBER OF PERSONS KILLED', 'COLLISION_ID' ] ]
    return df
    
def find_suicide(leading_cause, death):
    '''
    helper functions to seperate the number of deaths that is caused by themselves
    '''
    raw = r'Suicide{1}'
    raw_2 = r'Disorders{1}'
    
    if re.search(raw, leading_cause):
        return death
    elif re.search(raw_2, leading_cause):
        return death
    else:
        return 0
    
def find_other(leading_cause, death):
    '''
    helper functions to seperate the cause of death being other people
    '''
    raw = r'Homicide{1}'
    raw_2 = r'Accidents{1}'
    
    if re.search(raw, leading_cause):
        return death
    elif re.search(raw_2, leading_cause):
        return death
    else:
        return 0

def find_disease(leading_cause, death):
    '''
    helper functions to seperate the cause of death being disease whether it be cureable or not
    '''
    raw = r'Homicide{1}'
    raw_2 = r'Accidents{1}'
    raw_3 = r'Suicide{1}'
    raw_4 = r'Disorders{1}'
    
    if (re.search(raw, leading_cause) == None) and (re.search(raw_2, leading_cause) == None)\
        and (re.search(raw_3, leading_cause) == None) and (re.search(raw_4, leading_cause) == None):
        return death
    elif re.search(raw_2, leading_cause):
        return death
    else:
        return 0
    
def clean_death(death):
    '''
    cleaned the death column on the leading cause of Death in NYC
    '''
    if death != ".":
        return int(death)
    elif death == '.':
        return 0
    
def clean_vehicle_data(df):
    '''
    Cleaned the vehicle data so that it is more manageble to work with
    '''
    df['CRASH DATE'] = pd.to_datetime(df['CRASH DATE'])
    df['YEAR'] = df['CRASH DATE'].dt.year
    df['COLLISION_ID'] = 1
    df = df.dropna()

    df = df.drop(['CRASH DATE'], axis = 1)
    return df

def compute_lin_reg(x,y):
    """
    This function takes two Series, and returns theta_0,theta_1 for their 
    regression line, where theta_1 is the slope (r*(std of x)/(std of y)) 
    and theta_0 is the y-intercept ( (ave of y) - theta_1*(ave of x)).
    """

    theta_1 = x.corr(y)*y.std()/x.std()
    theta_0 = y.mean() - theta_1*x.mean()
    return theta_0,theta_1

def map_vehicle_crashes(df):
    '''
    This functions will map the most amount of death caused by car crashes
    '''
    nta_geo = "NTA map.geojson"
    outFile = "Car Crashes, 2012-2022.html"
    df['NUMBER OF PERSONS KILLED'] = df['NUMBER OF PERSONS KILLED'].astype(int)
    #df = df.groupby("BOROUGH")["NUMBER OF PERSONS KILLED"].mean()

    #Center map at Hunter:  40.7678° N, 73.9645° W
    m = folium.Map(location=[40.7678,-73.9645],zoom_start=10.5,tiles="cartodbpositron")

    #Add in a choice of map tiles & icon to switch layers:
    tiles = ['stamenwatercolor', 'cartodbpositron', 'openstreetmap', 'stamenterrain']
    for tile in tiles:
        folium.TileLayer(tile).add_to(m)
    legendTitle = f"Car Crashes Death, 2012 - 2022"
    
    choropleth = folium.Choropleth(
       geo_data=nta_geo,
       name="choropleth",
       data=df,
       columns=["nta2010","NUMBER OF PERSONS KILLED"],
       key_on="feature.properties.ntacode",
       fill_color="Reds",
       fill_opacity=0.75,
       line_opacity=0.75,
       legend_name=legendTitle,
       highlight = True
    ).add_to(m)
    
    folium.LayerControl().add_to(m)
    m.save(outFile)

def main():
    leading_cause = import_data('New_York_City_Leading_Causes_of_Death.csv')
    leading_cause['Deaths'] = leading_cause.apply(lambda row: clean_death(row['Deaths']),axis=1)
    leading_cause['Self'] = leading_cause.apply(lambda row: find_suicide(row['Leading Cause'], row['Deaths']),axis=1)
    leading_cause['Others'] = leading_cause.apply(lambda row: find_other(row['Leading Cause'], row['Deaths']),axis=1)
    leading_cause['Disease'] = leading_cause.apply(lambda row: find_disease(row['Leading Cause'], row['Deaths']),axis=1)
    leading_cause = leading_cause.drop(index = 0 )
    leading_cause.to_csv('NYC Leading Cause.csv', index = False)
    '''
    sns.countplot(x='Leading Cause', data=leading_cause)
    locs, labels = plt.xticks()
    plt.setp(labels, rotation = 90)
    plt.show()
    '''
    death_ = leading_cause.groupby('Year')['Deaths'].sum()
    death_ = death_.reset_index()
    self_ = leading_cause.groupby('Year')['Self'].sum()
    self_ = self_.reset_index()
    other_ = leading_cause.groupby('Year')['Others'].sum()
    other_ = other_.reset_index()
    disease_ = leading_cause.groupby('Year')['Disease'].sum()
    disease_ = disease_.reset_index()
    
    # sod is the data frame that consist of the summary of self, others, and diseases number of death
    sod = self_.merge(other_, on = 'Year')
    sod = sod.merge(disease_, on = 'Year')
    sod = sod.merge(death_, on = 'Year')
    
    '''
    #This section uses the data from the leading_cause that hasn't been group by year.
    b, m = compute_lin_reg(leading_cause['Disease'], leading_cause['Deaths'])
    xes = np.array([0,leading_cause['Disease'].max()])
    yes = m*xes + b
    plt.scatter(leading_cause['Disease'],leading_cause['Deaths'])
    plt.plot(xes,yes,color='r')
    plt.title(f'Regression line of Disease with m = {m:{4}.{2}} and y-intercept = {b:{4}.{4}}')
    plt.show()
    
    b_2, m_2 = compute_lin_reg(leading_cause['Self'], leading_cause['Deaths'])
    xes = np.array([0,leading_cause['Self'].max()])
    yes = m_2*xes + b_2
    plt.scatter(leading_cause['Self'],leading_cause['Deaths'])
    plt.plot(xes,yes,color='r')
    plt.title(f'Regression line of Self with m = {m_2:{4}.{2}} and y-intercept = {b_2:{4}.{4}}')
    plt.show()
    
    b_3, m_3 = compute_lin_reg(leading_cause['Others'], leading_cause['Deaths'])
    xes = np.array([0,leading_cause['Others'].max()])
    yes = m_3*xes + b_3
    plt.scatter(leading_cause['Others'],leading_cause['Deaths'])
    plt.plot(xes,yes,color='r')
    plt.title(f'Regression line of Others with m = {m_3:{4}.{2}} and y-intercept = {b_3:{4}.{4}}')
    plt.show()
    '''
    #this section uses data from sod (self_other_diseases) that has group the three columns by year.
    b, m = compute_lin_reg(sod['Disease'], sod['Deaths'])
    xes = np.array([0,sod['Disease'].max()])
    yes = m*xes + b
    plt.scatter(sod['Disease'],sod['Deaths'])
    plt.plot(xes,yes,color='r')
    plt.title(f'Regression line of Disease with m = {m:{4}.{2}} and y-intercept = {b:{4}.{4}}')
    plt.show()
    
    b_2, m_2 = compute_lin_reg(sod['Self'], sod['Deaths'])
    xes = np.array([0,sod['Self'].max()])
    yes = m_2*xes + b_2
    plt.scatter(sod['Self'],sod['Deaths'])
    plt.plot(xes,yes,color='r')
    plt.title(f'Regression line of Self with m = {m_2:{4}.{2}} and y-intercept = {b_2:{4}.{4}}')
    plt.show()
    
    b_3, m_3 = compute_lin_reg(sod['Others'], sod['Deaths'])
    xes = np.array([0,sod['Others'].max()])
    yes = m_3*xes + b_3
    plt.scatter(sod['Others'],sod['Deaths'])
    plt.plot(xes,yes,color='r')
    plt.title(f'Regression line of Others with m = {m_3:{4}.{2}} and y-intercept = {b_3:{4}.{4}}')
    plt.show()
   

    # This sections will print out the relative death count graphs
    plt.bar(sod['Year'],sod['Deaths'])
    plt.title('Relative Death Count')
    plt.xlabel('Year')
    plt.ylabel('Deaths')
    plt.plot(sod['Year'],sod['Self'], label = 'Self', color = 'red', marker = 'o')
    plt.plot(sod['Year'],sod['Other'], label = 'Others', color = 'green', marker = 'o')
    plt.plot(sod['Year'],sod['Disease'], label = 'Diseases', color = 'black', marker = 'o')
    plt.legend(loc = 'upper right')
    plt.show()
    
    
    vehicle_data = import_vehicle_data('Motor_Vehicle_Collisions_-_Crashes.csv')
    vehicle_data = clean_vehicle_data(vehicle_data)
    print(vehicle_data['COLLISION_ID'])
    
    #code for getting the car crashes to be mapped 
    vehicle_crashes = vehicle_data.groupby('BOROUGH')['NUMBER OF PERSONS KILLED'].sum()
    vehicle_crashes = vehicle_crashes.to_frame()
    housing_database = import_data("Housing_Database_by_NTA.csv")
    housing_database = housing_database[ ["nta2010", "boro"]]
    housing_database['BOROUGH'] = housing_database['boro'].str.upper()
    housing_database = housing_database.drop(['boro'], axis = 1)
    map_data = housing_database.merge(vehicle_crashes,left_on='boro', right_on='BOROUGH')
    map_data = vehicle_crashes.merge(housing_database,on='BOROUGH',how='outer')
    map_vehicle_crashes(map_data)
    
    collisions = vehicle_data.groupby('YEAR')['COLLISION_ID'].sum()
    collisions = collisions.reset_index()
    killed = vehicle_data.groupby('YEAR')['NUMBER OF PERSONS KILLED'].sum()
    killed = killed.reset_index()
    injured = vehicle_data.groupby('YEAR')['NUMBER OF PERSONS INJURED'].sum()
    injured = injured.reset_index()
    
    #CIK is data frame summary of collisions, killed, and injured
    cik = collisions.merge(killed, on = 'YEAR')
    cik = cik.merge(injured, on = 'YEAR')
    print(cik)

    #this part uses data from cik (collisions_injured_killed) since the original data's collisons are all 1.
    b_car, m_car = compute_lin_reg(cik['COLLISION_ID'], cik['NUMBER OF PERSONS KILLED'])
    xes = np.array([0,cik['COLLISION_ID'].max()])
    yes = m_car*xes + b_car
    plt.scatter(cik['COLLISION_ID'],cik['NUMBER OF PERSONS KILLED'])
    plt.plot(xes,yes,color='r')
    plt.title(f'Regression line of KILLED with m = {m_car:{4}.{2}} and y-intercept = {b_car:{4}.{4}}')
    plt.show()
    
    b_car, m_car = compute_lin_reg(cik['COLLISION_ID'], cik['NUMBER OF PERSONS INJURED'])
    xes = np.array([0,cik['COLLISION_ID'].max()])
    yes = m_car*xes + b_car
    plt.scatter(cik['COLLISION_ID'],cik['NUMBER OF PERSONS INJURED'])
    plt.plot(xes,yes,color='r')
    plt.title(f'Regression line of INJURED with m = {m_car:{4}.{2}} and y-intercept = {b_car:{4}.{4}}')
    plt.show()

    
    
    plt.bar(cik['YEAR'],cik['COLLISION_ID'])
    plt.title('Killed vs. Injured on Collisions')
    plt.xlabel('Year')
    plt.ylabel('Collisions')
    """
    # Uncoment to get the Collision per Year Graph
    plt.bar(cik['YEAR'],cik['COLLISION_ID'])
    plt.title('Collision per Year')
    plt.xlabel('Year')
    plt.ylabel('Collisions')
    """
    plt.plot(cik['YEAR'],cik['COLLISION_ID'], label = 'Killed', color = 'green', marker = 'o')
    plt.plot(cik['YEAR'],cik['NUMBER OF PERSONS KILLED'], label = 'Killed', color = 'red', marker = 'o')
    plt.plot(cik['YEAR'],cik['NUMBER OF PERSONS INJURED'], label = 'Injured', color = 'black', marker = 'o')
    plt.legend(loc = 'upper right')
    plt.show()


    
main()