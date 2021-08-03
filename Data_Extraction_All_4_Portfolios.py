#!/usr/bin/env python
# coding: utf-8

# In[10]:


#Installing packages
get_ipython().system('pip install yfinance')
get_ipython().system('pip install arch')
get_ipython().system('pip install hurst')

import yfinance as yf  #Package to extract data from yahoofinance.com
import pandas as pd    #Package to process data in dataframes
import math            #Package to assess N/A
import numpy as np     #Package to compute returns
import datetime        #Package to enter the dates correctly 
import matplotlib.pyplot as plt  #Package to plot series
import  statsmodels.api as sm   #Package to run an OLS regression on the log_prices to establish cointegration

from dateutil.relativedelta import relativedelta #Package to compute the rolling windows
from arch.unitroot import PhillipsPerron #Package to test the stationarity of the residuals
from hurst import compute_Hc  #Package to compute hurst exponent


# In[604]:


#Function to save data extracted from yahoo finance as a cvs file
def SaveData(df,dataname):
    df.to_csv('./'+dataname+'.csv')

#####################################################################################################
#Function to extract data 
#####################################################################################################
#Function to extract the data from yahoo finance
def getData(tick):
    ticker = yf.Ticker(tick)
    dataname=tick
    
    #Extracting the closing price 
    data = ticker.history(start = start_date,end=end_date)
    data = data["Close"]
    
    #Extracting the dividends and splits
    div_split=ticker.actions
    
    #Data processing the closing price to take into account stock splits
    counter_1 = 0
    for price in data[:len(data)]:
        str_1 = data.index[counter_1]
        counter_2 = 0
        for day in div_split["Dividends"][:len(div_split["Dividends"])]:
            str_2 = div_split["Dividends"].index[counter_2]
            if str_1 == str_2:
                ratio = div_split["Stock Splits"][counter_2]
                if ratio!=0:
                    counter_3=counter_1+1
                    for gross_price in data[counter_1+1:len(data)]:
                        gross_price = gross_price*ratio
                        data.iat[counter_3] = gross_price
                        counter_3+=1
            counter_2+=1
        counter_1 += 1
    
    #Data processing the closing price to add back dividends
    counter_1 = 0
    for price in data[:len(data)]:
        str_1 = data.index[counter_1]
        counter_2 = 0
        for day in div_split["Dividends"][:len(div_split["Dividends"])]:
            str_2 = div_split["Dividends"].index[counter_2]
            if str_1 == str_2:
                dividend = div_split["Dividends"][counter_2]
                price = price + dividend
                data.iat[counter_1] = price
            counter_2+=1
        counter_1 += 1
    
    data = pd.DataFrame(data)    
    data.rename(columns={'Close':dataname},inplace =  True)
    return data


#Function to get the first row of available data
def First(data):
    counter=0
    for ele in data:
        if math.isnan(ele)==False:
            break
        counter+=1
    return counter

#Function to fill in the blanks
def Filling(data):
    start = First(data)
    for ele in data[start:len(data)]:
        if math.isnan(ele)==True:
            data.iat[start] = data[start-1]
        start+=1       
    return data


##############################################################################################################
#Function to compute based on a rolling window
##############################################################################################################
 
#Function to determine the 6 months date
def resize_months(start_date, data, number_months):
    start_date = data.index[start_date]
    end_date = start_date+relativedelta(months=+number_months)
    int_ = data.index.searchsorted(end_date)+1
    return int_

def clean_up(decision_window):
    for ticker in decision_window:
        lenght = (len(decision_window[ticker].dropna(axis=0)))
        if lenght<25:
            decision_window.drop(columns=[ticker],inplace = True)
    return decision_window

###############################################################################################################
#Function used in pair construction
###############################################################################################################

#Function that take as a parameter a row of the pair dataframe and constructs a dataframe with their returns
def pair_return(data,row):
    df= pd.concat([data[row[0]],data[row[1]]],axis=1).dropna(axis=0)
    return df

#Function that retruns the index number of a date
def index_number(date,data):
    df_1 = data.copy()
    df = pd.DataFrame(df_1)
    df['A']=np.arange(len(df))
    return df['A'][date]

#Function to clean up data in order to avoid having two values for the same date
def clean_up_data(data):
    return data.reset_index().drop_duplicates(subset='index').set_index('index')

#Function to avoid double pairs and returns a boolean
def check_double(data, row):
    test=False
    for ele in data:
        if set(ele)==set(row):
            test= True
    return test
    


# In[594]:


######################################################CORRELATION BASED PORTFOLIO##############################################

#############################################################################################################
#Functions related to the best pair based on correlation
#############################################################################################################

#Function to determine the max other than 1 
def max_corr(data):
    max_ = -1
    for ele in data : 
        if max_< ele and ele!=1: 
            max_=ele
    return max_
            
#Function that returns the perfect pair for a given list 
def pair_of(data):
    counter = 0 
    for corr in data:
        if corr == max_corr(data):
            break
        counter+=1
    return data.index[counter]

##############################################################################################################
#Function that returns a dataframe containing all the pairs based on correlation
##############################################################################################################
def pairs(returns):
    corr_matrix = returns.corr(method='spearman')
    matrix=[]
    pair_1 = corr_matrix.index[0]
    pair_2 = pair_of(corr_matrix[pair_1])
    row = [pair_1,pair_2]
    matrix.append(row)
    for ele in corr_matrix.index:
        pair_1 = ele
        pair_2 = pair_of(corr_matrix[pair_1])
        row = [pair_1,pair_2]
        if check_double(matrix,row)==False:
            matrix = np.vstack([matrix,row])
    return matrix      

############################################################################################################
#Function to compute rolling correlation for a six month period takes as parameter pair return
############################################################################################################
def rolling_corr(data):
    #Initializing
    corr_6m =[]
    index=[]

    #Adjusting the run_time of the loop
    if len(data)>27:
        adjust_ = 1
    elif len(data) <= 27: 
        adjust_ = len(data) - 27
     
    for start_date in range(len(data)-25-adjust_):
        
        #Specifying the rolling window
        end_date = resize_months(start_date,data,6)
        if len(data)<end_date: 
            end_date = len(data)
        index.append(data.index[end_date-1])
        window = data[start_date:end_date]
        
        #Computing the correlation
        corr = window.corr(method='spearman')[data.columns[0]][data.columns[1]]
        
        #Stocking the correlation in the matrix
        corr_6m.append(corr)
        
    #Reindexing
    corr_6m = pd.DataFrame(corr_6m)
    for i in range(0,len(corr_6m)):
        corr_6m.rename(index={i:index[i]},inplace = True)
    corr_6m.rename(columns={0:'Rolling Correlation'},inplace = True)
    
    return clean_up_data(corr_6m)
 
###############################################################################################################
#Function to compute the moving average of the series takes as a parameter the rolling correlation
###############################################################################################################
def moving_average(rolling_cor):
    #Initializing
    moving_aver = []
    index=[]
    
    #Adjusting the run_time of the loop
    if len(rolling_cor)>27:
        adjust = 1
    elif len(rolling_cor) < 27 or len(rolling_cor)==27: 
        adjust = len(rolling_cor) - 27

    for start_date in range(0,len(rolling_cor)-24-adjust):
        #Specifying the moving window
        end_date = resize_months(start_date,rolling_cor,6)
        if len(rolling_cor)<end_date: 
            end_date = len(rolling_cor)
        
        index.append(rolling_cor.index[end_date-1])
        window = rolling_cor[start_date:end_date]
        
        #Computing the mean and the vol
        mean = window.mean()[0]
        vol = np.std(window)[0]
        moving_aver.append([mean,vol])
        
    #Reindexing
    moving_aver = pd.DataFrame(moving_aver)
    for i in range(0,len(moving_aver)):
        moving_aver.rename(index={i:index[i]},inplace = True)
    moving_aver.rename(columns={0: 'Correlation Mean', 1: 'Correlation Vol'},inplace = True)
    return clean_up_data(moving_aver)



##############################################################################################################
#Function that executes the trading strategy of the pair
##############################################################################################################
def trading_correlation(computation_window,trading_window, row_of_pair, delta_entry, delta_exit):
    
    #Stocking data in respective variable
    returns = pair_return(computation_window,row_of_pair)
    prices = pair_return(adj_close_w,row_of_pair)
    df = pd.concat([rolling_corr(returns),moving_average(rolling_corr(returns))],axis=1).dropna(axis=0)
    
    rolling_correlation = df['Rolling Correlation']
    moving_avrg = df['Correlation Mean']
    stdev = df['Correlation Vol']

    
    #Initializing
    weights_1 = 0
    weights_2 = 0
    signal_1 = False 
    signal_2 = False
    portfolio = []
    
    #Executing the trade for the given period
    for i in range(len(trading_window)):
        
        #Stocking info in  respective variable
        date = trading_window.index[i]
        return_1 = returns[row_of_pair[0]][date]
        return_2 = returns[row_of_pair[1]][date]
        price_1 = prices[row_of_pair[0]][date]
        price_2 = prices[row_of_pair[1]][date]
        
        #Finding the earliest date with rolling correlation its mean and it vol available
        date_r = rolling_correlation.index.searchsorted(date)
        if date_r >= len(rolling_correlation): 
            date_r += -1 # len(rolling_correlation.index)
        date_r = rolling_correlation.index[date_r]
        correlation = rolling_correlation[date_r]
        mean = moving_avrg[date_r]
        vol = stdev[date_r]
        
        #print('correlation is ', correlation , ' mean is ', mean, ' vol is ',vol)
         
         
        #Readjusting weights of the price
        if correlation < mean - delta_entry*vol:   #Trading strategy is triggered 
            #Determining the overpriced and undervalued stock 
            if return_1 > return_2: 
                weights_1 += -1
                weights_2 += 1
            else:
                weights_1 += 1
                weights_2 += -1
            signal_1 = True
            
        elif correlation > mean + delta_exit and signal_1 == True:   #Unwind the trading strategy
            weights_1 = 0 
            weights_2 = 0
            
                
        elif correlation > mean + delta_entry:   #Trading strategy is triggered
            #Determining the overpriced and undervalued stock
            if return_1 < return_2: 
                weights_1 += -1
                weights_2 += 1
            else:
                weights_1 += 1
                weights_2 += -1
            signal_2 = True 
        
        elif correlation < mean - delta_exit and signal_1 == True:   #Unwind the trading strategy
            weights_1 = 0 
            weights_2 = 0
        
        #print(weights_1, ' ___ ', weights_2)
        portfolio.append(weights_1*price_1 + weights_2*price_2)
    
    #Reindexing
    portfolio = pd.DataFrame(portfolio)
    for i in range(0,len(portfolio)):
        portfolio.rename(index={i:trading_window.index[i]},inplace = True)
    portfolio.rename(columns={0:'Portfolio Value '+ row_of_pair[0]+'-'+row_of_pair[1]},inplace = True)
    
    return portfolio
    

    
    


# In[611]:


##############################################################################################################################
###Executing the strategy on a rolling basis: first year to construct the pairs then execute the trading strategy in the 
#following six months, the pair selection is refreshed using the accumulated data since the first initial year 
##############################################################################################################################

def execution_correlation_strategy(weekly_ret,delta_entry,delta_exit):
    #Initializing
    #Defining the dataframe that will contain
    correlation_portfolio=[]

    #Decision window based on which the pairs will be traded
    decision_end_date = resize_months(0,weekly_ret,12)
    decision_window = clean_up(weekly_ret[1:decision_end_date])

    #Trading window based on which the trading strategy will be executed
    trading_start_date = decision_end_date + 1
    trading_end_date = resize_months(trading_start_date,weekly_ret,6)
    trading_window = weekly_ret[trading_start_date:trading_end_date].dropna(axis=1,thresh =2)

    #Computation window based on which we will be computing the rolling correlation, its mean and its vol
    computation_window = weekly_ret[0:trading_end_date]

    #Determining the pair trade based on a initial one year window and expanding
    matrix = pairs(decision_window)

    #Executing the pair trade based on the distance method for all pairs in the matrix
    portfolio_ret = pd.DataFrame(trading_correlation(pair_return(computation_window,matrix[0]),pair_return(
        trading_window,matrix[0]), matrix[0],delta_entry,delta_exit))
    for row in matrix[1:]:
        portfolio_ret = pd.concat([portfolio_ret,trading_correlation(pair_return(computation_window,row),pair_return(
        trading_window,row), row,delta_entry,delta_exit)],axis=1)
    portfolio_ret = pd.DataFrame(portfolio_ret.sum(axis=1)) 
    correlation_portfolio = portfolio_ret
    print(correlation_portfolio)

    #Looping for the rest of the period
    for start_date in range(1,len(weekly_ret)-81):

        #Decision window based on which the pairs will be traded
        decision_end_date = resize_months(start_date,weekly_ret,12)
        decision_window = clean_up(weekly_ret[0:decision_end_date])

        #Trading window based on which correlation will be evaluated and the trade strategy will be executed
        trading_start_date = decision_end_date + 1
        trading_end_date = resize_months(trading_start_date,weekly_ret,6)
        if trading_end_date>len(weekly_ret):
            trading_end_date = len(weekly_ret)
        trading_window = weekly_ret[trading_start_date:trading_end_date].dropna(axis=1,thresh = 2)

        #Computation window based on which we will be computing the rolling correlation, its mean and its vol
        computation_window = weekly_ret[0:trading_end_date]

        #Determining the pair trade based on a initial one year window and expanding
        matrix = pairs(decision_window)

        #Executing the pair trade based on correlation
        portfolio_ret = pd.DataFrame(trading_correlation(pair_return(computation_window,matrix[0]),pair_return(
            trading_window,matrix[0]), matrix[0],delta_entry,delta_exit))

        for row in matrix[1:]:
            portfolio_ret = pd.concat([portfolio_ret,trading_correlation(pair_return(computation_window,row),pair_return(
                trading_window,row), row,delta_entry,delta_exit)],axis=1)
        portfolio_ret = pd.DataFrame(portfolio_ret.sum(axis=1))
        correlation_portfolio = pd.concat([correlation_portfolio,portfolio_ret],axis=0)

    #Computing the returns of the portfolio based on correlation
    correlation_portfolio.rename(columns={0:'Correlation Portfolio'},inplace = True)
    correlation_portfolio_ret = ((correlation_portfolio - correlation_portfolio.shift(1))/correlation_portfolio.shift(1)).fillna(0)
    correlation_portfolio_ret.rename(columns={'Correlation Portfolio':'Correlation Portfolio Returns'},inplace = True)
    return correlation_portfolio_ret


# In[607]:


##################################################COINTEGRATION BASED PORTFOLIO################################################

####################################################################################################################
#Function that takes as parameters adj closing prices and returns a matrix with the most suited pairs
####################################################################################################################

def cointegration (data):
    data = np.log(data)
    matrix = ["_","_"]
    for ticker_1 in data.columns:
        #Dataframe to stock all p_values
        p_value=[]
        
        #Looping through tickers 
        for ticker_2 in data.columns:
            
            #Computing the p_value for each other stock
            if ticker_1 != ticker_2:
                data_coint = pair_return(data,[ticker_1,ticker_2]).dropna(axis=0)
                Y =  data_coint[ticker_1]
                X = data_coint[ticker_2]
                X = sm.add_constant(X)
                ols_regression = sm.OLS(Y,X).fit()
                residuals = Y - ols_regression.fittedvalues
                if math.isnan(residuals.sum(axis=0))==False and len(residuals.dropna(axis=0))>4:
                    pp = PhillipsPerron(residuals.dropna(axis=0))
                    p_value.append([ticker_2,pp.pvalue])
        
        #Deteminin the most cointegrated ticker_2 with ticker_1
        p_value = pd.DataFrame(p_value)
        for i in range(len(p_value[1])): 
            if p_value[1][i] == p_value[1].max():
                row = [ticker_1,p_value[0][i]]
                
        #Stocking the pairs in a matrix
        if check_double(matrix,row)==False:
            matrix = np.vstack([matrix,row])
            
    return matrix[1:]

###################################################################################################################
#Function to extract the moving mean of the cointegrated model and the cointegrated coefficient
###################################################################################################################
def moving_mean_coef(data):  #adj_close_w only contains the two tickers
    moving_mean = []
    index=[]
    
    #Adjusting the run_time of the loop

    if len(data) == 27 :
        adjust = 0 
    elif len(data)>27:
        adjust = 1
    elif len(data) < 27: 
        adjust = len(data) - 27
        
    for start_date in range(len(data)-26-adjust):
        
        #Defining the rolling window
        end_date = resize_months(start_date,data,6)
        if len(data)<end_date: 
            end_date = len(data)
        index.append(data.index[end_date-1])
        window = data[start_date:end_date]
        
        #Running the regression on the rolling window
        Y = window[data.columns[0]]
        X = window[data.columns[1]]
        X = sm.add_constant(X)
        ols_regression = sm.OLS(Y,X).fit()
        stdev= np.std(Y-ols_regression.params[1]*window[data.columns[1]])
        moving_mean.append([ols_regression.params[0],ols_regression.params[1],stdev])
        
        
    #Reindexing
    moving_mean = pd.DataFrame(moving_mean)
    for i in range(0,len(moving_mean)):
        moving_mean.rename(index={i:index[i]},inplace = True)
    moving_mean.rename(columns={0:'Cointegration Mean',1:'Cointegration Coefficient',2:'Volatility of the Spread'},inplace = True)
    
    return clean_up_data(moving_mean)

##########################################################################################################################
#Function to execute the trading strategy based on cointegration and returns a DataFrame of the value of the portfolio
##########################################################################################################################
def trading_cointegration(adj_close_w, row_of_pair, delta_entry,  delta_exit):
    
    #Stocking data in respective variable
    prices = pair_return(adj_close_w,row_of_pair)
    coint_mean_coef = moving_mean_coef(prices)

    spread = prices[row_of_pair[0]]-coint_mean_coef['Cointegration Coefficient']*prices[row_of_pair[1]]
    
     #Initializing
    weights_1 = 0
    weights_2 = 0
    portfolio = []
    signal_1 = False 
    signal_2 = False
    
    
    #Executing the trade for the given period
    for i in range(len(coint_mean_coef)):

        #Stocking info in  respective variable
        date = coint_mean_coef.index[i]
        price_1 = prices[row_of_pair[0]][date]
        price_2 = prices[row_of_pair[1]][date]
        
        stdv = coint_mean_coef['Volatility of the Spread'][date]
        coint_mean = coint_mean_coef['Cointegration Mean'][date]
        spread_t = spread [date]

        
        #Readjusting weights 
        if spread_t < coint_mean - delta_entry*stdv :
            weights_1 +=1
            weights_2 +=-1
            signal_1 = True
        elif signal_1 == True and spread_t > coint_mean + delta_exit*stdv: 
            weights_1 =0
            weights_2 =0
            signal_1 = False
        
        elif spread_t > coint_mean + delta_entry*stdv:
            weights_1+=-1
            weights_2+=1
            signal_2 = True
        elif signal_2 == True and spread_t < coint_mean - delta_exit*stdv:
            weights_1 =0
            weights_2 =0
            signal_2 = False
        
        portfolio.append(weights_1*price_1 + weights_2*price_2) 
       
    #Reindexing
    portfolio = pd.DataFrame(portfolio)
    for i in range(0,len(portfolio)):
        portfolio.rename(index={i:coint_mean_coef.index[i]},inplace = True)
    portfolio.rename(columns={0:'Portfolio Returns '+row[0]+'-'+row[1]},inplace = True)
 
    return portfolio


##############################################################################################################################
###Executing the strategy on a rolling basis: first year to construct the pairs then execute the trading strategy in the 
#following six months, the pair selection is refreshed using the accumulated data since the first initial year 
##############################################################################################################################
def execution_cointegration_method(adj_close_w, delta_entry,delta_exit):
    #Initializing
    #Defining the dataframe that will contain the portfolio
    cointegration_portfolio=[]

    #Decision window based on which the pairs will be traded
    decision_end_date = resize_months(0,adj_close_w,12)
    decision_window = clean_up(adj_close_w[1:decision_end_date])

    #Trading window based on which cointegration will be evaluated and the trade strategy will be executed
    trading_start_date = decision_end_date + 1
    trading_end_date = resize_months(trading_start_date,adj_close_w,6)
    trading_window = adj_close_w[trading_start_date:trading_end_date].dropna(axis=1,thresh=10)


    #Determining the pair trade based on a initial one year window and expanding
    matrix = cointegration(decision_window)

    #Executing the pair trade based on cointegration for all pairs in the matrix
    portfolio_ret = pd.DataFrame(trading_cointegration(pair_return(trading_window, matrix[0]), matrix[0],delta_entry,delta_exit))
    for row in matrix[1:]:
        portfolio_ret = pd.concat([portfolio_ret,trading_cointegration(pair_return(trading_window,row), row,delta_entry,delta_exit)],axis=1)
    portfolio_ret = pd.DataFrame(portfolio_ret.sum(axis=1)) 
    cointegration_portfolio = portfolio_ret


    #Looping for the rest of the period
    for start_date in range(1,len(adj_close_w)-81):

        #Decision window based on which the pairs will be traded
        decision_end_date = resize_months(start_date,adj_close_w,12)
        decision_window = clean_up(adj_close_w[1:decision_end_date])

        #Trading window based on which cointegration will be evaluated and the trade strategy will be executed
        trading_start_date = decision_end_date + 1
        trading_end_date = resize_months(trading_start_date,adj_close_w,6)
        if trading_end_date>len(adj_close_w):
            trading_end_date = len(adj_close_w)
        trading_window = adj_close_w[trading_start_date:trading_end_date].dropna(axis=1,thresh=10)

        #Determining the pair trade based on a initial one year window and expanding
        matrix = cointegration(decision_window)

        #Executing the pair trade based on cointegration
        portfolio_ret = pd.DataFrame(trading_cointegration(pair_return(trading_window, matrix[0]), matrix[0],delta_entry,delta_exit))
        for row in matrix[1:]:
            portfolio_ret = pd.concat([portfolio_ret,trading_cointegration(pair_return(trading_window,row), row,delta_entry,delta_exit)],axis=1)
        portfolio_ret = pd.DataFrame(portfolio_ret.sum(axis=1))
        cointegration_portfolio = pd.concat([cointegration_portfolio,portfolio_ret],axis=0)

    #Computing the returns of the portfolio
    cointegration_portfolio.rename(columns={0:'Cointegration Portfolio'},inplace = True)
    cointegration_portfolio_ret = ((cointegration_portfolio - cointegration_portfolio.shift(1))/cointegration_portfolio.shift(1)).fillna(0).fillna(0)
    cointegration_portfolio_ret.rename(columns={0:'Cointegration Portfolio Returns'},inplace = True)
    return cointegration_portfolio_ret


# In[379]:


###############################################DISTANCE METHOD BASED PORTFOLIO###########################################

##########################################################################################################################
#Function that takes as parameters returns a matrix with the most suited pairs
##########################################################################################################################
#Meaning the pair that have the lowest sum of squared differences between the cumulative return
def distance_method_pairs(data):#the data is returns
    
    #Initializing
    matrix = ["_","_"]
    
    #Because we are dealing with cumulative differences
    data = data.dropna(axis=0)
    
    #Selecting the best pairs
    for ticker_1 in data.columns:
        
        #Dataframe to stock all SSD
        SSD =[]
        
        #Looping through all pairs to compute the squared difference
        for ticker_2 in data.columns:
            if ticker_1!=ticker_2:
                cum_returns = pair_return(data,[ticker_1,ticker_2]).cumsum()
                Squared_Diff = (cum_returns[ticker_1]-cum_returns[ticker_2])**2
                SSD.append([ticker_2,Squared_Diff.sum(axis =0)])
        
        #Detemining the shortest distance between ticker_1 and ticker 2
        SSD = pd.DataFrame(SSD)
        for i in range(len(SSD[1])): 
            if SSD[1][i] == SSD[1].min():
                row = [ticker_1,SSD[0][i]]
                
        #Stocking the pairs in a matrix
        if check_double(matrix,row)==False:
            matrix = np.vstack([matrix,row])
            
    return matrix[1:]

##############################################################################################################################
#Function to execute the trading strategy based on the distance method and returns a DataFrame of the value of the portfolio
##############################################################################################################################
def trading_distance_method(returns, row_of_pair, delta_entry, delta_exit):
    
    #Stocking data in respective variable
    prices = pair_return(adj_close_w,row_of_pair)
    returns = pair_return(returns,row_of_pair)
    
     #Initializing
    weights_1 = 0
    weights_2 = 0
    signal_1 = False 
    signal_2 = False
    portfolio = []
    
    #Executing the trade for the given period
    for i in range(len(returns)):
        
        #Stocking info in  respective variable
        date = returns.index[i]
        price_1 = prices[row_of_pair[0]][date]
        price_2 = prices[row_of_pair[1]][date]
        return_1 = returns[row_of_pair[0]][date]
        return_2 = returns[row_of_pair[1]][date]
      
        vol_1 = np.std(returns[row_of_pair[0]][:date])
        vol_2 = np.std(returns[row_of_pair[1]][:date])
        spread = return_1/vol_1 - return_2/vol_2
    
       
        #Readjusting weights 
        if spread < delta_entry :
            weights_1 +=1
            weights_2 +=-1
            signal_1 = True
        elif signal_1 == True and spread > delta_exit: 
            weights_1 =0
            weights_2 =0
            signal_1 = False
        
        elif spread > delta_entry:
            weights_1+=-1
            weights_2+=1
            signal_2 = True
        elif signal_2 == True and spread < delta_exit:
            weights_1 =0
            weights_2 =0
            signal_2 = False
                  
        portfolio.append(weights_1*price_1 + weights_2*price_2)
        
    #Reindexing
    portfolio = pd.DataFrame(portfolio)
    for i in range(len(portfolio)):
        portfolio.rename(index={i:returns.index[i]},inplace = True)
    portfolio.rename(columns={0:'Portfolio '+row[0]+'-'+row[1]},inplace = True)

    return portfolio

##############################################################################################################################
###Executing the strategy on a rolling basis: first year to construct the pairs then execute the trading strategy in the 
#following six months, the pair selection is refreshed using the accumulated data since the first initial year 
##############################################################################################################################
def execution_distance_model_strategy(weekly_ret, delta_entry, delta_exit):

    #Initializing
    #Defining the dataframe that will contain
    distance_method_portfolio=[]

    #Decision window based on which the pairs will be traded
    decision_end_date = resize_months(0,weekly_ret,12)
    decision_window = clean_up(weekly_ret[0:decision_end_date])

    #Trading window based on which the trading strategy will be executed
    trading_start_date = decision_end_date + 1
    trading_end_date = resize_months(trading_start_date,weekly_ret,6)
    trading_window = weekly_ret[trading_start_date:trading_end_date].dropna(axis=1,thresh=10)


    #Determining the pair trade based on a initial one year window and expanding
    matrix = distance_method_pairs(decision_window)

    #Executing the pair trade based on the distance method for all pairs in the matrix
    portfolio_ret = pd.DataFrame(trading_distance_method(pair_return(trading_window, matrix[0]), matrix[0],delta_entry,delta_exit))
    for row in matrix[1:]:
        portfolio_ret = pd.concat([portfolio_ret,trading_distance_method(pair_return(trading_window,row),row,delta_entry,delta_exit)],axis=1)
    portfolio_ret = pd.DataFrame(portfolio_ret.sum(axis=1)) 
    distance_method_portfolio = portfolio_ret

    #Looping for the rest of the period
    for start_date in range(1,len(weekly_ret)-81):

        #Decision window based on which the pairs will be traded
        decision_end_date = resize_months(start_date,weekly_ret,12)
        decision_window = clean_up(weekly_ret[0:decision_end_date])

        #Trading window based on which the distance method will be evaluated and the trade strategy will be executed
        trading_start_date = decision_end_date + 1
        trading_end_date = resize_months(trading_start_date,weekly_ret,6)
        if trading_end_date>len(weekly_ret):
            trading_end_date = len(weekly_ret)
        trading_window = weekly_ret[trading_start_date:trading_end_date].dropna(axis=1,thresh=10)

        #Determining the pair trade based on a initial one year window and expanding
        matrix = distance_method_pairs(decision_window)


        #Executing the pair trade based on the distance method
        portfolio_ret = pd.DataFrame(trading_distance_method(pair_return(trading_window, matrix[0]), matrix[0],delta_entry,delta_exit))
        for row in matrix[1:]:
            portfolio_ret = pd.concat([portfolio_ret,trading_distance_method(pair_return(trading_window,row), row,delta_entry,delta_exit)],axis=1)
        portfolio_ret = pd.DataFrame(portfolio_ret.sum(axis=1))
        distance_method_portfolio = pd.concat([distance_method_portfolio,portfolio_ret],axis=0)
    distance_method_portfolio.rename(columns={0:'Distance Portfolio'},inplace = True)

    #Computing the returns of the portfolio based on the distance method
    distance_method_portfolio_ret = ((distance_method_portfolio - distance_method_portfolio.shift(1))/distance_method_portfolio.shift(1)).fillna(0)
    distance_method_portfolio_ret.rename(columns={'Distance Portfolio':'Distance Portfolio Returns'},inplace = True)
    
    return distance_method_portfolio_ret


# In[ ]:


############################################## HURTS EXPONENT METHOD BASED PORTFOLIO ##########################################

##########################################################################################################################
#Function that takes as parameters returns a matrix with the most suited pairs
##########################################################################################################################
#Meaning the pair that have the lowest sum of squared differences between the cumulative return
def hurst_exponent_pairs(price,returns):#the data is ajd_close
    
    #Initializing
    matrix = ["_","_"]
    
    #
    
    #Selecting the best pairs
    for ticker_1 in price.columns:
        
        if len(price[ticker_1].dropna(axis=0))>100:
            
            #Dataframe to stock all hurst exponent
            Hurst =[]

            #Looping through all pairs to compute the time series
            for ticker_2 in price.columns:

                price_ = pair_return(price,[ticker_1,ticker_2])
                returns_ = pair_return(price,[ticker_1,ticker_2]) 

                if ticker_1!=ticker_2 and len(price_)>100:
                    #Computing the times series to which we will apply the hurst exponent following the model of Ramos-Requena et.al
                        #formula is log(price_1) - b*log(price_2) and b = std(log_ret_1)/std(log_ret_2)
                        #Drawback, we have to have at least 100 observations
                    coef_b = np.std(returns_[ticker_1])/np.std(returns_[ticker_2])
                    time_series = np.log(price_[ticker_1]) - coef_b*np.log(price_[ticker_2])

                    #Computing Hurst Exponent for the time series
                    H, c, val = compute_Hc(time_series)

                    #Saving the Exponent in a DataFrame with the corresponding ticker_2
                    Hurst.append([ticker_2,H])

            Hurst = pd.DataFrame(Hurst)
            #Detemining the shortest distance between ticker_1 and ticker 2
            Hurst = pd.DataFrame(Hurst)
            for i in range(len(Hurst[1])): 
                if Hurst[1][i] == Hurst[1].min():
                    row = [ticker_1,Hurst[0][i]]

            #Stocking the pairs in a matrix
            if check_double(matrix,row)==False:
                matrix = np.vstack([matrix,row])

    return matrix[1:]

##############################################################################################################################
###Executing the strategy on a rolling basis: first hundred observation which is two years to construct the pairs then execute the trading strategy in the 
#following six months, the pair selection is refreshed using the accumulated data since the first initial year 
##############################################################################################################################
#We will use the same function used for the distance methode
def execution_hurst_exponent_strategy(weekly_ret, adj_close_w, delta_entry, delta_exit):

    #Initializing
    #Defining the dataframe that will contain
    hurst_exponent_portfolio=[]

    #Decision window based on which the pairs will be traded
    decision_end_date = resize_months(0,weekly_ret,24)
    decision_window_return = clean_up(weekly_ret[0:decision_end_date])
    decision_window_price = clean_up(adj_close_w[0:decision_end_date])
    
    #Trading window based on which the trading strategy will be executed
    trading_start_date = decision_end_date + 1
    trading_end_date = resize_months(trading_start_date,weekly_ret,6)
    trading_window = weekly_ret[trading_start_date:trading_end_date].dropna(axis=1,thresh=10)


    #Determining the pair trade based on a initial one year window and expanding
    matrix = hurst_exponent_pairs(decision_window_price,decision_window_return)

    #Executing the pair trade based on the distance method for all pairs in the matrix
    portfolio_ret = pd.DataFrame(trading_distance_method(pair_return(trading_window, matrix[0]), matrix[0],delta_entry,delta_exit))
    for row in matrix[1:]:
        portfolio_ret = pd.concat([portfolio_ret,trading_distance_method(pair_return(trading_window,row),row,delta_entry,delta_exit)],axis=1)
    portfolio_ret = pd.DataFrame(portfolio_ret.sum(axis=1)) 
    hurst_exponent_portfolio = portfolio_ret

    #Looping for the rest of the period
    for start_date in range(1,len(weekly_ret)-132):

        #Decision window based on which the pairs will be traded
        decision_end_date = resize_months(start_date,weekly_ret,24)
        decision_window_return = clean_up(weekly_ret[0:decision_end_date])
        decision_window_price = clean_up(adj_close_w[0:decision_end_date])

        #Trading window based on which the distance method will be evaluated and the trade strategy will be executed
        trading_start_date = decision_end_date + 1
        trading_end_date = resize_months(trading_start_date,weekly_ret,6)
        if trading_end_date>len(weekly_ret):
            trading_end_date = len(weekly_ret)
        trading_window = weekly_ret[trading_start_date:trading_end_date].dropna(axis=1,thresh=10)

        #Determining the pair trade based on a initial one year window and expanding
        matrix = hurst_exponent_pairs(decision_window_price,decision_window_return)


        #Executing the pair trade based on the distance method
        portfolio_ret = pd.DataFrame(trading_distance_method(pair_return(trading_window, matrix[0]), matrix[0],delta_entry,delta_exit))
        for row in matrix[1:]:
            portfolio_ret = pd.concat([portfolio_ret,trading_distance_method(pair_return(trading_window,row), row,delta_entry,delta_exit)],axis=1)
        portfolio_ret = pd.DataFrame(portfolio_ret.sum(axis=1))
        hurst_exponent_portfolio = pd.concat([hurst_exponent_portfolio,portfolio_ret],axis=0)
    hurst_exponent_portfolio.rename(columns={0:'Hurst Exponent Portfolio'},inplace = True)

    #Computing the returns of the portfolio based on the distance method
    hurst_exponent_portfolio_ret = ((hurst_exponent_portfolio - hurst_exponent_portfolio.shift(1))/hurst_exponent_portfolio.shift(1)).fillna(0)
    hurst_exponent_portfolio_ret.rename(columns={'Hurst Exponent Portfolio':'Hurst Exponent Portfolio Returns'},inplace = True)
    
    return hurst_exponent_portfolio_ret


# In[612]:


#Defining ticker list + end and start date
ticker_list = ['NFLX','FB2A.DE','AAPL','BKNG34.SA','SU.PA','TKWY.AS','MSFT']
               
start_date = "1990-01-01"
end_date = "2021-03-31"

#Extracting the closing prices from yahoofinance.com and making adjustment on them
adj_close = pd.DataFrame(getData(ticker_list[0]))
for tick in ticker_list[1:len(ticker_list)]:
    print(tick)
    adj_close= pd.concat([adj_close,getData(tick)],axis=1)
    
#Cleaning up the data by filling in the gaps
for ticker in ticker_list:
    Filling(adj_close[ticker])

#Computing the weekly adjusted closing prices
adj_close_w = adj_close.resample("1W").last()
    
#Computing returns: 
daily_ret = np.log(adj_close) - np.log(adj_close.shift(1))
weekly_ret = np.log(adj_close_w) - np.log(adj_close_w.shift(1))

#Computing the in and out of sample dataframes
in_sample_returns = weekly_ret[datetime.datetime(1990,1,2):datetime.datetime(2016,12,28)].dropna(axis=1,how='all')
in_sample_prices = adj_close_w[datetime.datetime(1990,1,2):datetime.datetime(2016,12,28)].dropna(axis=1,how='all')
out_sample_returns = weekly_ret[datetime.datetime(2016,12,29):]
out_sample_prices = adj_close_w[datetime.datetime(2016,12,29):]



# In[614]:


print(in_sample_prices)

