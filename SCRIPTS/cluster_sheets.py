# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 12:47:06 2018

@author: kennedy
"""

from os import chdir
import pandas as pd
import numpy as np



class Master_sheet(object):
  '''Docstring
  
  Functions:
    dataframe:
      This function is responsible for collecting
      the sheets in the workbook
      
      return type:
        returns the sheets in the workbook
        
    preprocess:
      This function is responsible for processing
      the worksheets
      
      return type:
        returns the processed worksheets.
        
  '''
  
  
  def dataframe(dataset):
    #load sheets into dataframe
    sheet_name = []
    for ii in dataset.sheet_names:
      sheet_name.append(ii)
    #sheet CC
    df_CC = dataset.parse(sheet_name[0])
    #sheet BR
    df_BR = dataset.parse(sheet_name[1])
    #sheet Mea
    df_Mea = dataset.parse(sheet_name[2])
    #Sheet Peos
    df_Peos = dataset.parse(sheet_name[3])
    #Sheet Wilms
    df_Wilms = dataset.parse(sheet_name[4])
    return df_CC, df_BR, df_Mea, df_Peos, df_Wilms


  def preprocess(df):
    '''
    :Return:
      processed datasheets
    '''
    df = df.dropna(how='all', thresh=50)
    df = df.dropna(how='all', thresh=5, axis = 1)
    df.columns = df.iloc[0]
    df = df.iloc[1:]
    df = df.reset_index()
    df = df.iloc[:, 1:]
  #  df = df.set_index('Device')
    return df




#check to see if there are any nan values in our dataset 
#df_BR.isnull().any().any()

  
#if __name__ == '__main__':
#  
#  df_CC, df_BR, df_Mea, df_Peos, df_Wilms = dataframe(dataset)
#  #preprocess the sheets
#  df_BR = preprocess(df_BR)
#  df_CC = preprocess(df_CC)
#  df_Mea = preprocess(df_Mea)
#  df_Peos = preprocess(df_Peos)
#  df_Wilms = preprocess(df_Wilms)

