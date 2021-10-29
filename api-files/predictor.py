import os
import pickle

import numpy as np


class ClientRanker():
    def __init__(self, model):
        self.home_path             = os.getcwd()
        self.vintage_scaler        = pickle.load(open(os.path.join(self.home_path,'../pkl/vintage_scaler.pkl'), 'rb'))
        self.annual_premium_scaler = pickle.load(open(os.path.join(self.home_path,'../pkl/annual_premium_scaler.pkl'), 'rb'))
        self.age_scaler            = pickle.load(open(os.path.join(self.home_path,'../pkl/age_scaler.pkl'), 'rb'))
        self.region_code_encoder   = pickle.load(open(os.path.join(self.home_path,'../pkl/region_code_encoder.pkl'), 'rb'))
        self.sales_channel_encoder = pickle.load(open(os.path.join(self.home_path,'../pkl/sales_channel_encoder.pkl'), 'rb'))
        self.model                 = model
    
    def _set_columns_names(self, df):
        df.columns= df.columns.str.lower()
        return df
    
    def _rescale(self, df):
        df['vintage']         = self.vintage_scaler.transform( df[['vintage']] )
        df['annual_premium']  = self.annual_premium_scaler.transform(df[['annual_premium']])
        df['age']             = self.age_scaler.transform( df[['age']] )
        
        return df

    def _encode(self, df):
        df['vehicle_damage'] = df[['vehicle_damage']].apply(lambda x: 1 if x['vehicle_damage'] == 'Yes' else 0, axis=1)

        df['region_code']  = self.region_code_encoder.transform(df[['region_code']])

        df['policy_sales_channel']  = self.sales_channel_encoder.transform(df[['policy_sales_channel']])
        
        return df.dropna()
     
                                                 
    def _select_columns(self, df):
        return df[['vintage','annual_premium','age','region_code','vehicle_damage','policy_sales_channel','previously_insured']]
    
    def _score_clients(self, df):
        clients_score = self.model_adaboost.predict_proba(df)
        df['score']   = clients_score[:,1]
        return df
        
        
    def _rescale_score(self, df):
        df['score']   = df[['score']].apply(lambda x: np.exp(x['score']+5), axis=1)
        
        return df
    
    def predict(self, df):
        df = self._set_columns_names(df)
        df = self._select_columns(df)
        df = self._rescale(df)
        df = self._encode(df)
        df = self._score_clients(df)
        df = self._rescale_score(df)
        
        return df