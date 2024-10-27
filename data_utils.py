import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from glob import glob
from tqdm import tqdm
from collections import Counter
from sklearn.model_selection import train_test_split
from mne_features.feature_extraction import extract_features
import warnings
warnings.filterwarnings("ignore")

pd.options.mode.use_inf_as_na = True


def detect_outliers(df_in, col_name,coeff=1.5):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1
    fence_low  = q1-coeff*iqr
    fence_high = q3+coeff*iqr
    return (df_in[col_name] < fence_low) | (df_in[col_name] > fence_high)

def do_preprocessing(df,sampling_period):
    df.rename(columns={'Time': 'time'}, inplace=True) #some file has 'Time' column name -> deal with diffrent namings
    df['time'] = df['time'].apply(datetime.fromisoformat)
    df = df.set_index('time').resample(f'{sampling_period}ms').mean().reset_index()
    start_time = df.time.iloc[0] + timedelta(minutes=2)
    end_time =  df.time.iloc[-1] - timedelta(minutes=2)
    df = df.query('time>=@start_time and time <=@end_time').reset_index(drop=True)

    channel_columns = [col for col in df.columns if len(set.intersection(set(col.split('_')),set(['Theta', 'Alpha', 'beta', 'Gamma'])))>0]
    for ch in channel_columns:
        out_ids = detect_outliers(df,ch,3)
        df.loc[out_ids,ch] = None
    df.interpolate(inplace=True)
    
    if df.isna().sum().sum()>0:
        df = df.iloc[np.where(df.isna())[0].max()+1:][['time']+channel_columns].reset_index(drop=True)
    return df
    
def do_windowing(df,sampling_period,window_size,overlap):
    splitted_data = []
    min_len_thresh = int(window_size*1000/sampling_period) #minimum acceptable threshold for sample size (in case of time gaps)
    channel_columns = [col for col in df.columns if len(set.intersection(set(col.split('_')),set(['Theta', 'Alpha', 'beta', 'Gamma'])))>0]
    start_time = df['time'].iloc[0]
    while start_time<=df['time'].iloc[-1]:
        end_time = start_time + timedelta(seconds=window_size)
        sub_df = df.query('time>=@start_time and time<=@end_time')[channel_columns]
        if sub_df.shape[0]>=min_len_thresh:
            splitted_data.append(sub_df.values.T[:,:min_len_thresh])
        start_time = start_time + timedelta(seconds=window_size) - timedelta(seconds=window_size*overlap)
    return np.stack(splitted_data)

class EEGdataset:
    def __init__(self, path_to_data, sampling_period=149):
        self.sampling_period=sampling_period
        self.data_dict = {int(os.path.basename(eeg_csv_file).split('_')[-1].replace('.csv','')):pd.read_csv(eeg_csv_file) 
                          for eeg_csv_file in glob(os.path.join(path_to_data,'*.csv'))}
        with open(os.path.join(path_to_data,'label_dict.json')) as file:
            self.label_markup = json.load(file)
        self.prepare_data()
        
    def prepare_data(self):
        for subject_num,df in self.data_dict.items():
            self.data_dict[subject_num]=do_preprocessing(df,self.sampling_period)
        self.channel_columns = [col for col in df.columns if len(set.intersection(set(col.split('_')),set(['Theta', 'Alpha', 'beta', 'Gamma'])))>0]

    def get_splitted_data(self,window_size=2,overlap=0):
        splitted_data=[]
        label_list=[]
        subject_list=[]
        for subject_num,df in tqdm(self.data_dict.items()):
            tmp = do_windowing(df,self.sampling_period,window_size,overlap)
            splitted_data.append(tmp)
            subject_list+=[subject_num]*tmp.shape[0]
            new_label = 0 if self.label_markup[str(subject_num)] == 'casual' else 1
            label_list+=[new_label]*tmp.shape[0]
        assert np.vstack(splitted_data).shape[0] == len(label_list) == len(subject_list)
        return np.vstack(splitted_data),label_list,subject_list

    def get_train_val_test_indexes(self,subject_labels):
        train_ids = [n for n,i in enumerate(subject_labels) if str(i) in self.train_subjects]
        val_ids = [n for n,i in enumerate(subject_labels) if str(i) in self.val_subjects]
        test_ids = [n for n,i in enumerate(subject_labels) if str(i) in self.test_subjects]
        return train_ids,val_ids,test_ids

    def get_channel_mapping(self):
        return {ch:n for n,ch in enumerate(self.channel_columns)}
        
#helper functions for computing autonomic EEG indices 
def func1(x1,x2):
    return (x1-x2).mean(axis=1)
def func2(x1,x2):
    return ((x1-x2)/(x1+x2)).mean(axis=1)
def func3(x1,x2,x3,x4):
    return (x1/x2 - x3/x4).mean(axis=1)
    
def get_features(eeg_data,channel_mapping,sampling_rate,extract_rch_features=True):
    ch_names = list(channel_mapping.keys())
    #time-frequency domain features (based on spectral signal)
    extracted_features = extract_features(eeg_data,ch_names=ch_names,sfreq=sampling_rate,n_jobs=-1,selected_funcs=[
                                                               'app_entropy', 
                                                               'variance', 
                                                               'decorr_time',
                                                               'hjorth_complexity', 
                                                               'hjorth_mobility', 
                                                               'katz_fd', 
                                                               'kurtosis', 
                                                               'line_length', 
                                                               'mean', 
                                                               'ptp_amp', 
                                                               'quantile', 
                                                               'rms', 
                                                               'skewness', 
                                                               'std'],return_as_df=True)
    
    extracted_features = extracted_features.fillna(extracted_features.mean()) #deal with nan values appeared after features calculation if any
    feature_names =  ['_'.join(col) for col in extracted_features.columns]+[f'median_{ch}' for ch in ch_names]+[f'min_{ch}' for ch in ch_names]+[f'max_{ch}' for ch in ch_names]+[f'auc_{ch}' for ch in ch_names]
    add_features = np.hstack([np.median(eeg_data,axis=2),
                            np.min(eeg_data,axis=2),
                            np.max(eeg_data,axis=2),
                            np.trapz(eeg_data, dx = 1, axis=2)])
    
    extracted_features = np.hstack([extracted_features.values,add_features])

    if extract_rch_features:
        #autonomic EEG indices
        a_f4 = eeg_data[:,channel_mapping['F4_Alpha'],:]
        a_f3 = eeg_data[:,channel_mapping['F3_Alpha'],:]
        a_af4 = eeg_data[:,channel_mapping['AF4_Alpha'],:]
        a_af3 = eeg_data[:,channel_mapping['AF3_Alpha'],:]
        b_f4 = eeg_data[:,channel_mapping['F4_Low_beta'],:]+eeg_data[:,channel_mapping['F4_High_beta'],:]
        b_f3 = eeg_data[:,channel_mapping['F3_Low_beta'],:]+eeg_data[:,channel_mapping['F3_High_beta'],:]
        b_af4 = eeg_data[:,channel_mapping['AF4_Low_beta'],:]+eeg_data[:,channel_mapping['AF4_High_beta'],:]
        b_af3 = eeg_data[:,channel_mapping['AF3_Low_beta'],:]+eeg_data[:,channel_mapping['AF3_High_beta'],:]
    
        th_f4 = eeg_data[:,channel_mapping['F4_Theta'],:]
        th_f3 = eeg_data[:,channel_mapping['F3_Theta'],:]
        g_af4 = eeg_data[:,channel_mapping['AF4_Gamma'],:]
        g_af3 = eeg_data[:,channel_mapping['AF3_Gamma'],:]
    
        th_t7 = eeg_data[:,channel_mapping['T7_Theta'],:]
        th_t8 = eeg_data[:,channel_mapping['T8_Theta'],:]
        th_p7 = eeg_data[:,channel_mapping['P7_Theta'],:]
        th_p8 = eeg_data[:,channel_mapping['P8_Theta'],:]
    
        g_t7 = eeg_data[:,channel_mapping['T7_Gamma'],:]
        g_t8 = eeg_data[:,channel_mapping['T8_Gamma'],:]
        g_p7 = eeg_data[:,channel_mapping['P7_Gamma'],:]
        g_p8 = eeg_data[:,channel_mapping['P8_Gamma'],:]
    
        a_t7 = eeg_data[:,channel_mapping['T7_Alpha'],:]
        a_t8 = eeg_data[:,channel_mapping['T8_Alpha'],:]
        a_p7 = eeg_data[:,channel_mapping['P7_Alpha'],:]
        a_p8 = eeg_data[:,channel_mapping['P8_Alpha'],:]
    
        b_t7 = eeg_data[:,channel_mapping['T7_Low_beta'],:]+eeg_data[:,channel_mapping['T7_High_beta'],:]
        b_t8 = eeg_data[:,channel_mapping['T8_Low_beta'],:]+eeg_data[:,channel_mapping['T8_High_beta'],:]
        b_p7 = eeg_data[:,channel_mapping['P7_Low_beta'],:]+eeg_data[:,channel_mapping['P7_High_beta'],:]
        b_p8 = eeg_data[:,channel_mapping['P8_Low_beta'],:]+eeg_data[:,channel_mapping['P8_High_beta'],:]
    
        autonomic_eeg_features = np.stack([func1(a_f4,a_f3), #Approach-Withdrawal (AW) Index 1
                                                func2(a_f4,a_f3), #Approach-Withdrawal (AW) Index 2
                                                func3(b_f3,a_f3,b_f4,a_f4), #Valence Index 1
                                                func1(np.log(a_f3),np.log(a_f4)), #Valence Index 2
                                                func3(a_f4,b_f4,a_f3,b_f3), #Valence Index 3
                                                func3(b_f3+b_af3,a_f3+a_af3,b_f4+b_af4,a_f4+a_af4), #Valence Index 4
                                                func1(np.log(a_f3+a_af3),np.log(a_f4+a_af4)), #Valence Index 5
                                                func2(th_f4,th_f3), #Effort Index 1
                                                func2(np.log(g_af3),np.log(g_af4)), #Choice Index 1
                                                func2(np.log(b_af3),np.log(b_af4)) #Choice Index 2
                                               ]).T
        autonomic_eeg_feature_names = ['aw1','aw2','v1','v2','v3','v4','v5','e1','ch1','ch2']
        #custom T-channel based eeg features
        custom_t_channels_based_features = np.stack([func1(a_t7,a_t8),
                                                    func1(b_t7,b_t8),
                                                    func1(g_t7,g_t8),
                                                    func1(th_t7,th_t8),
                                                    func2(a_t7,a_t8),
                                                    func2(b_t7,b_t8),
                                                    func2(g_t7,g_t8),
                                                    func2(th_t7,th_t8)]).T
        t_channels_based_feature_names = ['T7/T8_alpha_diff1','T7/T8_beta_diff1','T7/T8_gamma_diff1','T7/T8_theta_diff1',
                                          'T7/T8_alpha_diff2','T7/T8_beta_diff2','T7/T8_gamma_diff2','T7/T8_theta_diff2']
        extracted_features = np.hstack([extracted_features,autonomic_eeg_features,custom_t_channels_based_features]) 
        feature_names = feature_names + autonomic_eeg_feature_names + t_channels_based_feature_names
    return extracted_features,feature_names