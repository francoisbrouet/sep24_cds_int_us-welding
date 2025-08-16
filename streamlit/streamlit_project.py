import streamlit as st
import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from lightgbm import LGBMRegressor
from scipy.interpolate import griddata
from myfunctions_streamlit import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# Draft-Notebook see 8-draft_streamlit.ipynb


# Load dataframes
base_dir = r'C:\FBr\Weiterbildung\Project\GitHub\sep24_cds_int_us-welding'
#base_dir = r'D:\Entwicklungen\share\DataScienceProject\sep24_cds_int_us-welding'

ft_param_path = os.path.join(base_dir, 'ft_files', '03_feat_parameters.ft')
ft_modes_path = os.path.join(base_dir, 'ft_files', '03_coll_modes.ft')
ft_nodes_path = os.path.join(base_dir, 'ft_files', '03_coll_nodes.ft')
ft_defs_path = os.path.join(base_dir, 'ft_files', '03_coll_defs.ft')
dump_dir = os.path.join(base_dir, 'model_dumps')

df_params = pd.read_feather (ft_param_path)
df_modes = pd.read_feather (ft_modes_path)
df_nodes = pd.read_feather (ft_nodes_path)
df_defs = pd.read_feather (ft_defs_path)

# Select the relevant columns of df_params only
lst_param = df_params.columns.tolist()
lst_param_geom = lst_param[1:24]
lst_param_sel = [lst_param[0]] + lst_param_geom + lst_param[24:26] + lst_param[29:32] + lst_param[57:61] + lst_param[137:139]
df_params_1 = df_params[lst_param_sel]

# Make displacement plots
fig_76 = plot_displacement_streamlit(df_modes, df_nodes, df_defs, 1001, 76, False)
fig_77 = plot_displacement_streamlit(df_modes, df_nodes, df_defs, 1001, 77, True)
fig_78 = plot_displacement_streamlit(df_modes, df_nodes, df_defs, 1001, 78, False)
fig_55 = plot_displacement_streamlit(df_modes, df_nodes, df_defs, 11429, 55, False)
fig_57 = plot_displacement_streamlit(df_modes, df_nodes, df_defs, 11429, 57, False)

# Load variables
lst_slot_class = lst_param[541:545]
lst_expl = lst_param[1:4] + lst_param[5:7] + lst_param[8:24] + lst_param[32:34]
target = 'freq_long'
X = df_params[lst_expl + lst_slot_class]
y = df_params[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=55)
df_params_filt = df_params.loc[df_params['mode_mac_long'] > 0.75]
X_filt = df_params_filt[lst_expl + lst_slot_class]
y_filt = df_params_filt[target]
X_train_filt, X_test_filt, y_train_filt, y_test_filt = train_test_split(X_filt, y_filt, test_size=0.2, random_state=55)

# Load models
pipe_base_lgbm = pickle.load(open(os.path.join(dump_dir, '4_model_lgbm_base_long_frequencies.pkl'), 'rb'))
pipe_tuned_lgbm = pickle.load(open(os.path.join(dump_dir, '4_model_lgbm_tuned_long_frequencies.pkl'), 'rb'))
pipe_filt_lgbm = pickle.load(open(os.path.join(dump_dir, '6_model_lgbm_filt_long_frequencies.pkl'), 'rb'))
score_base_lgbm = pipe_base_lgbm.score(X_test, y_test)
score_tuned_lgbm = pipe_tuned_lgbm.score(X_test, y_test)
score_filt_lgbm = pipe_filt_lgbm.score(X_test_filt, y_test_filt)
y_pred_base_lgbm = pipe_base_lgbm.predict(X_test)
y_pred_tuned_lgbm = pipe_tuned_lgbm.predict(X_test)
y_pred_filt_lgbm = pipe_filt_lgbm.predict(X_test_filt)
residuals_lgbm = y_pred_tuned_lgbm-y_test
residuals_lgbm_abs = np.abs(residuals_lgbm)
mae_base_lgbm = mean_absolute_error(y_test, y_pred_base_lgbm)
mae_tuned_lgbm = mean_absolute_error(y_test, y_pred_tuned_lgbm)
mae_filt_lgbm = mean_absolute_error(y_test_filt, y_pred_filt_lgbm)

# Make model plots
y_dict = {}
y_dict['test'] = y_test
y_dict['test_filt'] = y_test_filt
y_dict['pred_base'] = y_pred_base_lgbm
y_dict['pred_tuned'] = y_pred_tuned_lgbm
y_dict['pred_filt'] = y_pred_filt_lgbm
fig_base = plot_predictions_residuals(0, y_dict)
fig_tuned = plot_predictions_residuals(1, y_dict)
fig_filt = plot_predictions_residuals(2, y_dict)

# Plot Relationship MAC - Residuals
idx_test = X_test.index
mac_long_test = df_params.loc[idx_test, 'mode_mac_long']
fig_mac_res = plt.figure(figsize=(12, 6))
ax = fig_mac_res.add_subplot(1, 2, 1)
ax.scatter(mac_long_test, residuals_lgbm)
ax.plot([0.75, 0.75], [residuals_lgbm.min(), residuals_lgbm.max()], color='r')
ax.set_xlabel('MAC value')
ax.set_ylabel('Frequency residuals [Hz]')
ax.set_title('Relationship between the \n MAC values and the residuals')
ax.grid(True)
ax2 = fig_mac_res.add_subplot(1, 2, 2)
ax2.set_axis_off()



# ----------------------------------------------------------
st.title ('Ultrasonic welding')

st.sidebar.title ('Table of contents')
pages = ['Data Exploration', 'Data Visualization', 'Modelling of the frequencies']
page = st.sidebar.radio('Go to', pages)

if page == pages[0]:
    
    st.write('## Exploration of the data')
    
    st.write ('### DataFrame exploration')
    
    st.write ('DataFrame of parameters')
    st.dataframe(df_params_1.head(100), column_config={'dp_no':st.column_config.NumberColumn(format='%f')})
    st.write('Size:', df_params_1.shape)

    st.write ('DataFrame of modes')
    st.dataframe(df_modes.head(152), column_config={'dp_no':st.column_config.NumberColumn(format='%f')})
    st.write('Size:', df_modes.shape)
    
    st.write ('DataFrame of nodes')
    st.dataframe(df_nodes.head(495), column_config={'dp_no':st.column_config.NumberColumn(format='%f')})
    st.write('Size:', df_nodes.shape)
    
    st.write ('DataFrame of displacements')
    st.dataframe(df_defs.head(495), column_config={'dp_no':st.column_config.NumberColumn(format='%f')})
    st.write('Size:', df_defs.shape)


if page == pages[1]:

    st.write ('## Data visualization')
    
    st.write ('### Distribution of the explanatory variables')
    fig, ax = plt.subplots(figsize=(24, 12))
    plt.boxplot(df_params[lst_param_geom], labels=lst_param_geom)
    plt.xticks(rotation=90)
    plt.grid(True)
    st.pyplot(fig)

    # Example of the design point 1001
    st.write('### Example of Design Point #1001')

    st.write ('Geometrical parameters:')
    st.dataframe(df_params.loc[df_params['dp_no'] == 1001], column_config={'dp_no':st.column_config.NumberColumn(format='%f')})

    id_long = df_params.loc[df_params['dp_no'] == 1001, 'mode_no_long'].item()
    st.write ('Identified longitudinal mode: ' + str(id_long))

    lst_modes = range(id_long-3, id_long+4)
    st.dataframe(df_modes.loc[(df_modes['dp_no'] == 1001) & (df_modes['mode_no'].isin(lst_modes))], column_config={'dp_no':st.column_config.NumberColumn(format='%f')})
    
    display = st.radio('Show displacements of:',
        ['Mode 76', 'Mode 77', 'Mode 78'],
        captions=['Mode below', '**Longitudinal mode**', 'Mode above'],
        index=1)

    if display == 'Mode 76':
        st.pyplot(fig_76)
    if display == 'Mode 77':
        st.pyplot(fig_77)
    if display == 'Mode 78':
        st.pyplot(fig_78)


if page == pages[2] : 
    st.write('## Modelling the frequency of the longitudinal mode')

    choice = ['Base model', 'Tuned model', 'Model with filtered data']
    option = st.selectbox('Choice of the model', choice)    

    if option == choice[0]:
        st.write ('Score R² of the test set:', score_base_lgbm)
        st.write ('Mean Absolute Error:', mae_base_lgbm, 'Hz')        
        st.pyplot(fig_base)        
        
    if option == choice[1]:
        st.write ('Score R² of the test set:', score_tuned_lgbm)  
        st.write ('Mean Absolute Error:', mae_tuned_lgbm, 'Hz')         
        st.pyplot(fig_tuned)
    
    if option == choice[2]:
        st.write ('Score R² of the test set:', score_filt_lgbm)
        st.write ('Mean Absolute Error:', mae_filt_lgbm, 'Hz') 
        st.pyplot(fig_filt)
    
    if st.checkbox('Show the worst prediction') :
        # Worst prediction
        idx_res_max = residuals_lgbm_abs.idxmax()
        dp_no_res_max = df_params.loc[idx_res_max, 'dp_no']
        id_long = df_params.loc[idx_res_max, 'mode_no_long']

        st.write ('Design point with the worst prediction: DP #' + str(dp_no_res_max))
        st.write ('Max absolute error:', residuals_lgbm_abs.max(), 'Hz')
        st.write ('Number of the identified longitudinal mode:', id_long)    
    
        st.write ('List of modes of DP ' + str(dp_no_res_max))
        df_11429 = df_modes.loc[(df_modes['dp_no'] == dp_no_res_max) & ((df_modes['mode_no'] >= id_long-4) & (df_modes['mode_no'] < id_long+3)), ['dp_no', 'mode_no', 'freq', 'mode_mac']]
        st.dataframe(df_11429, column_config={'dp_no':st.column_config.NumberColumn(format='%f')})
        
        st.pyplot(fig_55)
        st.pyplot(fig_57)
        
        # display = st.radio('Show displacements of:',
            # ['Mode 55', 'Mode 57'],
            # captions=['Mode 55', 'Mode 57'],
            # index=0)
        # if display == 'Mode 55':
            # st.pyplot(fig_55)
        # if display == 'Mode 57':
            # st.pyplot(fig_57)
            
    if st.checkbox('Relationship MAC values - Frequency residuals'):
        st.pyplot(fig_mac_res)
        

# Frequency residuals = f(MAC value)



# (radio) best prediction, worst prediction
# Frequency, MAC values, displacements, frequencies neighbor modes



