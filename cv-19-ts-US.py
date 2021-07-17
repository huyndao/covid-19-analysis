#!/usr/bin/env python3
import numpy as np
import pandas as pd
import seaborn as sns
import argparse, re, gc, os
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt

pd.set_option('compute.use_numexpr', True)
pd.set_option('compute.use_bottleneck', True)

url_confirmed = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv'
url_death = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv'

csvname_confirmed = url_confirmed.split('/')[-1]
csvname_death = url_death.split('/')[-1]

ROOT_DIR = 'covid-19'

if not os.path.isdir(ROOT_DIR):
    os.makedirs(ROOT_DIR)

cv_confirmed = pd.read_table(url_confirmed, sep=',', dayfirst=False)
cv_death = pd.read_table(url_death, sep=',', dayfirst=False)

cv_confirmed.rename({'Province/State':'Province_State','Country/Region':'Country_Region'}, axis=1, inplace=True)
cv_death.rename({'Province/State':'Province_State','Country/Region':'Country_Region'}, axis=1, inplace=True)
 
# province_state level
cv_confirmed = cv_confirmed.groupby('Province_State').sum()
cv_death = cv_death.groupby('Province_State').sum()

cv_confirmed = cv_confirmed.reset_index()[~ cv_confirmed.reset_index()['Province_State'].isin(['Diamond Princess','Grand Princess'])].set_index('Province_State')
cv_death = cv_death.reset_index()[~ cv_death.reset_index()['Province_State'].isin(['Diamond Princess','Grand Princess'])].set_index('Province_State')

# drop columns: populations, UID, code3, FIPS, Lat, Long_
cv_confirmed = cv_confirmed.iloc[:,5:]
cv_death = cv_death.iloc[:,6:]
 
cv_confirmed.columns = pd.to_datetime(cv_confirmed.columns, utc=True, dayfirst=False)
cv_death.columns = pd.to_datetime(cv_death.columns, utc=True, dayfirst=False)

# filter by interested states
states = re.compile(r'(georgia,carolina,arizona,texas,new,cali)',re.I)
# cv_confirmed = cv_confirmed.filter(regex=states, axis=0)
# cv_death = cv_death.filter(regex=states, axis=0)
 
# calculate daily change and drop the initial column (which now contains NA's)
cv_confirmed_diff = cv_confirmed.diff(1, axis=1).dropna(axis=1)
cv_death_diff = cv_death.diff(1, axis=1).dropna(axis=1)
 
# there can't be daily decrease, ie. people can't come back from the dead or suddenly un-have covid
#cv_confirmed_diff = cv_confirmed_diff.stack().mask(cv_confirmed_diff.stack() < 0, 0).unstack()
#cv_death_diff = cv_death_diff.stack().mask(cv_death_diff.stack() < 0, 0).unstack()

# sort index by first non-zero cases
cv_confirmed_by_date = cv_confirmed_diff.mask(cv_confirmed_diff == 0).apply(pd.DataFrame.first_valid_index, axis=1).sort_values().reset_index()['Province_State'].values
cv_death_by_date = cv_death_diff.mask(cv_death_diff == 0).apply(pd.DataFrame.first_valid_index, axis=1).sort_values().reset_index()['Province_State'].values

# sort index by max cases
#cv_confirmed_by_max = cv_confirmed_diff.apply(np.argmax, axis=1).sort_values().reset_index()['Province_State'].values
#cv_death_by_max = cv_death_diff.apply(np.argmax, axis=1).sort_values().reset_index()['Province_State'].values
cv_confirmed_by_max = cv_confirmed_diff.idxmax(axis=1).sort_values().index.values
cv_death_by_max = cv_death_diff.idxmax(axis=1).sort_values().index.values

# 7 days rolling mean
cv_confirmed_diff_roll = cv_confirmed_diff.rolling(7, axis=1).mean().dropna(axis=1)
cv_death_diff_roll = cv_death_diff.rolling(7, axis=1).mean().dropna(axis=1)

# normalize
cv_confirmed_diff_norm = cv_confirmed_diff.apply(lambda x: x/x.max(), axis=1).fillna(0,axis=0)
cv_death_diff_norm = cv_death_diff.apply(lambda x: x/x.max(), axis=1).fillna(0,axis=0)
 
cv_confirmed_diff_norm = cv_confirmed_diff_norm.reindex(cv_confirmed_by_max)
cv_death_diff_norm = cv_death_diff_norm.reindex(cv_confirmed_by_max)

# 7 days rolling mean
cv_confirmed_diff_norm_roll = cv_confirmed_diff_norm.rolling(7, axis=1).mean().dropna(axis=1)
cv_death_diff_norm_roll = cv_death_diff_norm.rolling(7, axis=1).mean().dropna(axis=1)

#cv_confirmed_diff_norm_roll = cv_confirmed_norm_diff_roll.reindex(cv_confirmed_by_max)
#cv_death_diff_norm_roll = cv_death_diff_norm_roll.reindex(cv_confirmed_by_max)

# z-score standardization
zscore = lambda x: (x - np.mean(x))/(np.std(x) + 1e-10)

def corrlag(x, y):
    n = x.shape[0]
    corr = np.correlate(y/n, x, 'full')
    return np.max(corr), (np.argmax(corr) - (n - 1))

cv_confirmed_diff_z = cv_confirmed_diff.apply(zscore, raw=True, axis=1).fillna(0,axis=0)
cv_death_diff_z = cv_death_diff.apply(zscore, raw=True, axis=1).fillna(0,axis=0)

# 7 days rolling mean
cv_confirmed_diff_z_roll = cv_confirmed_diff_z.rolling(7, axis=1).mean().dropna(axis=1)
cv_death_diff_z_roll = cv_death_diff_z.rolling(7, axis=1).mean().dropna(axis=1)

confirmed_death_corr_lag = pd.DataFrame([corrlag(cv_confirmed_diff_z.loc[i,:], cv_death_diff_z.loc[i,:]) for i, _ in cv_confirmed_diff_z.iterrows()], columns=['correlation', 'lag'], index=cv_confirmed_diff_z.index)
#confirmed_death_corr_lag = pd.DataFrame([corrlag(cv_confirmed_diff_z_roll.loc[i,:], cv_death_diff_z_roll.loc[i,:]) for i, _ in cv_confirmed_diff_z_roll.iterrows()], columns=['correlation', 'lag'], index=cv_confirmed_diff_z_roll.index)

#confirmed_death_corr_lag = confirmed_death_corr_lag.assign(lag = confirmed_death_corr_lag.loc[:,'lag'].apply(pd.to_timedelta, unit='D'))

#cv_confirmed_diff_z_roll_by_max = cv_confirmed_diff_z_roll.apply(np.argmax, raw=True, axis=1).sort_values().reset_index()['Province_State'].values
#cv_death_diff_z_roll_by_max = cv_death_diff_z_roll.apply(np.argmax, raw=True, axis=1).sort_values().reset_index()['Province_State'].values
cv_confirmed_diff_z_roll_by_max = cv_confirmed_diff_z_roll.idxmax(axis=1).sort_values().index.values
cv_death_diff_z_roll_by_max = cv_death_diff_z_roll.idxmax(axis=1).sort_values().index.values

cv_confirmed_diff_z_roll_max = cv_confirmed_diff_z_roll.reindex(cv_confirmed_diff_z_roll_by_max)
cv_death_diff_z_roll_max = cv_death_diff_z_roll.reindex(cv_confirmed_diff_z_roll_by_max)

##################
## plot normalized
# 
#confirmed_cmap = sns.cubehelix_palette(n_colors=512,start=2,rot=1,gamma=1.05,as_cmap=True,reverse=True,hue=1,light=1)
#death_cmap = sns.cubehelix_palette(n_colors=512,start=3,rot=1,gamma=1.05,as_cmap=True,reverse=True,hue=1,light=1)
# 
#plt.close('all')
#fig, ax = plt.subplots(1,2,figsize=(18,15))
#sns.heatmap(data=cv_confirmed_diff_norm_roll, ax=ax[0],robust=True, vmin=0, vmax=1, cmap=confirmed_cmap)
#sns.heatmap(data=cv_death_diff_norm_roll, ax=ax[1],robust=True, vmin=0, vmax=1, cmap=confirmed_cmap)
#
#date_max = pd.to_datetime(cv_confirmed_diff_norm_roll.columns).max()
#date_min = pd.to_datetime(cv_confirmed_diff_norm_roll.columns).min()
# 
#x_dates = []
#x_dates.append(pd.date_range(start=date_min, end=date_max, periods=ax[0].get_xticks().size).date)
#x_dates.append(pd.date_range(start=date_min, end=date_max, periods=ax[1].get_xticks().size).date)
#fig.suptitle('COVID-19 US Cases\n(ordered by date of peak cases)', y=0.99)
#ax[0].set_title('Daily Confirmed Cases - Normalized (7 days rolling mean)')
#ax[1].set_title('Daily Deaths - Normalized (7 days rolling mean)')
#ax[0].set_xticklabels(x_dates[0])
#ax[1].set_xticklabels(x_dates[1])
# 
#ax[0].set_xlabel('')
#ax[1].set_xlabel('')
#ax[0].set_ylabel('')
#ax[1].set_ylabel('')
#
#plt.figtext(0.45,0.005,f'JHU CSSE COVID-19 Data\nhttps://github.com/CSSEGISandData/COVID-19', fontdict={'size':'x-small'})
#plt.figtext(0.93,0.005,f'Copyright (c) {datetime.date.today().strftime("%Y-%m-%d")}\nHuy Dao',fontdict={'size':'x-small'})
#fig.set_tight_layout(True)
#fig.savefig(os.path.join(ROOT_DIR,f'covid-19-US-normalized-rolling-max.png'), dpi='figure', format='png')
##plt.tight_layout()
##plt.show()
 
#################
# plot raw
 
confirmed_cmap = sns.cubehelix_palette(n_colors=512,start=1.5,rot=1,gamma=1.0,as_cmap=True,reverse=True,hue=1,light=1)
death_cmap = sns.cubehelix_palette(n_colors=512,start=2.5,rot=1,gamma=1.0,as_cmap=True,reverse=True,hue=1,light=1)
 
plt.close('all')
fig, ax = plt.subplots(1,2,figsize=(18,15))
sns.heatmap(data=cv_confirmed_diff_roll, ax=ax[0],robust=True, vmin=0, vmax=np.ceil(cv_confirmed_diff_roll.max().max()), cmap=confirmed_cmap)
sns.heatmap(data=cv_death_diff_roll, ax=ax[1],robust=True, vmin=0, vmax=np.ceil(cv_death_diff_roll.max().max()), cmap=confirmed_cmap)
 
date_max = pd.to_datetime(cv_confirmed_diff_roll.columns).max()
date_min = pd.to_datetime(cv_confirmed_diff_roll.columns).min()
 
x_dates = []
x_dates.append(pd.date_range(start=date_min, end=date_max, periods=ax[0].get_xticks().size).date)
x_dates.append(pd.date_range(start=date_min, end=date_max, periods=ax[1].get_xticks().size).date)
fig.suptitle('COVID-19 US Cases\n(ordered by date of peak cases)', y=0.99)
ax[0].set_title('Daily Confirmed Cases - Raw (7 days rolling mean)')
ax[1].set_title('Daily Deaths - Raw (7 days rolling mean)')
ax[0].set_xticklabels(x_dates[0])
ax[1].set_xticklabels(x_dates[1])
 
ax[0].set_xlabel('')
ax[1].set_xlabel('')
ax[0].set_ylabel('')
ax[1].set_ylabel('')

plt.figtext(0.45,0.005,f'JHU CSSE COVID-19 Data\nhttps://github.com/CSSEGISandData/COVID-19', fontdict={'size':'x-small'})
plt.figtext(0.93,0.005,f'Copyright (c) {datetime.date.today().strftime("%Y-%m-%d")}\nHuy Dao',fontdict={'size':'x-small'})
fig.set_tight_layout(True)
fig.savefig(os.path.join(ROOT_DIR,f'covid-19-US-raw-rolling.png'), dpi='figure', format='png')
#plt.tight_layout()
#plt.show()

#################
# z score normalized

plt.close('all')
fig, ax = plt.subplots(1,2,figsize=(18,15))
sns.heatmap(cv_confirmed_diff_z_roll_max, cmap='RdBu_r', center=0, robust=True, ax=ax[0])
sns.heatmap(cv_death_diff_z_roll_max, cmap='RdBu_r', center=0, robust=True, ax=ax[1])

date_max = pd.to_datetime(cv_confirmed_diff_z_roll_max.columns).max()
date_min = pd.to_datetime(cv_confirmed_diff_z_roll_max.columns).min()

x_dates = []
x_dates.append(pd.date_range(start=date_min, end=date_max, periods=ax[0].get_xticks().size).date)
x_dates.append(pd.date_range(start=date_min, end=date_max, periods=ax[1].get_xticks().size).date)
fig.suptitle('COVID-19 US Cases\n(ordered by date of peak cases)', y=0.99)
ax[0].set_title('Daily Confirmed Cases - Z Score (7 days rolling mean)')
ax[1].set_title('Daily Deaths - Z Score (7 days rolling mean)')
ax[0].set_xticklabels(x_dates[0])
ax[1].set_xticklabels(x_dates[1])

ax[0].set_xlabel('')
ax[1].set_xlabel('')
ax[0].set_ylabel('')
ax[1].set_ylabel('')

plt.figtext(0.45,0.005,f'JHU CSSE COVID-19 Data\nhttps://github.com/CSSEGISandData/COVID-19', fontdict={'size':'x-small'})
plt.figtext(0.93,0.005,f'Copyright (c) {datetime.date.today().strftime("%Y-%m-%d")}\nHuy Dao',fontdict={'size':'x-small'})
fig.set_tight_layout(True)
fig.savefig(os.path.join(ROOT_DIR,f'covid-19-US-z-score-rolling-max.png'), dpi='figure', format='png')
#fig.savefig(os.path.join(ROOT_DIR,f'covid-19-US-zscore-{datetime.date.today().strftime("%Y-%m-%d")}.png'), dpi='figure', format='png')
#plt.tight_layout()
#plt.show()
 
#################
plt.close('all')
fig, axes = plt.subplots(cv_confirmed_diff_z_roll_max.shape[0], 1, figsize=(18,150), sharex=False)
date_max = cv_confirmed_diff_z_roll_max.columns.max()
date_min = cv_confirmed_diff_z_roll_max.columns.min()

confirmed_death_corr_lag_max = confirmed_death_corr_lag.reindex(cv_confirmed_diff_z_roll_by_max)

for i, ax in enumerate(axes):
#    ax.plot(cv_confirmed_diff_z_roll_max.iloc[i], color='b', label='cases', aa=True)
    cv_confirmed_diff_z_roll_max.iloc[i].plot(ax=ax, color='b', label='cases', aa=True, use_index=True)
#    ax.plot(cv_death_diff_z_roll_max.iloc[i], color='r', label='deaths', aa=True)
    cv_death_diff_z_roll_max.iloc[i].plot(ax=ax, color='r', label='deaths', aa=True, use_index=True)

    ax.set_xticks(np.linspace(ax.get_xticks().min(), ax.get_xticks().max(), 15), minor=False)
    x_dates = pd.date_range(start=date_min, end=date_max, periods=ax.get_xticks().size).date
    ax.set_xticklabels(x_dates, fontsize='x-small', minor=False)
    ax.set_yticks([-6,-4,-2,0,2,4,6])
    ax.set_xlim((ax.get_xticks().min()-20, ax.get_xticks().max()+14))

    ax.axvline(cv_confirmed_diff_z_roll_max.iloc[i].idxmax(), color='b',linestyle='--', linewidth=2, label='max cases')

    if cv_death_diff_z_roll_max.iloc[i].max() > 0:
        ax.axvline(cv_death_diff_z_roll_max.iloc[i].idxmax(), color='r',linestyle='--', linewidth=2, label='max deaths')

    if (confirmed_death_corr_lag_max.iloc[i,0] > 0) and ((cv_confirmed_diff_z_roll_max.iloc[i].idxmax() + pd.to_timedelta(confirmed_death_corr_lag_max.iloc[i,1], 'd')) >= date_min):
        ax.axvline(cv_confirmed_diff_z_roll_max.iloc[i].idxmax() + pd.to_timedelta(confirmed_death_corr_lag_max.iloc[i,1], 'd'), color='k',linestyle=':', linewidth=2, label='max cases + x-corr lag')

    ax.set_title(f'{cv_confirmed_diff_z_roll_max.iloc[i].name}\ncases mean = {cv_confirmed_diff.loc[cv_confirmed_diff_z_roll_max.iloc[i].name].mean():.2f}; std dev = {cv_confirmed_diff.loc[cv_confirmed_diff_z_roll_max.iloc[i].name].std():.2f}; total = {cv_confirmed.loc[cv_confirmed_diff_z_roll_max.iloc[i].name][-1]}; peak = {cv_confirmed_diff.loc[cv_confirmed_diff_z_roll_max.iloc[i].name].max():.0f}\ndeaths mean = {cv_death_diff.loc[cv_confirmed_diff_z_roll_max.iloc[i].name].mean():.2f}; std dev = {cv_death_diff.loc[cv_confirmed_diff_z_roll_max.iloc[i].name].std():.2f}; total = {cv_death.loc[cv_confirmed_diff_z_roll_max.iloc[i].name][-1]}; peak = {cv_death_diff.loc[cv_confirmed_diff_z_roll_max.iloc[i].name].max():.0f}\npeak cases to peak deaths time = {(cv_death_diff_z_roll_max.iloc[i].idxmax() - cv_confirmed_diff_z_roll_max.iloc[i].idxmax()).days} days; x-corr lag = {confirmed_death_corr_lag_max.iloc[i,1]} days; x-corr coef = {confirmed_death_corr_lag_max.iloc[i,0]:.2f}', loc='left', fontdict={'fontsize':'medium','fontweight':'bold'})

    ax.set_ylabel('Z Score', fontdict={'fontsize':'small'})
    ax.legend(loc='best', fontsize='x-small')
    ax.grid(b=True, which='major', axis='both', lw=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle('COVID-19 US 7 days rolling mean - Confirmed Cases - Death lag cross-correlation ', y=0.9995)
plt.figtext(0.75,0.997,f'JHU CSSE COVID-19 Data\nhttps://github.com/CSSEGISandData/COVID-19', fontdict={'size':'x-small'})
plt.figtext(0.92,0.997,f'Copyright (c) {datetime.date.today().strftime("%Y-%m-%d")}\nHuy Dao', fontdict={'size':'x-small'})
fig.set_tight_layout(True)
fig.savefig(os.path.join(ROOT_DIR,f'covid-19-US-z-score-running-mean-ordered-by-max-confirmed-death-lag.png'), dpi='figure', format='png')

#################
plt.close('all')

min_corr = 0.6

g = sns.FacetGrid(despine=False, height=8, aspect=1.5, data=confirmed_death_corr_lag.loc[confirmed_death_corr_lag.loc[:,'correlation'] >= min_corr])
g.map(sns.histplot, 'lag', stat='probability', kde=True, color='teal')
g.set_axis_labels('lag (day)', 'probability')
g.ax.grid(b=True, which='major', axis='both', lw=0.5)
#g.ax.set_xlim((-10,30))
#g.ax.set_xticks(np.arange(-10,30,5))
#g.ax.set_xticklabels(np.arange(-10,30,5))
g.fig.text(0.60,0.999,f'JHU CSSE COVID-19 Data\nhttps://github.com/CSSEGISandData/COVID-19', fontdict={'size':'x-small'})
g.fig.text(0.92,0.999,f'Copyright (c) {datetime.date.today().strftime("%Y-%m-%d")}\nHuy Dao', fontdict={'size':'x-small'})
g.fig.suptitle(f'probability distribution of US confirmed cases to deaths lag, with minimum correlation coefficient of {min_corr}')
g.tight_layout()
g.savefig(os.path.join(ROOT_DIR,f'covid-19-US-probability-distribution-of-confirmed-cases-to-deaths-lag.png'), dpi='figure', format='png')


