#!/usr/bin/env python3
# Copyright © 2020, Huy Dao

import numpy as np
import pandas as pd
import seaborn as sns
import argparse, re, gc, os
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt

pd.set_option('compute.use_numexpr', True)
pd.set_option('compute.use_bottleneck', True)

url_confirmed = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
url_death = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'

csvname_confirmed = url_confirmed.split('/')[-1]
csvname_death = url_death.split('/')[-1]

ROOT_DIR = 'covid-19'

if not os.path.isdir(ROOT_DIR):
    os.makedirs(ROOT_DIR)

cv_confirmed = pd.read_table(url_confirmed, sep=',', dayfirst=False)
cv_death = pd.read_table(url_death, sep=',', dayfirst=False)

cv_confirmed.rename({'Province/State':'Province_State','Country/Region':'Country_Region'}, axis=1, inplace=True)
cv_death.rename({'Province/State':'Province_State','Country/Region':'Country_Region'}, axis=1, inplace=True)

# country level instead of province level
cv_confirmed = cv_confirmed.groupby('Country_Region').sum()
cv_death = cv_death.groupby('Country_Region').sum()

# drop Province_State, Lat, Long columns
cv_confirmed = cv_confirmed.iloc[:,2:]
cv_death = cv_death.iloc[:,2:]

cv_confirmed.columns = pd.to_datetime(cv_confirmed.columns, utc=True, dayfirst=False)
cv_death.columns = pd.to_datetime(cv_death.columns, utc=True, dayfirst=False)

cv_confirmed = cv_confirmed.reset_index().replace({'Taiwan*':'Taiwan', 'West Bank and Gaza':'Palestine'})
cv_confirmed.set_index('Country_Region', inplace=True)
cv_confirmed.sort_index(inplace=True)

cv_death = cv_death.reset_index().replace({'Taiwan*':'Taiwan', 'West Bank and Gaza':'Palestine'})
cv_death.set_index('Country_Region', inplace=True)
cv_death.sort_index(inplace=True)

# filter by interested countries
#countries = re.compile(r'(^us$|brazil|china|india|japan|vietnam|korea|russia|sweden|france|zealand|switzerland|german|united king|iran|saudi|turkey|ireland|singapore|taiwan|australia|spain|italy|canada|mexico)',re.I)
#cv_confirmed = cv_confirmed.filter(regex=countries, axis=0)
#cv_death = cv_death.filter(regex=countries, axis=0)

# calculate daily change and drop the initial column (which now contains NA's)
cv_confirmed_diff = cv_confirmed.diff(1, axis=1).dropna(axis=1)
cv_death_diff = cv_death.diff(1, axis=1).dropna(axis=1)

# there can't be daily decrease, ie people can't come back from the dead or suddenly un-have covid the next day
cv_confirmed_diff = cv_confirmed_diff.stack().mask(cv_confirmed_diff.stack() < 0, 0).unstack()
cv_death_diff = cv_death_diff.stack().mask(cv_death_diff.stack() < 0, 0).unstack()

# sort index by first non-zero cases
cv_confirmed_by_date = cv_confirmed_diff.mask(cv_confirmed_diff == 0).apply(pd.DataFrame.first_valid_index, axis=1).sort_values().reset_index()['Country_Region'].values
cv_death_by_date = cv_death_diff.mask(cv_death_diff == 0).apply(pd.DataFrame.first_valid_index, axis=1).sort_values().reset_index()['Country_Region'].values

# sort index by max cases
#cv_confirmed_by_max = cv_confirmed_diff.apply(np.argmax, axis=1).sort_values().reset_index()['Country_Region'].values
#cv_death_by_max = cv_death_diff.apply(np.argmax, axis=1).sort_values().reset_index()['Country_Region'].values
cv_confirmed_by_max = cv_confirmed_diff.idxmax(axis=1).sort_values().index.values
cv_death_by_max = cv_death_diff.idxmax(axis=1).sort_values().index.values

# min-max normalization
cv_confirmed_diff_norm = cv_confirmed_diff.apply(lambda x: x/x.max(), axis=1).fillna(0,axis=0)
cv_death_diff_norm = cv_death_diff.apply(lambda x: x/x.max(), axis=1).fillna(0,axis=0)

#cv_confirmed_diff_norm = cv_confirmed_diff_norm.reindex(cv_confirmed_by_max)
#cv_death_diff_norm = cv_death_diff_norm.reindex(cv_confirmed_by_max)

# 7 days rolling mean
cv_confirmed_diff_roll = cv_confirmed_diff.rolling(7, axis=1).mean().dropna(axis=1)
cv_death_diff_roll = cv_death_diff.rolling(7, axis=1).mean().dropna(axis=1)

#cv_confirmed_diff_roll = cv_confirmed_diff_roll.reindex(cv_confirmed_by_max)
#cv_death_diff_roll = cv_death_diff_roll.reindex(cv_confirmed_by_max)

# z score standardization
zscore = lambda x: (x - np.mean(x))/(np.std(x) + 1e-10)

cv_confirmed_diff_z = cv_confirmed_diff.apply(zscore, raw=True, axis=1).fillna(0,axis=0)
cv_death_diff_z = cv_death_diff.apply(zscore, raw=True, axis=1).fillna(0,axis=0)

# cross correlation function, returns correlation and lag
def corrlag(x, y):
    n = x.shape[0]
    corr = np.correlate(y/n, x, 'full')
    return np.max(corr), (np.argmax(corr) - (n - 1))

## prepare index list for country to country cross-correlation
#left_arr = np.empty(0, dtype=np.object)
#right_arr = np.empty(0, dtype=np.object)
#
#for i in np.arange(cv_confirmed_diff_z.index.shape[0]):
#    left_index = cv_confirmed_diff_z.index.values
#    right_index = np.roll(cv_confirmed_diff_z.index, i, 0)
#    left_arr = np.r_[left_arr, left_index]
#    right_arr = np.r_[right_arr, right_index]
#
#c2c_confirmed_corr_lag = pd.DataFrame(columns=['correlation', 'lag'], index=(left_arr + ' - ' + right_arr))
#c2c_death_corr_lag = pd.DataFrame(columns=['correlation', 'lag'], index=(left_arr + ' - ' + right_arr))
#
#left_confirmed_df = cv_confirmed_diff_z.reindex(left_arr)
#right_confirmed_df = cv_confirmed_diff_z.reindex(right_arr)
#left_death_df = cv_death_diff_z.reindex(left_arr)
#right_death_df = cv_death_diff_z.reindex(right_arr)
#
#for i in np.arange(left_arr.size):
#    c2c_confirmed_corr_lag.iloc[i,:] = corrlag(left_confirmed_df.iloc[i,:], right_confirmed_df.iloc[i,:])
#    c2c_death_corr_lag.iloc[i,:] = corrlag(left_death_df.iloc[i,:], right_death_df.iloc[i,:])
#
#c2c_confirmed_corr_lag = pd.concat([pd.DataFrame(list(c2c_confirmed_corr_lag.index.str.split(' - ')), columns=['c1','c2'], index=c2c_confirmed_corr_lag.index), c2c_confirmed_corr_lag], axis=1).reset_index().drop(columns='index')
#
#c2c_death_corr_lag = pd.concat([pd.DataFrame(list(c2c_death_corr_lag.index.str.split(' - ')), columns=['c1','c2'], index=c2c_death_corr_lag.index), c2c_death_corr_lag], axis=1).reset_index().drop(columns='index')
#
#c2c_confirmed_corr_lag = c2c_confirmed_corr_lag.astype({'correlation':'float64', 'lag':'int64'})
#c2c_death_corr_lag = c2c_death_corr_lag.astype({'correlation':'float64', 'lag':'int64'})
#
##c2c_confirmed_corr_lag.query('c1.str.contains("US")', engine='python').sort_values(by=['correlation'])

# 7 days rolling mean
cv_confirmed_diff_z_roll = cv_confirmed_diff_z.rolling(7, axis=1).mean().dropna(axis=1)
cv_death_diff_z_roll = cv_death_diff_z.rolling(7, axis=1).mean().dropna(axis=1)

# confirmed cases versus death cross-correlation -> the below return DataFrame confirmed_death_corr_lag, which has 2 columns: correlation and lag
#confirmed_death_corr_lag = pd.DataFrame([corrlag(cv_confirmed_diff_z_roll.loc[i,:], cv_death_diff_z_roll.loc[i,:]) for i, _ in cv_confirmed_diff_z_roll.iterrows()], columns=['correlation', 'lag'], index=cv_confirmed_diff_z_roll.index)
confirmed_death_corr_lag = pd.DataFrame([corrlag(cv_confirmed_diff_z.loc[i,:], cv_death_diff_z.loc[i,:]) for i, _ in cv_confirmed_diff_z.iterrows()], columns=['correlation', 'lag'], index=cv_confirmed_diff_z.index)

#confirmed_death_corr_lag = confirmed_death_corr_lag.assign(lag = confirmed_death_corr_lag.loc[:,'lag'].apply(pd.to_timedelta, unit='D'))

# by first non-zero date
cv_confirmed_diff_z_roll_by_date = cv_confirmed_diff_z_roll.mask(cv_confirmed_diff_z_roll == 0).apply(pd.DataFrame.first_valid_index, axis=1).sort_values().reset_index()['Country_Region'].values
cv_death_diff_z_roll_by_date = cv_death_diff_z_roll.mask(cv_death_diff_z_roll == 0).apply(pd.DataFrame.first_valid_index, axis=1).sort_values().reset_index()['Country_Region'].values

# by max
#cv_confirmed_diff_z_roll_by_max = cv_confirmed_diff_z_roll.apply(np.argmax, raw=True, axis=1).sort_values().reset_index()['Country_Region'].values
#cv_death_diff_z_roll_by_max = cv_death_diff_z_roll.apply(np.argmax, raw=True, axis=1).sort_values().reset_index()['Country_Region'].values
cv_confirmed_diff_z_roll_by_max = cv_confirmed_diff_z_roll.idxmax(axis=1).sort_values().index.values
cv_death_diff_z_roll_by_max = cv_death_diff_z_roll.idxmax(axis=1).sort_values().index.values

cv_confirmed_diff_z_roll_max = cv_confirmed_diff_z_roll.reindex(cv_confirmed_diff_z_roll_by_max)
cv_death_diff_z_roll_max = cv_death_diff_z_roll.reindex(cv_confirmed_diff_z_roll_by_max)

cv_confirmed_diff_z_by_max = cv_confirmed_diff_z.idxmax(axis=1).sort_values().index.values
cv_death_diff_z_by_max = cv_death_diff_z.idxmax(axis=1).sort_values().index.values

cv_confirmed_diff_z_max = cv_confirmed_diff_z.reindex(cv_confirmed_diff_z_roll_by_max)
cv_death_diff_z_max = cv_death_diff_z.reindex(cv_confirmed_diff_z_roll_by_max)

##################
## plot min-max normalized
#
#confirmed_cmap = sns.cubehelix_palette(n_colors=512,start=2,rot=1,gamma=1.05,as_cmap=True,reverse=True,hue=1,light=1)
#death_cmap = sns.cubehelix_palette(n_colors=512,start=3,rot=1,gamma=1.05,as_cmap=True,reverse=True,hue=1,light=1)
#
#plt.close('all')
#fig, ax = plt.subplots(1,2,figsize=(18,36))
#sns.heatmap(data=cv_confirmed_diff_roll_norm, ax=ax[1],robust=True, vmin=0, vmax=1, cmap=confirmed_cmap)
#sns.heatmap(data=cv_death_diff_roll_norm, ax=ax[1],robust=True, vmin=0, vmax=1, cmap=confirmed_cmap)
#
#date_max = cv_confirmed_diff_roll_norm.columns.max()
#date_min = cv_confirmed_diff_roll_norm.columns.min()
#
#x_dates = []
#x_dates.append(pd.date_range(start=date_min, end=date_max, periods=ax[0].get_xticks().size).date)
#x_dates.append(pd.date_range(start=date_min, end=date_max, periods=ax[1].get_xticks().size).date)
#fig.suptitle('COVID-19 Global Cases\n(ordered by date of peak cases)', y=0.99)
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
#
#plt.figtext(0.45,0.005,f'JHU CSSE COVID-19 Data\nhttps://github.com/CSSEGISandData/COVID-19', fontdict={'size':'x-small'})
#
#plt.figtext(0.92,0.005,f'Copyright (c) {datetime.date.today().strftime("%Y-%m-%d")}\nHuy Dao',fontdict={'size':'x-small'})
#fig.set_tight_layout(True)
#fig.savefig(os.path.join(ROOT_DIR,f'covid-19-global-normalized-{datetime.date.today().strftime("%Y-%m-%d")}.png'), dpi='figure', format='png')
##plt.tight_layout()
##plt.show()
#
#################
# plot raw

confirmed_cmap = sns.cubehelix_palette(n_colors=512,start=1.5,rot=1,gamma=1.0,as_cmap=True,reverse=True,hue=1,light=1)
death_cmap = sns.cubehelix_palette(n_colors=512,start=2.5,rot=1,gamma=1.0,as_cmap=True,reverse=True,hue=1,light=1)

plt.close('all')
fig, ax = plt.subplots(1,2,figsize=(18,36))
sns.heatmap(data=cv_confirmed_diff, ax=ax[0],robust=True, vmin=0, vmax=np.ceil(cv_confirmed_diff.max().max()), cmap=confirmed_cmap)
sns.heatmap(data=cv_death_diff, ax=ax[1],robust=True, vmin=0, vmax=np.ceil(cv_death_diff.max().max()), cmap=confirmed_cmap)

date_max = cv_confirmed_diff.columns.max()
date_min = cv_confirmed_diff.columns.min()

x_dates = []
x_dates.append(pd.date_range(start=date_min, end=date_max, periods=ax[0].get_xticks().size).date)
x_dates.append(pd.date_range(start=date_min, end=date_max, periods=ax[1].get_xticks().size).date)
fig.suptitle('COVID-18 Global Cases', y=0.99)
ax[0].set_title('Daily Confirmed Cases - Raw')
ax[1].set_title('Daily Deaths - Raw')
ax[0].set_xticklabels(x_dates[0])
ax[1].set_xticklabels(x_dates[1])

ax[0].set_xlabel('')
ax[1].set_xlabel('')
ax[0].set_ylabel('')
ax[1].set_ylabel('')

plt.figtext(0.45,0.005,f'JHU CSSE COVID-19 Data\nhttps://github.com/CSSEGISandData/COVID-19', fontdict={'size':'x-small'})

plt.figtext(0.92,0.005,f'Copyright (c) {datetime.date.today().strftime("%Y-%m-%d")}\nHuy Dao',fontdict={'size':'x-small'})
fig.set_tight_layout(True)
fig.savefig(os.path.join(ROOT_DIR,f'covid-19-global-raw.png'), dpi='figure', format='png')
#plt.tight_layout()
#plt.show()

##################
## plot raw + rolling
#
#confirmed_cmap = sns.cubehelix_palette(n_colors=512,start=1.5,rot=1,gamma=1.0,as_cmap=True,reverse=True,hue=1,light=1)
#death_cmap = sns.cubehelix_palette(n_colors=512,start=2.5,rot=1,gamma=1.0,as_cmap=True,reverse=True,hue=1,light=1)
#
#plt.close('all')
#fig, ax = plt.subplots(1,2,figsize=(18,36))
#sns.heatmap(data=cv_confirmed_diff_roll, ax=ax[0],robust=True, vmin=0, vmax=np.ceil(cv_confirmed_diff_roll.max().max()), cmap=confirmed_cmap)
#sns.heatmap(data=cv_death_diff_roll, ax=ax[1],robust=True, vmin=0, vmax=np.ceil(cv_death_diff_roll.max().max()), cmap=confirmed_cmap)
#
#date_max = cv_confirmed_diff_roll.columns.max()
#date_min = cv_confirmed_diff_roll.columns.min()
#
#x_dates = []
#x_dates.append(pd.date_range(start=date_min, end=date_max, periods=ax[0].get_xticks().size).date)
#x_dates.append(pd.date_range(start=date_min, end=date_max, periods=ax[1].get_xticks().size).date)
#fig.suptitle('COVID-19 Global Cases\n(ordered by date of peak cases)', y=0.99)
#ax[0].set_title('Daily Confirmed Cases - Raw (7 days rolling mean)')
#ax[1].set_title('Daily Deaths - Raw (7 days rolling mean)')
#ax[0].set_xticklabels(x_dates[0])
#ax[1].set_xticklabels(x_dates[1])
#
#ax[0].set_xlabel('')
#ax[1].set_xlabel('')
#ax[0].set_ylabel('')
#ax[1].set_ylabel('')
#
#plt.figtext(0.45,0.005,f'JHU CSSE COVID-19 Data\nhttps://github.com/CSSEGISandData/COVID-19', fontdict={'size':'x-small'})
#
#plt.figtext(0.92,0.005,f'Copyright (c) {datetime.date.today().strftime("%Y-%m-%d")}\nHuy Dao',fontdict={'size':'x-small'})
#fig.set_tight_layout(True)
#fig.savefig(os.path.join(ROOT_DIR,f'covid-19-global-raw-{datetime.date.today().strftime("%Y-%m-%d")}.png'), dpi='figure', format='png')
##plt.tight_layout()
##plt.show()

#################
# z score normalized

plt.close('all')
fig, ax = plt.subplots(1,2,figsize=(18,36))

sns.heatmap(cv_confirmed_diff_z_roll, cmap='RdBu_r', center=0, robust=True, ax=ax[0])
sns.heatmap(cv_death_diff_z_roll, cmap='RdBu_r', center=0, robust=True, ax=ax[1])

date_max = cv_confirmed_diff_z_roll.columns.max()
date_min = cv_confirmed_diff_z_roll.columns.min()

#sns.heatmap(cv_confirmed_diff_z, cmap='RdBu_r', center=0, robust=True, ax=ax[0])
#sns.heatmap(cv_death_diff_z, cmap='RdBu_r', center=0, robust=True, ax=ax[1])
#
#date_max = cv_confirmed_diff_z.columns.max()
#date_min = cv_confirmed_diff_z.columns.min()

x_dates = []
x_dates.append(pd.date_range(start=date_min, end=date_max, periods=ax[0].get_xticks().size).date)
x_dates.append(pd.date_range(start=date_min, end=date_max, periods=ax[1].get_xticks().size).date)
fig.suptitle('COVID-19 Global Cases', y=0.99)
ax[0].set_title('Daily Confirmed - Z Score (7 days rolling mean)')
ax[1].set_title('Daily Deaths - Z Score (7 days rolling mean)')
ax[0].set_xticklabels(x_dates[0])
ax[1].set_xticklabels(x_dates[1])

ax[0].set_xlabel('')
ax[1].set_xlabel('')
ax[0].set_ylabel('')
ax[1].set_ylabel('')

plt.figtext(0.45,0.005,f'JHU CSSE COVID-19 Data\nhttps://github.com/CSSEGISandData/COVID-19', fontdict={'size':'x-small'})

plt.figtext(0.92,0.005,f'Copyright (c) {datetime.date.today().strftime("%Y-%m-%d")}\nHuy Dao',fontdict={'size':'x-small'})
fig.set_tight_layout(True)
fig.savefig(os.path.join(ROOT_DIR,f'covid-19-global-z-score-running-mean.png'), dpi='figure', format='png')
#plt.tight_layout()
#plt.show()

#################
# z score normalized by max

plt.close('all')
fig, ax = plt.subplots(1,2,figsize=(18,36))

sns.heatmap(cv_confirmed_diff_z_roll_max, cmap='RdBu_r', center=0, robust=True, ax=ax[0])
sns.heatmap(cv_death_diff_z_roll_max, cmap='RdBu_r', center=0, robust=True, ax=ax[1])

date_max = cv_confirmed_diff_z_roll_max.columns.max()
date_min = cv_confirmed_diff_z_roll_max.columns.min()

#sns.heatmap(cv_confirmed_diff_z, cmap='RdBu_r', center=0, robust=True, ax=ax[0])
#sns.heatmap(cv_death_diff_z, cmap='RdBu_r', center=0, robust=True, ax=ax[1])
#
#date_max = cv_confirmed_diff_z.columns.max()
#date_min = cv_confirmed_diff_z.columns.min()

x_dates = []
x_dates.append(pd.date_range(start=date_min, end=date_max, periods=ax[0].get_xticks().size).date)
x_dates.append(pd.date_range(start=date_min, end=date_max, periods=ax[1].get_xticks().size).date)
fig.suptitle('COVID-19 Global Cases\n(ordered by date of peak cases)', y=0.99)
ax[0].set_title('Daily Confirmed - Z Score (7 days rolling mean)')
ax[1].set_title('Daily Deaths - Z Score (7 days rolling mean)')
ax[0].set_xticklabels(x_dates[0])
ax[1].set_xticklabels(x_dates[1])

ax[0].set_xlabel('')
ax[1].set_xlabel('')
ax[0].set_ylabel('')
ax[1].set_ylabel('')

plt.figtext(0.45,0.005,f'JHU CSSE COVID-19 Data\nhttps://github.com/CSSEGISandData/COVID-19', fontdict={'size':'x-small'})

plt.figtext(0.92,0.005,f'Copyright (c) {datetime.date.today().strftime("%Y-%m-%d")}\nHuy Dao', fontdict={'size':'x-small'})
fig.set_tight_layout(True)
#fig.savefig(os.path.join(ROOT_DIR,f'covid-19-global-zscore-{datetime.date.today().strftime("%Y-%m-%d")}.png'), dpi='figure', format='png')
fig.savefig(os.path.join(ROOT_DIR,f'covid-19-global-z-score-running-mean-ordered-by-max.png'), dpi='figure', format='png')
#plt.tight_layout()
#plt.show()

#################
plt.close('all')
fig, axes = plt.subplots(cv_confirmed_diff_z_roll_max.shape[0], 1, figsize=(18,325), sharex=False)
date_max = cv_confirmed_diff_z_roll_max.columns.max()
date_min = cv_confirmed_diff_z_roll_max.columns.min()

confirmed_death_corr_lag_max = confirmed_death_corr_lag.reindex(cv_confirmed_diff_z_roll_by_max)

for i, ax in enumerate(axes):
#    ax.plot(cv_confirmed_diff_z_roll_max.iloc[i], color='b', label='cases', aa=True)
#    ax.plot(cv_death_diff_z_roll_max.iloc[i], color='r', label='deaths', aa=True)
    cv_confirmed_diff_z_roll_max.iloc[i].plot(ax=ax, color='b', label='cases', aa=True)
    cv_death_diff_z_roll_max.iloc[i].plot(ax=ax, color='r', label='deaths', aa=True)

    ax.set_xticks(np.linspace(ax.get_xticks().min(), ax.get_xticks().max(), 15), minor=False)
    x_dates = list(pd.date_range(start=date_min, end=date_max, periods=ax.get_xticks().size).date)
    ax.set_xticklabels(x_dates, fontsize='x-small', minor=False)
    ax.set_yticks([-2,0,2,4,6])
    ax.set_xlim((ax.get_xticks().min()-40, ax.get_xticks().max()+14))

    ax.axvline(cv_confirmed_diff_z_roll_max.iloc[i].idxmax(), color='b',linestyle='--', linewidth=2, aa=True, label='max cases')

    if cv_death_diff_z_roll_max.iloc[i].max() > 0:
        ax.axvline(cv_death_diff_z_roll_max.iloc[i].idxmax(), color='r',linestyle='--', linewidth=2, aa=True, label='max deaths')

    if (confirmed_death_corr_lag_max.iloc[i,0] > 0) and ((cv_confirmed_diff_z_roll_max.iloc[i].idxmax() + pd.to_timedelta(confirmed_death_corr_lag_max.iloc[i,1], 'd')) >= date_min):
        ax.axvline(cv_confirmed_diff_z_roll_max.iloc[i].idxmax() + pd.to_timedelta(confirmed_death_corr_lag_max.iloc[i,1], 'd'), color='k',linestyle=':', linewidth=2, aa=True, label='max cases + x-corr lag')

    ax.set_title(f'{cv_confirmed_diff_z_roll_max.iloc[i].name}\ncases mean = {cv_confirmed_diff.loc[cv_confirmed_diff_z_roll_max.iloc[i].name].mean():.2f}; std dev = {cv_confirmed_diff.loc[cv_confirmed_diff_z_roll_max.iloc[i].name].std():.2f}; total = {cv_confirmed.loc[cv_confirmed_diff_z_roll_max.iloc[i].name][-1]}; peak = {cv_confirmed_diff.loc[cv_confirmed_diff_z_roll_max.iloc[i].name].max():.0f}\ndeaths mean = {cv_death_diff.loc[cv_confirmed_diff_z_roll_max.iloc[i].name].mean():.2f}; std dev = {cv_death_diff.loc[cv_confirmed_diff_z_roll_max.iloc[i].name].std():.2f}; total = {cv_death.loc[cv_confirmed_diff_z_roll_max.iloc[i].name][-1]}; peak = {cv_death_diff.loc[cv_confirmed_diff_z_roll_max.iloc[i].name].max():.0f}\npeak cases to peak deaths time = {(cv_death_diff_z_roll_max.iloc[i].idxmax() - cv_confirmed_diff_z_roll_max.iloc[i].idxmax()).days} days; x-corr lag = {confirmed_death_corr_lag_max.iloc[i,1]} days; x-corr coef = {confirmed_death_corr_lag_max.iloc[i,0]:.2f}', loc='left', fontdict={'fontsize':'medium'})
    ax.set_ylabel('Z Score', fontdict={'fontsize':'small'})
    ax.legend(loc='best', fontsize='x-small')
    ax.grid(b=True, which='major', axis='both', lw=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle('COVID-19 Global 7 days rolling mean - Confirmed Cases - Death lag cross-correlation ', y=0.9995)
plt.figtext(0.75,0.999,f'JHU CSSE COVID-19 Data\nhttps://github.com/CSSEGISandData/COVID-19', fontdict={'size':'x-small'})
plt.figtext(0.92,0.999,f'Copyright (c) {datetime.date.today().strftime("%Y-%m-%d")}\nHuy Dao', fontdict={'size':'x-small'})
fig.set_tight_layout(True)
fig.savefig(os.path.join(ROOT_DIR,f'covid-19-global-z-score-running-mean-ordered-by-max-confirmed-death-lag.png'), dpi='figure', format='png')

#################
plt.close('all')

min_corr = 0.8

g = sns.FacetGrid(despine=False, height=8, aspect=1.5, data=confirmed_death_corr_lag.loc[confirmed_death_corr_lag.loc[:,'correlation'] >= min_corr])
g.map(sns.histplot, 'lag', stat='probability', kde=True, color='teal')
g.set_axis_labels('lag (day)', 'probability')
g.ax.grid(b=True, which='major', axis='both', lw=0.5)
#g.ax.set_xlim((-10,30))
#g.ax.set_xticks(np.arange(-10,30,5))
#g.ax.set_xticklabels(np.arange(-10,30,5))
g.fig.suptitle(f'probability distribution of global confirmed cases to deaths lag, with minimum correlation coefficient of {min_corr}')
g.fig.text(0.60,0.999,f'JHU CSSE COVID-19 Data\nhttps://github.com/CSSEGISandData/COVID-19', fontdict={'size':'x-small'})
g.fig.text(0.92,0.999,f'Copyright (c) {datetime.date.today().strftime("%Y-%m-%d")}\nHuy Dao', fontdict={'size':'x-small'})
g.tight_layout()
g.savefig(os.path.join(ROOT_DIR,f'covid-19-global-probability-distribution-of-confirmed-cases-to-deaths-lag.png'), dpi='figure', format='png')

#################
# the below is for the README.md file

print("confirmed_death_corr_lag.sort_values(by=['correlation', 'lag']).tail(20)")
print(f"{confirmed_death_corr_lag.sort_values(by=['correlation', 'lag']).tail(20)}")
print()


