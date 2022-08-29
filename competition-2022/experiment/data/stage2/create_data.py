import pandas as pd
import numpy as np
from functools import reduce
from sklearn.model_selection import train_test_split

# rootdir = '/home/olivetti/Projects/covid19-forecast-hub'
rootdir = '/home/bill/projects/covid19-forecast-hub'
datadir = f'{rootdir}/data-truth'
loc = 'New York'

def remove_outliers(df, d=4, w=pd.to_timedelta(1,unit='W'),fill=True):
    """remove outliers.
    :param w: time window
    :param d: # of std deviations
    For each point:
        - compute average of +- window centered at point
        - if point is greater than d standard deviations from the average, mark it as outlier
    2. remote outliers
    
    """
    cols = df['value_name'].unique()
#     outliers = {k:[] for k in cols}
    outliers = dict()
    for c in cols: 
        x = df.loc[df.value_name==c]
        outliers[c] = {'idx':[], 'fillval':[]} 
        for idx,row in x.iterrows():
#             ipdb.set_trace()
            L = clip_window(x, row.date-w, row.date)
            R = clip_window(x, row.date, row.date+w)
            if len(L) == 0:
                M = L
            elif len(R) ==0:
                M = R
            else:
                M = pd.concat([L,R])
#             print(f'L: {L.shape}; R: {R.shape}')
#             display(M)
            MA = M['value'].median()
            MS = M['value'].quantile(.25)
            if np.abs(row.value-MA) > d*MS:
                outliers[c]['idx'].append(idx)
                outliers[c]['fillval'].append(MA)
    # fill or remove marked points
    dfc = df.copy()         
    if fill:
        for c in cols:
            dfc.loc[outliers[c]['idx'],'value'] = outliers[c]['fillval']
    else:
        dfc = dfc.dropna()
    
#     display(dfc)
    return dfc, outliers

def clip_window(df, start=None, end=None):
    """clip data to within start and end dates"""
    if start == None:
        start = df.date.min() - np.timedelta64(1,'D')
    if end == None:
        end = df.date.max() + np.timedelta64(1,'D')
    return df.loc[(df.date > start)
                  & (df.date < end)]

def create_lag(df, days_ahead, nfeatures):
    start = days_ahead # days_ahead is how many days ahead the prediction should be 
    end = start + nfeatures # how many days before the current one should we use 
    targets = ['value_cases', 'value_deaths', 'value_hosp']
    for i in range(start, end): # the raw values of the known period 
        for n in targets:
            var_name = f'{n}-{i}'
            df[var_name] = df[n].shift(i)
            var_name = f'{n}-Delta-{i}'
            df[var_name] = df[f'{n}-{i}'] - df[f'{n}-{start}']
    for n in targets: # the cummulative sum of the total period until today 
        var_name = f'{n}-tot'
        df[var_name] = df[n].shift(start).cumsum()
    return df.iloc[end:] # make sure there is no nan, the 'value_cases, value_deaths, value_hosp' variables will be the target

def create_Xy(df, target):
    targets = ['value_cases', 'value_deaths', 'value_hosp']
    y = df[target]
    X = df.loc[:, ~df.columns.isin(targets)]
    # dfc = df.loc[:, ~df.columns.isin(targets)]
    X = X.rename(columns={k:k.replace('-','_') for k in X.columns}) 
    header = list(X.columns.values)
    header.append(target) 
    return X, y, ",".join(header)

def split_train_test(X, y, n_train):
    X_train = X[:n_train, :]
    y_train = y[:n_train]
    X_test = X[n_train:, :]
    y_test = y[n_train:]
    return X_train, y_train, X_test, y_test

def save_sets(rs, fname, days_ahead, nfeatures, X_train, y_train, X_test, y_test, header):
    df_train = X_train.join(y_train, on='date') 
    df_test = X_test.join(y_test, on='date') 
    # df_train = pd.merge(X_train, y_train)
    # Z_train = np.c_[X_train, y_train]
    # Z_test = np.c_[X_test, y_test]
    df_train.to_csv(f'data/{rs}_{fname}_{days_ahead}_{nfeatures}_train.csv')
    df_test.to_csv(f'data/{rs}_{fname}_{days_ahead}_{nfeatures}_test.csv')
    # np.savetxt(f'data/{rs}_{fname}_{days_ahead}_{nfeatures}_train.csv', Z_train, delimiter=',', header=header,comments='')
    # np.savetxt(f'data/{rs}_{fname}_{days_ahead}_{nfeatures}_test.csv', Z_test, delimiter=',', header=header,comments='')

def make_exogenous(dates):
    """Add data about masking and vaccination status in NY by date. 
    source: https://ballotpedia.org/Documenting_New_York%27s_path_to_recovery_from_the_coronavirus_(COVID-19)_pandemic,_2020-2021
    mask_mandate
    vaccination_rate
    travel_restriction
    school_closure
    """
    
    """
    Vaccinations
    """
    df_vax = pd.read_csv('New_York_State_Statewide_COVID-19_Vaccination_Data_by_County.csv')
    df_vax = df_vax.rename(columns={'Report as of':'date'})
    df_vax['date'] = df_vax['date'].astype(np.datetime64)
    total_vax =df_vax.groupby('date').sum().max()
    vax_total = df_vax.groupby('date').sum()['Series Complete'].rename('CompleteVaccinationsTotal')
    vax_diff = df_vax.groupby('date').sum().diff()['Series Complete'].rename('CompleteVaccinationsDaily')
    
    """
    Masking
        - On April 15, 2020, Gov. Andrew Cuomo (D) signed an executive order requiring individuals to wear face coverings in public. 
        On April 27, 2021, Cuomo announced people who were fully vaccinated did not have to wear masks in public outdoor spaces. 
        The New York City Health Department released guidance on face coverings. 
        - Starting May 19, 2021, vaccinated people did not have to wear masks in most indoor public settings, 
        aligning the state’s policy with Centers for Disease Control and Prevention (CDC) guidance. 
        - On Dec. 10, Gov. Kathy Hochul (D) announced a new statewide mask requirement would take effect starting Dec. 13. 
        Masks were required regardless of vaccination status at indoor public settings, unless the business or venue required 
        proof of vaccination.
        - On Feb. 9, Hochul announced the statewide mask requirement would end, effective Feb. 10.
    """
    
    mask_mandate = [
        (np.datetime64('2020-04-15'), np.datetime64('2021-05-19')),
        (np.datetime64('2021-12-13'), np.datetime64('2022-02-10'))
                   ]
    
    """
    travel restriction
        - June 24, 2020: Govs. Ned Lamont (D-Conn.), Phil Murphy (D-N.J.), and 
        Andrew Cuomo (D-N.Y.) announced on June 24 that travelers arriving in
        their states from states with a high infection rate must quarantine 
        for 14 days. 
        The infection rate is based on a seven-day rolling average of the 
        number of infections per 100,000 residents. 
        As of June 24, the states that meet that threshold are Alabama, 
        Arkansas, Arizona, Florida, North Carolina, South Carolina, Texas, 
        and Utah.[55]
        - April 1, 2021: Travelers to New York are no longer required to 
        self-quarantine upon arrival or display a negative COVID-19 test
    """
    travel_restriction = [(np.datetime64('2020-06-24'), np.datetime64('2021-04-01'))]
    
    """
    School closures
        - March 16, 2020: Cuomo announced that schools across the state would 
        close for at least two weeks beginning March 18 
        - narrator: "they were to remain closed much longer than that"
        - Sept. 10, 2020: Schools re-open. At the beginning of the school year, 
        Burbio reported about half of schools were in-person in New York
        - June 29, 2021: At the end of the school year, Burbio reported most 
        schools were in-person in New York.
    """
    school_closed = [(np.datetime64('2020-03-18'), np.datetime64('2020-09-10'))] 
    
    """
    make dataframe
    """
    policy_status = []
    for day in dates:
        masking = any( mm[0] <= day <= mm[1] for mm in mask_mandate)
        travel = any( tr[0] <= day <= tr[1] for tr in travel_restriction)
        school = any( tr[0] <= day <= tr[1] for tr in school_closed)
        policy_status.append(dict(
            date=day,
            mask_mandate=masking,
            travel_restriction=travel,
            school_closed=school
        ))
        
    df = pd.DataFrame.from_records(policy_status).set_index('date').astype(int)  
    df = (df
     .join(vax_total, on='date')
     .join(vax_diff, on='date')
    )
    
    df_clean,_ =  remove_outliers(df.reset_index().melt(id_vars='date',
                                                        var_name='value_name'), 
                               d=1, 
                               w = pd.to_timedelta(1,'W') 
                              )
    return df_clean.pivot(columns='value_name',values='value',index='date')

################################################################################
frames = []
for v in ['Cases','Deaths','Hospitalizations']:
    tmp = pd.read_csv(f'{datadir}/truth-Incident {v}.csv')
    tmp = tmp.loc[tmp.location_name==loc]
    tmp['value_name'] = v
    print(v,'shape:',tmp.shape)
    frames.append(tmp)

df = pd.concat(frames, ignore_index=True)
df['date'] = pd.to_datetime(df['date']) 

df_clipped = clip_window(df, start=np.datetime64('2020-08'))
df_in = df_clipped
tmp,outliers = remove_outliers(df_in, 
                               d=1, 
                               w = pd.to_timedelta(1,'W') 
                              )

dfplt = df_in.copy()
dfplt['outliers'] = False
outidx = np.hstack([outliers[c]['idx'] for c in df_in.value_name.unique()])
dfplt.loc[outidx,'outliers'] = True 
df_clean = tmp

targets = ['value_cases', 'value_deaths', 'value_hosp']
df_cases = df_clean[df_clean.value_name=='Cases'].rename(columns={'value':'value_cases'}).set_index('date')
df_deaths = df_clean[df_clean.value_name=='Deaths'].rename(columns={'value':'value_deaths'}).set_index('date')
df_hosp = df_clean[df_clean.value_name=='Hospitalizations'].rename(columns={'value':'value_hosp'}).set_index('date')

df_final = (df_cases
            .join(df_deaths, lsuffix='_cases')
            .join(df_hosp, on='date', rsuffix='_hosp')
            [targets]
           )
# add exogenous variables
df_ex = make_exogenous(df_final.index) 
df_final = df_final.join(df_ex,on='date').fillna(0)

df_final['value_cases'] = df_final['value_cases'].ewm(halflife="7 days", times=df_final.index).mean()
df_final['value_deaths'] = df_final['value_deaths'].ewm(halflife="7 days", times=df_final.index).mean()
df_final['value_hosp'] = df_final['value_hosp'].ewm(halflife="7 days", times=df_final.index).mean()

def mae(y, yhat):
    return np.abs(y-yhat).mean()

targets = ['value_cases', 'value_deaths', 'value_hosp']
# difficulties = [(1, 1), (5, 7), (7, 7)] # next day forecast using current day, five days forecast using past 7 days, 7 day forecast using last 7 days.
difficulties = [
    # (7, 7), 
    (14, 14)
] # one week forecast using past 7 days, two week forecast using last 14 days.

for t in targets:
    for n,p in difficulties:
        for rs in [135]:
            print(f'{t}, {n}, {p}:')
            
            X, y, header = create_Xy(create_lag(df_final.copy(), n, p), t)    
            # X_train, y_train, X_test, y_test = split_train_test(X, y, 300)
            # divide into contiguous train test splits 
            w = 56
            split = .625
            # w = 900
            # split = .5
            i = 0
            print('training window:',int(split*w),'days')
            print('test window:',int((1-split)*w),'days')
            X_train, X_test, y_train, y_test = None, None, None, None
            while i < len(X):

                j = int(i+split*w)
                k = i+w
                # print(i,j,k)
                # print('X_train:',X.iloc[i:j])
                # print('X_test:',X.iloc[j:k])
                if X_train is None:
                    X_train = X.iloc[i:j]
                    y_train = y.iloc[i:j]
                    X_test = X.iloc[j:k]
                    y_test = y.iloc[j:k]
                else:
                    X_train = X_train.append(X.iloc[i:j]) 
                    y_train = y_train.append(y.iloc[i:j]) 
                    X_test = X_test.append(X.iloc[j:k]) 
                    y_test = y_test.append(y.iloc[j:k]) 
                i += w

            save_sets(rs, f'NY_{t}', n, p, X_train, y_train, X_test, y_test, header)

            from sklearn.linear_model import LinearRegression
            reg = LinearRegression()
            reg.fit(X_train, y_train)
            print("LR scores: ", 
                  'R2:',reg.score(X_test, y_test), 
                  'MAE:',mae(y_test, reg.predict(X_test))
                 )
