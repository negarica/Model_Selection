import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

def ModelQuality(data,
                 time_var,
                 location_var,
                 outcome_var,
                 Frequency,
                 control_vars=None,
                 MDE=1.01,
                 sims=100,
                 method='ols',
                 agg=False):
    '''The function returns type I error and power of different modelling approaches for assessing the effect of a
       switchback test that results in a fixed minimum change (MDE) in the outcome variable of interest. The
       required inputs are as below:
       data: dataframe containing variables of interest,
       time_var: name of the dataframe column containing time of the order variable,
       location_var: name of the dataframe column containing location of order variable,
       outcome_var: name of the dataframe column containing the outcome variable of interest,
       control_vars: (optional) list of control variables
       Frequency: time frequency used to identify region_time units.
       MDE: 1+ minimum detectable effect of choice (defaults to 0.01),
       sims: number of simulations (default is 100) ,
       method: ols, mlm (default is ols)
       agg: binary flag to determine whether to test on order level or aggregate to region_time level.
    '''
    if method not in ['ols', 'mlm']:
        raise Exception('Function not implemented for method {}'.format(method))
    if agg and method == 'mlm':
        raise Exception('Aggregation not supported for method mlm')
    # make a local copy of the relevant data cols:
    cols = [time_var, location_var, outcome_var]
    if control_vars is not None:
        cols.extend(control_vars)
    data = data[cols].copy()
    data[time_var] = pd.to_datetime(data[time_var])
    alpha = 0.05

    significant_resultsAA = []
    significant_resultsAB = []

    if agg:
        data_agg = data.groupby([location_var, pd.Grouper(key=time_var, freq=Frequency)]).mean()
        for i in range(0, sims):
            binom = np.random.binomial(1, 0.5, len(data_agg))
            variant = data_agg[binom == 1]
            # Create a variant variable that takes one when an order is in the variant group and zero otherwise:
            data_agg['variant'] = 0
            data_agg.loc[data_agg.index.isin(variant.index), 'variant'] = 1

            if method == 'ols':
                # Create the outcome variable required for the A/B tests:
                data_agg['outcome_sim'] = data_agg[outcome_var]
                data_agg.loc[data_agg['variant'] == 1, 'outcome_sim'] = data_agg[outcome_var] * MDE
                
                # A/A test:
                modelAA = sm.OLS(data_agg[outcome_var],
                                 sm.add_constant(data_agg['variant'])).fit()
                p_value_AA = modelAA.pvalues.loc['variant']
                # A/B test:
                modelAB = sm.OLS(data_agg['outcome_sim'],
                                 sm.add_constant(data_agg['variant'])).fit()
                p_value_AB = modelAB.pvalues.loc['variant']

            elif method == 'mlm':
                raise Exception('Aggregation not supported for method mlm')

            significant_resultsAA.append(float(p_value_AA < alpha))
            significant_resultsAB.append(float(p_value_AB < alpha))
    else:
        gb = data.groupby(
            [pd.Grouper(key=time_var, freq=Frequency, sort=True), location_var],
            as_index=False)
        s = gb.apply(lambda x: x[outcome_var])
        # Get the order level indicies
        order = s.index.get_level_values(1)
        # A new column that assigns group level indexes of gb to order level observations
        data.loc[order, 'period_city'] = s.index.get_level_values(0)
        data.sort_values('period_city', inplace=True)
        unique_PeriodCity = data.period_city.unique()
        for i in range(0, sims):
            binom = np.random.binomial(1, 0.5, unique_PeriodCity.size)
            variant = unique_PeriodCity[binom == 1]
            # Create a variant variable that takes one when an order is in the variant group and zero otherwise:
            data['variant'] = 0
            data.loc[data['period_city'].isin(variant), 'variant'] = 1

            # Create the outcome variable required for the A/B tests:
            data['outcome_sim'] = data[outcome_var]
            data.loc[data['variant'] == 1, 'outcome_sim'] = data[outcome_var] * MDE

            if method == 'ols':
                # A/A test:
                modelAA = sm.OLS(data[outcome_var],
                                 sm.add_constant(data['variant'])).fit(cov_type='cluster',
                                                                       cov_kwds={'groups': data['period_city']})
                p_value_AA = modelAA.pvalues.loc['variant']
                # A/B test:
                modelAB = sm.OLS(data['outcome_sim'],
                                 sm.add_constant(data['variant'])).fit(cov_type='cluster',
                                                                       cov_kwds={'groups': data['period_city']})
                p_value_AB = modelAB.pvalues.loc['variant']

            elif method == 'mlm':
                # MLM A/A test:
                modelAA = smf.mixedlm('{} ~ variant'.format(outcome_var), data, groups=data['period_city']).fit()
                p_value_AA = modelAA.pvalues.loc['variant']

                # MLM A/B test:
                modelAB = smf.mixedlm('outcome_sim ~ variant', data, groups=data['period_city']).fit()
                p_value_AB = modelAB.pvalues.loc['variant']

            significant_resultsAA.append(float(p_value_AA < alpha))
            significant_resultsAB.append(float(p_value_AB < alpha))

    typeI_error = np.mean(significant_resultsAA)
    power = np.mean(significant_resultsAB)
    performance = [typeI_error, power]
    return (performance)

