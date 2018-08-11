import os
import csv
import arrow
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier


def last_month(date):
    """get the name of last month"""
    date = str(date)
    date = date[:4] + '-' + date[4:]
    date0 = arrow.get(date)
    lastmon = date0.shift(months=-1)
    lastmon = lastmon.format('YYYYMM')
    return lastmon


def next_month(date):
    """get the name of next month"""
    date = str(date)
    date = date[:4] + '-' + date[4:]
    date0 = arrow.get(date)
    nextmon = date0.shift(months=1)
    nextmon = nextmon.format('YYYYMM')
    return nextmon


def get_train(date, quantile):
    """get the file name of the training data"""
    date = str(date)
    date0 = date[:4] + '-' + date[4:]
    first = arrow.get(date0)
    quan = quantile.split('m_')[0]
    m = -1 * int(quan)
    second = first.shift(months=-1)
    second = second.format("YYYYMM")
    first = first.shift(months=m)
    first = first.format('YYYYMM')
    ret = first + '-' + second + '_train.csv'
    return ret


def get_test(date):
    """get the file name of the test data"""
    date = str(date)
    ret = date + 'pred.csv'
    return ret


def rf_train(stock_pool_loc, data_index, close_price_data, start_date, end_date, num_trees=100, 
                        max_depth=7, min_samples_leaf=20, stock_num=32, method='Scikit-Learn'):

    """This function uses the random forest to find the superior securities in the A shares, and 
    returns the backtest report within a certain period, and the forecast results of the selected 
    month. This function also provides two training engines: sklearn and lightgbm"""

    df_index = list(close_price_data.index)
    returns_box = []
    days = []
    details = []
    date = start_date
    lines = []
    while date != end_date:
        train_file = os.path.join(stock_pool_loc, 'training', data_index, get_train(date, data_index))
        test_file = os.path.join(stock_pool_loc, 'testing' + get_test(date))
        print(train_file)
        print(test_file)

        train = pd.read_csv(train_file)
        test = pd.read_csv(test_file)
        target = 'yield_class'
        predictors = [x for x in train.columns if x not in [target, 'id']]

        if method == 'Scikit-Learn':
            X_train = train[predictors].values
            y_train = train[target].values
            rfc = RandomForestClassifier(n_estimators=num_trees, max_depth=max_depth, random_state=50, 
                                                                min_samples_leaf=min_samples_leaf)
            rfc.fit(X_train, y_train)
            predictions = rfc.predict_proba(test[predictors].values)[:, 1]
        # elif method == 'LightGBM':
        else:
            train_set = lgb.Dataset(train[predictors].values, label=train[target].values, silent=True)
            params = {'boosting_type': 'rf', 'max_depth': max_depth, 'min_data_in_leaf': min_samples_leaf, 
                            'seed': 50, 'feature_fraction': 0.95,'bagging_fraction': 0.8, 'bagging_freq': 5, 
                            'bagging_seed': 4, 'num_leaves': 31, 'learning_rate': 0.1, 'verbose': 0}
            rf_gbm = lgb.train(params, train_set, num_boost_round=num_trees)
            predictions = rf_gbm.predict(test[predictors].values)

        sorted_pred = np.sort(predictions, axis=0)[::-1]
        predictions = predictions.reshape(len(predictions), )
        predictions = list(predictions)
        sorted_stock_once = []
        for i in range(stock_num):
            label_index = predictions.index(sorted_pred[[i]])
            sorted_stock_once.append(test['id'][label_index][:9])
            predictions[label_index] = 1

        # load close prices of these selected stocks
        # TODO pandas.DataFrame.ix indexer is deprecated, in favor of the more strict .iloc and .loc indexers.
        date1 = '-'.join((date[0:4], date[4:]))
        output_df = close_price_data.ix[date1, sorted_stock_once]
        month_start = df_index[df_index.index(output_df.index[0]) - 1]
        month_end = output_df.index[-1]
        output_df = close_price_data.ix[month_start:month_end, sorted_stock_once]
        
        # calculate monthly returns
        returns_rates = output_df.iloc[-1].values / output_df.iloc[0].values
        returns_box.append(np.mean(returns_rates))
        print('Monthly return is', np.mean(returns_rates) * 100, '%')
        
        # calculate daily returns
        for day in range(1, len(output_df.index)):
                returns_rates_detail = output_df.iloc[day].values / output_df.iloc[day-1].values
                details.append(returns_rates_detail)

        l_tmp = list(output_df.index)
        if date != last_month(end_date):
            l_tmp.pop()
        days.extend(l_tmp)
        line = [date]
        line.extend([str(stock) for stock in sorted_stock_once])
        lines.append(line)
        print(date, 'has been done.')
        
        date = next_month(date)

    # output data of selected stocks
    save_path = stock_pool_loc + '/GUI data/' + data_index + '/RF/' + str(num_trees)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file = save_path + '/RF_' + start_date + '_' + end_date + '_' + str(stock_num) + '.txt'
    # TODO do not use csv module
    with open(save_file, 'w', newline='') as f:
        cw = csv.writer(f, delimiter="\t")
        cw.writerows(lines)

    # calculate annualized returns
    returns_box = np.array(returns_box)
    annualized_return = ((np.prod(returns_box) ** (250 / (len(details)-1))) - 1) * 100
    details = np.array(details)
    results = [1, np.mean(details[0])]
    for k in range(2, len(details) + 1):
        results.append(np.mean(np.prod(details[0:k], axis=0)))
    print('Train has been done.')

    return annualized_return, days, np.array(results), save_file
