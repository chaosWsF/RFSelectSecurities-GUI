import arrow


def last_month(date):
    """get the name of last month"""
    date = str(date)
    date = date[:4] + '-' + date[4:]
    date0 = arrow.get(date)
    lastmon = date0.shift(months=-1)
    lastmon = lastmon.format('YYYYMM')
    return lastmon


def base_a(data, start_date, end_date):
    """get the base of A shares"""
    df_index = list(data.index)
    end_date = last_month(end_date)
    start_date = '-'.join((start_date[0:4], start_date[4:]))
    end_date = '-'.join((end_date[0:4], end_date[4:]))
    output_data = data[start_date:end_date]

    month_start = df_index[df_index.index(output_data.index[0]) - 1]
    month_end = output_data.index[-1]
    output_data = data.ix[month_start:month_end]

    day_stamps = len(output_data.index) - 1

    standard_returns = output_data.values / output_data.values[0]
    annualized_return = ((standard_returns[-1] ** (250 / day_stamps)) - 1) * 100

    return annualized_return[0], standard_returns, list(output_data.index)
