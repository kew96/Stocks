from datetime import datetime, date
from talib import MA_Type


def check_convert_date(var, name):
    if type(var) not in (date, datetime, str):
        while True:
            try:
                str_date = input(f'Please enter a valid date for the {name} date using the pattern YYYY-MM-DD:\n')
                return datetime.strptime(str_date, '%Y-%m-%d').date()
            except ValueError:
                pass
    elif type(var) is str:
        while True:
            try:
                return datetime.strptime(var, '%Y-%m-%d').date()
            except ValueError:
                var = input(f'Please enter a valid date for the {name} date using the pattern YYYY-MM-DD:\n')
    elif type(var) is datetime:
        return var.date()
    else:
        return var


def check_list_options(var, options, name):
    if type(var) == str:
        output = f'Please choose one of the following options by number for {name}: \n'
        for i, o in enumerate(options):
            output = output + str(i + 1) + '. ' + o + '\n'
        while True:
            if var.lower() in options:
                return var.lower()
            else:
                if var is None:
                    pass
                elif var.lower() != 'help':
                    print('"' + var + '" is not a valid option.\n')
                try:
                    var = options[int(input(output)) - 1].lower()
                except IndexError:
                    print('Please restart and only enter valid indices for choices.')
    elif callable(var):
        raise NotImplementedError


def check_series_type(series_type):
    return check_list_options(series_type, ['close', 'open', 'high', 'low'], 'series_type')


# noinspection PyProtectedMember
def check_matype(var, name):
    if type(var) == str:
        output = f'Please choose one of the following options by number for {name}: \n'
        for i, o in enumerate(MA_Type._lookup.values()):
            output = output + str(i + 1) + '. ' + o + '\n'
        while True:
            if var.lower() in MA_Type._lookup.values():
                return list(MA_Type._lookup.values()).index(var.lower())
            elif int(var) in MA_Type._lookup.keys():
                return int(var)
            else:
                if var is None:
                    raise NotImplementedError
                elif var.lower() != 'help':
                    print('"' + var + '" is not a valid option.\n')
                try:
                    var = int(input(output)) - 1
                except IndexError:
                    print('Please restart and only enter valid indices for choices.')
