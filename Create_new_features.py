import pandas
import numpy
from sklearn.feature_selection import SelectKBest
from feature_format import featureFormat,targetFeatureSplit

def check_NaN(data_dict):
    # load data into pandas
    df = pandas.DataFrame.from_records(list(data_dict.values()))
    persons = pandas.Series(list(data_dict.keys()))
    
    # Convert to numpy nan
    df.replace(to_replace='NaN', value=numpy.nan, inplace=True)

    # Count number of NaN's for columns
    return df

def kbest(data_dict):
    features_list = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments', 
                 'loan_advances', 'bonus', 'restricted_stock_deferred', 
                 'deferred_income', 'total_stock_value', 'expenses', 'from_poi_to_this_person', 
                 'exercised_stock_options', 'from_messages', 'other', 'from_this_person_to_poi', 
                 'long_term_incentive', 'shared_receipt_with_poi', 'restricted_stock', 'director_fees']

    data = featureFormat(data_dict, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)

    #select k best
    selection = SelectKBest(k=10)
    features = selection.fit_transform(features, labels)
    features_selected = selection.get_support(indices = True)
    scores = selection.scores_
    unsorted_pairs = zip(features_list[1:], scores)
    sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
    k_best_features = dict(sorted_pairs[:10])
    return k_best_features

def new_features(data_dict,features_list):
    fields = ['to_messages', 'from_messages',
              'from_poi_to_this_person', 'from_this_person_to_poi']
    for record in data_dict:
        person = data_dict[record]
        total_email=True
        fraction_email=True
        for field in fields:
            if person[field] == 'NaN':
                total_email=False
                fraction_email=False
        if total_email and fraction_email:
            poi_messages = person['from_poi_to_this_person'] +\
                           person['from_this_person_to_poi']
            total_messages = person['to_messages'] +\
                             person['from_messages']
            person['poi_ratio'] = float(poi_messages) / total_messages
        else:
            person['poi_ratio'] = 'NaN'
        
        if fraction_email:
            person['fraction_to_poi'] = float(person['from_this_person_to_poi'])/\
            person['from_messages']
            person['fraction_from_poi'] = float(person['from_poi_to_this_person'])/\
            person['to_messages']
        else:
            person['fraction_to_poi'] = 'NaN'  
            person['fraction_from_poi'] = 'NaN'     
    features_list += ['poi_ratio','fraction_to_poi','fraction_from_poi']  
