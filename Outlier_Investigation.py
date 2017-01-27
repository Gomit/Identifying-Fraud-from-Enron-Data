import matplotlib.pyplot
from pprint import pprint

from feature_format import featureFormat, targetFeatureSplit

def plot(data_dict):

    features = ["salary", "bonus"]
    data = featureFormat(data_dict, features)

    # plot salary and bonus
    for point in data:
        salary = point[0]
        bonus = point[1]
        matplotlib.pyplot.scatter( salary, bonus )

    matplotlib.pyplot.xlabel("Salary");
    matplotlib.pyplot.ylabel("Bonus");
    matplotlib.pyplot.show();
    
def bonus_outlier(data_dict):
    bonus_outliers = []
    for key in data_dict:
        val = data_dict[key]['bonus']
        if val == 'NaN':
            continue
        bonus_outliers.append((key,int(val)))
   
    # print the top 5 bonus_outliers
    pprint(sorted(bonus_outliers,key=lambda x:x[1],reverse=True)[:5])
    
    
def salary_outliers(data_dict):
    salary_outliers = []
    for key in data_dict:
        val = data_dict[key]['salary']
        if val == 'NaN':
            continue
        salary_outliers.append((key,int(val)))
    
    # print the top 5 bonus_outliers
    pprint(sorted(salary_outliers,key=lambda x:x[1],reverse=True)[:5])
    
def plot_clean(data_dict):

    features = ["salary", "bonus"]
    # erase outliers
    data_dict.pop( "TOTAL", 0 )
    data_dict.pop( "THE TRAVEL AGENCY IN THE PARK", 0 )
    data = featureFormat(data_dict, features)

    # plot salary and bonus again

    for point in data:
        salary = point[0]
        bonus = point[1]
        matplotlib.pyplot.scatter( salary, bonus )

    matplotlib.pyplot.xlabel("Salary");
    matplotlib.pyplot.ylabel("Bonus");
    matplotlib.pyplot.show();