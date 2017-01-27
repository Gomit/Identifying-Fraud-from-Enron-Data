import pickle

def Q_A(data_dict):

    #Question: How many data points (people) are in the dataset?
    print "Number of people in the dataset:",len(data_dict)

    #Question: For each person, how many features are available??
    print "Number of feature for each person:",len(data_dict.itervalues().next())

    #Question: How many POIs are there in the Enron dataset?
    x=0
    for i in data_dict:
        if data_dict[i]["poi"]==1:
            x+=1
    print "number of POI's in the dataset:",x

    #Question: What is the total value of the stock belonging to James Prentice?
    print "Total value of the stock - James Prentice:", data_dict["PRENTICE JAMES"]["total_stock_value"]

    #Question: How many email messages do we have from Wesley Colwell to persons of interest?
    print "Nr of email messages to poi - Wesley Colwell:", data_dict["COLWELL WESLEY"]["from_this_person_to_poi"]

    #Question: What's the value of stock options exercised by Jeffrey K Skilling?
    print "Value of stock options - Jeffrey K Skillin:", data_dict["SKILLING JEFFREY K"]["exercised_stock_options"]

    #Question: Of these three individuals (Lay, Skilling and Fastow), who took home the most money (largest value of "total_payments" feature)
    print "Total Payments - Lay:", data_dict["LAY KENNETH L"]["total_payments"]
    print "Total Payments - Skilling:", data_dict["SKILLING JEFFREY K"]["total_payments"]
    print "Total Payments - Fastow:", data_dict["FASTOW ANDREW S"]["total_payments"]

    #Question: How many folks in this dataset have a quantified salary? 
    x=0
    for i in data_dict:
        if data_dict[i]["salary"]!='NaN':
            x+=1
    print "Quantified salary:",x

    #What about a known email address?
    x=0
    for i in data_dict:
        if data_dict[i]["email_address"]!='NaN':
            x+=1
    print "Known email address:",x

    #Question: How many people in the E+F dataset (as it currently exists) have 
    #"NaN" for their total payments? What percentage of people in the dataset as a whole is this? 
    x=0
    for i in data_dict:
        if data_dict[i]["total_payments"]=='NaN':
            x+=1
    print "Nr of 'NaN' for their total payments:",x
    print "Percentage of people:", (float(x) / len(data_dict)) * 100

    #Question: How many POIs in the E+F dataset have 'NaN' for their total 
    #payments? What percentage of POI's as a whole is this?"
    x=0
    for i in data_dict:
        if data_dict[i]["poi"]==True and data_dict[i]["total_payments"]=='NaN':
            x+=1
    print "Nr of 'NaN' for their total payments:",x
    print "Percentage of people:", (float(x) / len(data_dict)) * 100;
