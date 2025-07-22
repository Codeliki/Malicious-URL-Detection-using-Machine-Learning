import pandas as pd
import itertools
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from lightgbm import LGBMClassifier
import os
import seaborn as sns
from wordcloud import WordCloud

dset=pd.read_csv('/content/archive (14).zip')

print(dset.shape)
dset.head()

from google.colab import drive
drive.mount('/content/drive')

dset.type.value_counts()

dset_phish = dset[dset.type=='phishing']
dset_malware = dset[dset.type=='malware']
dset_deface = dset[dset.type=='defacement']
dset_benign = dset[dset.type=='benign']

phish_url = " ".join(j for j in dset_phish.url)
wcloud = WordCloud(width=1600, height=800,colormap='Paired').generate(phish_url)
plt.figure( figsize=(12,14),facecolor='k')
plt.imshow(wcloud, interpolation='bilinear')
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

malware_url = " ".join(j for j in dset_malware.url)
wcloud = WordCloud(width=1600, height=800,colormap='Paired').generate(malware_url)
plt.figure( figsize=(12,14),facecolor='k')
plt.imshow(wcloud, interpolation='bilinear')
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

deface_url = " ".join(j for j in dset_deface.url)
wcloud = WordCloud(width=1600, height=800,colormap='Paired').generate(deface_url)
plt.figure( figsize=(12,14),facecolor='k')
plt.imshow(wcloud, interpolation='bilinear')
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

benign_url = " ".join(j for j in dset_benign.url)
wcloud = WordCloud(width=1600, height=800,colormap='Paired').generate(benign_url)
plt.figure( figsize=(12,14),facecolor='k')
plt.imshow(wcloud, interpolation='bilinear')
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

import re
#Use of IP or not in domain
def having_ip_address(url):
    match = re.search(
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
        '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4
        '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)' # IPv4 in hexadecimal
        '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)  # Ipv6
    if match:
        # print match.group()
        return 1
    else:
        # print 'No matching pattern found'
        return 0
dset['use_of_ip'] = dset['url'].apply(lambda j: having_ip_address(j))

from urllib.parse import urlparse

def abnormal_url(url):
    hostname = urlparse(url).hostname
    hostname = str(hostname)
    match = re.search(hostname, url)
    if match:
        # print match.group()
        return 1
    else:
        # print 'No matching pattern found'
        return 0


dset['abnormal_url'] = dset['url'].apply(lambda j: abnormal_url(j))

#!pip install googlesearch-python

from googlesearch import search

def google_index(url):
    site = search(url, 5)
    return 1 if site else 0
dset['google_index'] = dset['url'].apply(lambda j: google_index(j))

def count_dot(url):
    count_dot = url.count('.')
    return count_dot

dset['count.'] = dset['url'].apply(lambda j: count_dot(j))
dset.head()

def count_www(url):
    url.count('www')
    return url.count('www')

dset['count-www'] = dset['url'].apply(lambda j: count_www(j))

def count_atrate(url):
     
    return url.count('@')

dset['count@'] = dset['url'].apply(lambda j: count_atrate(j))


def no_of_dir(url):
    urldir = urlparse(url).path
    return urldir.count('/')

dset['count_dir'] = dset['url'].apply(lambda j: no_of_dir(j))

def no_of_embed(url):
    urldir = urlparse(url).path
    return urldir.count('//')

dset['count_embed_domian'] = dset['url'].apply(lambda j: no_of_embed(j))


def shortening_service(url):
    match = re.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                      'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                      'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                      'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                      'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                      'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                      'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
                      'tr\.im|link\.zip\.net',
                      url)
    if match:
        return 1
    else:
        return 0
    
    
dset['short_url'] = dset['url'].apply(lambda j: shortening_service(j))

def count_https(url):
    return url.count('https')

dset['count-https'] = dset['url'].apply(lambda j : count_https(j))

def count_http(url):
    return url.count('http')

dset['count-http'] = dset['url'].apply(lambda j : count_http(j))

def count_per(url):
    return url.count('%')

dset['count%'] = dset['url'].apply(lambda j : count_per(j))

def count_ques(url):
    return url.count('?')

dset['count?'] = dset['url'].apply(lambda j: count_ques(j))

def count_hyphen(url):
    return url.count('-')

dset['count-'] = dset['url'].apply(lambda j: count_hyphen(j))

def count_equal(url):
    return url.count('=')

dset['count='] = dset['url'].apply(lambda j: count_equal(j))

def url_length(url):
    return len(str(url))


#Length of URL
dset['url_length'] = dset['url'].apply(lambda j: url_length(j))
#Hostname Length

def hostname_length(url):
    return len(urlparse(url).netloc)

dset['hostname_length'] = dset['url'].apply(lambda j: hostname_length(j))
dset.head()

def suspicious_words(url):
    match = re.search('PayPal|login|signin|bank|account|update|free|lucky|service|bonus|ebayisapi|webscr',
                      url)
    if match:
        return 1
    else:
        return 0
dset['sus_url'] = dset['url'].apply(lambda j: suspicious_words(j))


def digit_count(url):
    digits = 0
    for j in url:
        if j.isnumeric():
            digits = digits + 1
    return digits


dset['count-digits']= dset['url'].apply(lambda j: digit_count(j))


def letter_count(url):
    letters = 0
    for j in url:
        if j.isalpha():
            letters = letters + 1
    return letters


dset['count-letters']= dset['url'].apply(lambda j: letter_count(j))

dset.head()

#!pip install tld

pip install tld

#Importing dependencies
from urllib.parse import urlparse
from tld import get_tld
import os.path

#First Directory Length
def fd_length(url):
    urlpath= urlparse(url).path
    try:
        return len(urlpath.split('/')[1])
    except:
        return 0

dset['fd_length'] = dset['url'].apply(lambda j: fd_length(j))

#Length of Top Level Domain
dset['tld'] = dset['url'].apply(lambda j: get_tld(j,fail_silently=True))


def tld_length(tld):
    try:
        return len(tld)
    except:
        return -1

dset['tld_length'] = dset['tld'].apply(lambda j: tld_length(j))

dset = dset.drop("tld",1)

dset.columns

dset['type'].value_counts()

import seaborn as sns
sns.set(style="darkgrid")
ax = sns.countplot(y="type", data=dset,hue="use_of_ip")

sns.set(style="darkgrid")
ax = sns.countplot(y="type", data=dset,hue="abnormal_url")

sns.set(style="darkgrid")
ax = sns.countplot(y="type", data=dset,hue="google_index")

sns.set(style="darkgrid")
ax = sns.countplot(y="type", data=dset,hue="short_url")

sns.set(style="darkgrid")
ax = sns.countplot(y="type", data=dset,hue="sus_url")

sns.set(style="darkgrid")
ax = sns.catplot(x="type", y="count.", kind="box", data=dset)

sns.set(style="darkgrid")
ax = sns.catplot(x="type", y="count-www", kind="box", data=dset)

sns.set(style="darkgrid")
ax = sns.catplot(x="type", y="count@", kind="box", data=dset)

sns.set(style="darkgrid")
ax = sns.catplot(x="type", y="count_dir", kind="box", data=dset)

sns.set(style="darkgrid")
ax = sns.catplot(x="type", y="hostname_length", kind="box", data=dset)

sns.set(style="darkgrid")
ax = sns.catplot(x="type", y="fd_length", kind="box", data=dset)

sns.set(style="darkgrid")
ax = sns.catplot(x="type", y="tld_length", kind="box", data=dset)

from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
dset["type_code"] = lb_make.fit_transform(dset["type"])
dset["type_code"].value_counts()

#Predictor Variables
# filtering out google_index as it has only 1 value
X = dset[['use_of_ip','abnormal_url', 'count.', 'count-www', 'count@',
       'count_dir', 'count_embed_domian', 'short_url', 'count-https',
       'count-http', 'count%', 'count?', 'count-', 'count=', 'url_length',
       'hostname_length', 'sus_url', 'fd_length', 'tld_length', 'count-digits',
       'count-letters']]

#Target Variable
y = dset['type_code']

X.head()

X.columns

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2,shuffle=True, random_state=5)

import sklearn.metrics as metrics

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100,max_features='sqrt')
rf.fit(X_train,y_train)
y_pred_rf = rf.predict(X_test)
print(classification_report(y_test,y_pred_rf,target_names=['benign', 'defacement','phishing','malware']))

score = metrics.accuracy_score(y_test, y_pred_rf)
print("accuracy:   %0.3f" % score)

cm = confusion_matrix(y_test, y_pred_rf)
cm_df = pd.DataFrame(cm,
                     index = ['benign', 'defacement','phishing','malware'], 
                     columns = ['benign', 'defacement','phishing','malware'])
plt.figure(figsize=(8,6))
sns.heatmap(cm_df, annot=True,fmt=".1f")
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()

feat_importances = pd.Series(rf.feature_importances_, index=X_train.columns)
feat_importances.sort_values().plot(kind="barh",figsize=(10, 6))

lgb = LGBMClassifier(objective='multiclass',boosting_type= 'gbdt',n_jobs = 5, 
          silent = True, random_state=5)
LGB_C = lgb.fit(X_train, y_train)


y_pred_lgb = LGB_C.predict(X_test)
print(classification_report(y_test,y_pred_lgb,target_names=['benign', 'defacement','phishing','malware']))

score = metrics.accuracy_score(y_test, y_pred_lgb)
print("accuracy:   %0.3f" % score)

cm = confusion_matrix(y_test, y_pred_lgb)
cm_df = pd.DataFrame(cm,
                     index = ['benign', 'defacement','phishing','malware'], 
                     columns = ['benign', 'defacement','phishing','malware'])
plt.figure(figsize=(8,6))
sns.heatmap(cm_df, annot=True,fmt=".1f")
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()

feat_importances = pd.Series(lgb.feature_importances_, index=X_train.columns)
feat_importances.sort_values().plot(kind="barh",figsize=(10, 6))

xgb_c = xgb.XGBClassifier(n_estimators= 100)
xgb_c.fit(X_train,y_train)
y_pred_x = xgb_c.predict(X_test)
print(classification_report(y_test,y_pred_x,target_names=['benign', 'defacement','phishing','malware']))


score = metrics.accuracy_score(y_test, y_pred_x)
print("accuracy:   %0.3f" % score)

cm = confusion_matrix(y_test, y_pred_x)
cm_df = pd.DataFrame(cm,
                     index = ['benign', 'defacement','phishing','malware'], 
                     columns = ['benign', 'defacement','phishing','malware'])
plt.figure(figsize=(8,6))
sns.heatmap(cm_df, annot=True,fmt=".1f")
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()

feat_importances = pd.Series(xgb_c.feature_importances_, index=X_train.columns)
feat_importances.sort_values().plot(kind="barh",figsize=(10, 6))

def main(url):
    
    status = []
    
    status.append(having_ip_address(url))
    status.append(abnormal_url(url))
    status.append(count_dot(url))
    status.append(count_www(url))
    status.append(count_atrate(url))
    status.append(no_of_dir(url))
    status.append(no_of_embed(url))
    
    status.append(shortening_service(url))
    status.append(count_https(url))
    status.append(count_http(url))
    
    status.append(count_per(url))
    status.append(count_ques(url))
    status.append(count_hyphen(url))
    status.append(count_equal(url))
    
    status.append(url_length(url))
    status.append(hostname_length(url))
    status.append(suspicious_words(url))
    status.append(digit_count(url))
    status.append(letter_count(url))
    status.append(fd_length(url))
    tld = get_tld(url,fail_silently=True)
      
    status.append(tld_length(tld))
    
    
    

    return status

def get_prediction_from_url(test_url):
    features_test = main(test_url)
    # Due to updates to scikit-learn, we now need a 2D array as a parameter to the predict function.
    features_test = np.array(features_test).reshape((1, -1))

    

    pred = lgb.predict(features_test)
    if int(pred[0]) == 0:
        
        res="SAFE"
        return res
    elif int(pred[0]) == 1.0:
        
        res="DEFACEMENT"
        return res
    elif int(pred[0]) == 2.0:
        res="PHISHING"
        return res
        
    elif int(pred[0]) == 3.0:
        
        res="MALWARE"
        return res

urls = ['titaniumcorporate.co.za','en.wikipedia.org/wiki/North_Dakota']
for url in urls:
     print(get_prediction_from_url(url))















