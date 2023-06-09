import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import pickle
import random
from sklearn.metrics import accuracy_score,f1_score,classification_report
from sklearn.neighbors import KNeighborsClassifier

from tqdm import tqdm
import sys
import os
import warnings
import platform

if not sys.warnoptions:
	warnings.simplefilter("ignore")							#1
	os.environ["PYTHONWARNINGS"] = "ignore"					#2

	warnings.filterwarnings('ignore', 'Solver terminated early.*')		#3
	# warnings.filterwarnings('ignore', category="ConvergenceWarning")	#4
from datetime import datetime

# 데이터 읽기
# for mac shell file's path
if platform.system() == 'Darwin':
    train = pd.read_csv('/Users/jeonmincheol/2023-1/ML1_final_project/2023_ML_프로젝트_데이터/Train.csv')
    test = pd.read_csv('/Users/jeonmincheol/2023-1/ML1_final_project/2023_ML_프로젝트_데이터/Test.csv')

elif platform.system() == 'Windows':
# for window batch file's path
    train = pd.read_csv('C:/Users/alseh/Desktop/jeonmincheol/2023/2023-1/binary_classification/2023_ML_프로젝트_데이터/Train.csv')
    test = pd.read_csv('C:/Users/alseh/Desktop/jeonmincheol/2023/2023-1/binary_classification/2023_ML_프로젝트_데이터/Test.csv')

# index drop
train = train.drop('Index',axis=1)
test = test.drop('Index',axis=1)
# train용 데이터의 독립변수와 종속변수의 분리
X = train.loc[:,:'X999']
y = train.iloc[:,-1]

# score 를 보기 위해 split, 제출용은 Full Train으로 제작할 것
def test_smote_based_ensemble_model():
    # score 를 보기 위해 split, 제출용은 Full Train으로 제작할 것
    m = datetime.now().month
    d = datetime.now().day

    i = random.randint(1,1000)
    j = random.randint(1,1000)
    k = random.randint(1,1000)
    l = random.randint(1,1000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=j)
    smote = SMOTE(random_state=i)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # 앙상블 방식
    classifiers = [
        RandomForestClassifier(n_estimators=30,random_state=k),
        LogisticRegression(max_iter=20),
        GradientBoostingClassifier(n_estimators=30),
        DecisionTreeClassifier(max_depth=20),
        GaussianNB(),
        # SVC(kernel="rbf")
        # AdaBoostClassifier(n_estimators = 30),
        XGBClassifier(objective='binary:logistic'),
        MLPClassifier(random_state=l)
    ]
    result = random.sample(classifiers,3)

    a,b,c = result[0],result[1],result[2]

    weights1,weights2,weights3 = random.randint(1,6),random.randint(1,6),random.randint(1,6)
    model = VotingClassifier(estimators=[('1',a),('2',b),('3',c)],voting = 'soft',weights=[weights1,weights2,weights3])
    ensemble_train = model.fit(X_train_resampled,y_train_resampled)

    ensemble_pred = model.predict(X_test)
    score = f1_score(y_test,ensemble_pred,pos_label='positive')
    if round(score,2) >= 0.52:
        with open('model/smote_test_{}_{}월_{}일_({},{},{}).pkl'.format(round(score,4),m,d,a,b,c),'wb') as f:
            pickle.dump(model,f)
    return round(score,4)

if __name__ == "__main__":
    for i in tqdm(range(200)):
        a = test_smote_based_ensemble_model()
        if a > 0.5:
            print(a)