##############################
# DIABETES FEATURE ENGINEERING
##############################

"""
Problem:
It is desired to develop a machine learning model that can predict whether people 
have diabetes when their characteristics are specified. You are expected to perform 
the necessary data analysis and feature engineering steps before developing the model.

Dataset Story:
The dataset is part of the large dataset held at the National Institutes of 
Diabetes-Digestive-Kidney Diseases in the USA. Data used for diabetes research on 
Pima Indian women aged 21 and over living in Phoenix, the 5th largest city in the State
of Arizona in the USA. It consists of 768 observations and 8 numerical independent variables.
The target variable is specified as "outcome"; 1 indicates positive diabetes test result, 
0 indicates negative.

Pregnancies:              Number of pregnancies
Glucose:                  Glucose
BloodPressure:            Blood pressure (Diastolic(Small Blood Pressure))
SkinThickness:            Skin Thickness
Insulin:                  Insulin.
BMI:                      Body mass index.
DiabetesPedigreeFunction: A function that calculates our probability of having diabetes based
                          on our ancestry.
Age:                      Age (years)
Outcome:                  Information whether the person has diabetes or not. 
                          Have the disease (1) or not (0)
"""

##############################
# TASKS
##############################
# TASK 1: DISCOVERY DATA ANALYSIS
    # Step 1: Examine the overall picture.
    # Step 2: Capture the numeric and categorical variables.
    # Step 3: Analyze the numerical and categorical variables.
    # Step 4: Perform target variable analysis. (The average of the target variable according to the categorical variables, the average of the numerical variables according to the target variable)
    # Step 5: Analyze outliers.
    # Step 6: Perform a missing observation analysis.
    # Step 7: Perform correlation analysis.

# TASK 2: FEATURE ENGINEERING
    # Step 1: Take necessary actions for missing and outlier values. There are no missing observations in the data set
    # but Glucose, Insulin etc. Observation units containing a value of 0 in the variables may represent the missing value.
    # For example; a person's glucose or insulin value will not be 0. Taking this situation into account, zero values
    # assign the values ​​as NaN and then add the missing values
    # you can apply operations.
    # Step 2: Create new variables.
    # Step 3: Perform the encoding operations.
    # Step 4: Standardize for numeric variables.
    # Step 5: Create the model.

##############################

# Required Library and Functions
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)



###################################################################
########         TASK 1: DISCOVERY DATA ANALYSIS         ##########
###################################################################

# Step 1: Examine the overall picture.
##################################
# GENERAL PICTURE
##################################

def load_diabetes():
    data = pd.read_csv("D:\\Users\\mstff\\CourseProject\\Miuul_Python_Machine_Learning\\pythonProject\\datasets\\diabetes.csv")
    return data

df = load_diabetes()

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Info #####################")
    print(dataframe.info())
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("################ Null Values ##################")
    print(dataframe.isnull().values.any())
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)



#########################################################################################
# Step 2: Capture the numeric and categorical variables.

##################################
# CAPTURE OF NUMERICAL AND CATEGORY VARIABLES
##################################

# grab_col_names Function of Analysis Numerical and Categroical Variables
def grab_col_names(dataframe, cat_th=10,  car_th=20):
    """
    It gives the names of categorical, numerical and categorical but cardinal variables in the data set.
    Note: Categorical variables with numerical appearance are also included in categorical variables.
    Parameters
    ----------
    dataframe: dataframe
        it is the dataframe from which variable names will be taken.
    cat_th: int, float
        class threshold for numeric but categorical variables
    car_th: int, float
        class threshold for categorical but cardinal variables

    Returns
    -------
    cat_cols: list
        Categorical variable list
    num_cols: list
        Numerical variable list
    cat_but_car: list
        Categorical view cardinal variable list

    Notes
    ------
    cat_cols + num_cols + cat_but_car = total number of variables
    num_but_cat is inside cat_cols.

    """

    # cat_cols, cat_but_car: Analysis of Categorical Variables
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols Analysis of Numerical Variables
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]


    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols), cat_cols}')
    print(f'num_cols: {len(num_cols), num_cols}')
    print(f'cat_but_car: {len(cat_but_car), cat_but_car}')
    print(f'num_but_cat: {len(num_but_cat), num_but_cat}')

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)
print(cat_cols, num_cols, cat_but_car)


#########################################################################################
# Step 3: Analyze the numerical and categorical variables.

##################################
# ANALYSIS OF CATEGORY VARIABLES
##################################

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df, col, plot=True)

##################################
# ANALYSIS OF NUMERICAL VARIABLES
##################################

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T, end="\n\n\n")

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(df, col, plot=True)


#########################################################################################
# Step 4: Perform target variable analysis. (The average of the target variable according to the
# categorical variables, the average of the numerical variables according to the target variable)

#############################################
# Analysis of Target Variable
#############################################

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Outcome", col)


def target_sum_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN" :dataframe.groupby(categorical_col)[target].mean()}))

for col in cat_cols:
    target_sum_with_cat(df, "Outcome", col)


#########################################################################################
# Step 5: Analyze outliers.

#############################################
# OUTLIER THRESHOLDS FUNC  
#############################################
check_df(df) # q1, q3 belirleyebilmek için genel olarak tekrar bakıyorum.

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

for col in num_cols:
    print(col, "-->", outlier_thresholds(df, col))


#############################################
# CHECK OUTLIER  FUNC 
#############################################

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col, "-->", check_outlier(df, col))

# output: Insulin --> True

#############################################
# grab_outliers
#############################################

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

for col in num_cols:
    print(col, "-->", grab_outliers(df, col, index=True))    


#########################################################################################
# Step 6: Perform a missing observation analysis.

#############################################
# Missing Values
#############################################

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

na_columns = missing_values_table(df, na_name=True)

# Missing value does not appear. The initial check_df query showed us that there are no null values.
# Then there is a possibility that the null values ​​might be padded with zero.

# Let's look at correlation, base model and importance without doing anything.

#########################################################################################
# Step 7: Perform correlation analysis

##################################
# CORRELATION
##################################

df.corr()

# CORRELATION MATRIX
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()


##################################
# BASE MODEL
##################################

y = df["Outcome"]
X = df.drop("Outcome", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred, y_test), 3)}")
print(f"Precision: {round(precision_score(y_pred, y_test), 2)}")
print(f"F1: {round(f1_score(y_pred, y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred, y_test), 2)}")


# Accuracy: 0.77
# Recall: 0.706 
# Precision: 0.59
# F1: 0.64
# Auc: 0.75


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_model, X)


###################################################################
########           TASK 2: FEATURE ENGINEERING           ##########
###################################################################

# Step 1: Take necessary actions for missing and outlier values. There are no missing observations in the data set
# but Glucose, Insulin etc. Observation units containing a value of 0 in the variables may represent the missing value.
# For example; a person's glucose or insulin value will not be 0. Taking this situation into account, zero values
# assign the values ​​as NaN and then add the missing values
# you can apply operations.

##################################
# MISSING VALUE ANALYSIS
##################################
# It is known that variable values other than Pregnancies and Outcome cannot be 0 in a human.
# Therefore, an action decision should be taken regarding these values. Values that are 0 can be assigned NaN.

zero_columns = [col for col in df.columns if (df[col].min() == 0 and col not in ["Pregnancies", "Outcome"])]

# We went to each of the variables with 0 in the observation units and changed the observation values ​​containing 0 with NaN.for col in zero_columns:
for col in zero_columns:
    df[col] = np.where(df[col] == 0, np.nan, df[col])


# Missing Observation Analysis
df.isnull().sum()

na_columns = missing_values_table(df, na_name=True)

###########################################################################
# Examining the Relationship of Missing Values ​​with the Dependent Variable
###########################################################################
def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


missing_vs_target(df, "Outcome", na_columns)


# Filling in Missing Values
for col in zero_columns:
    df.loc[df[col].isnull(), col] = df[col].median()

df.isnull().sum()


##################################
#  OUTLIERS ANALYSIS
##################################
for col in num_cols:
    print(col, " outlier_thresholds-->", outlier_thresholds(df, col))
    print(col, " check_outlier-->", check_outlier(df, col))
    print(col, " grab_outliers-->", grab_outliers(df, col, index=True))    
    print("############", end="\n\n")


def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# Outlier Analysis and Suppression Process
for col in df.columns:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

for col in df.columns:
    print(col, check_outlier(df, col))


##################################
# Step 2: Create new variables

##################################
# CREATE NEW VARIABLES
##################################

# Creating a new age variable by categorizing the age variable
df.loc[(df["Age"] >= 21) & (df["Age"] < 50), "NEW_AGE_CAT"] = "mature"
df.loc[(df["Age"] >= 50), "NEW_AGE_CAT"] = "senior"
# BMI below 18.5 is underweight, between 18.5 and 24.9 is normal, between 24.9 and 29.9 is overweight and over 30 is obese
df['NEW_BMI'] = pd.cut(x=df['BMI'], bins=[0, 18.5, 24.9, 29.9, 100],
                       labels=["Underweight", "Healthy", "Overweight", "Obese"])

# Convert glucose value to categorical variable
df["NEW_GLUCOSE"] = pd.cut(x=df["Glucose"], bins=[0, 140, 200, 300], labels=["Normal", "Prediabetes", "Diabetes"])

# Creating a categorical variable by considering age and body mass index together
df.loc[(df["BMI"] < 18.5) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "underweightmature"
df.loc[(df["BMI"] < 18.5) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "underweightsenior"
df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & (
        (df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "healthymature"
df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "healthysenior"
df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & (
        (df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "overweightmature"
df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "overweightsenior"
df.loc[(df["BMI"] > 18.5) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "obesemature"
df.loc[(df["BMI"] > 18.5) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "obesesenior"

# Creating a categorical variable by considering age and glucose values ​​together
df.loc[(df["Glucose"] < 70) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "lowmature"
df.loc[(df["Glucose"] < 70) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "lowsenior"
df.loc[((df["Glucose"] >= 70) & (df["Glucose"] < 100)) & (
        (df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "normalmature"
df.loc[((df["Glucose"] >= 70) & (df["Glucose"] < 100)) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "normalsenior"
df.loc[((df["Glucose"] >= 100) & (df["Glucose"] <= 125)) & (
        (df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "hiddenmature"
df.loc[((df["Glucose"] >= 100) & (df["Glucose"] <= 125)) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "hiddensenior"
df.loc[(df["Glucose"] > 125) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "highmature"
df.loc[(df["Glucose"] > 125) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "highsenior"


# Deriving Categorical Variable with Insulin Value
def set_insulin(dataframe, col_name="Insulin"):
    if 16 <= dataframe[col_name] <= 166:
        return "Normal"
    else:
        return "Abnormal"


df["NEW_INSULIN_SCORE"] = df.apply(set_insulin, axis=1)

df["NEW_GLUCOSE*INSULIN"] = df["Glucose"] * df["Insulin"]
df["NEW_GLUCOSE*PREGNANCIES"] = df["Glucose"] * df["Pregnancies"]


df.columns = [col.upper() for col in df.columns]

df.head()
df.shape


##################################
# Step 3: Perform the encoding operations.

##################################
# ENCODING
##################################


## LABEL ENCODING

# Separation of variables according to their types
cat_cols, num_cols, cat_but_car = grab_col_names(df)

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]
binary_cols

for col in binary_cols:
    label_encoder(df, col)


## ONE - HOT ENCODING

# Update process of cat_cols list
cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["OUTCOME"]]
cat_cols

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


df = one_hot_encoder(df, cat_cols, drop_first=True)

df.shape
df.head()


##################################
# Step 4: Standardize for numeric variables.

##################################
# STANDARDIZATION
##################################

num_cols

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df.head()
df.shape


##################################
# Step 5: Create the model.

##################################
# MODELLING
##################################
y = df["OUTCOME"]
X = df.drop("OUTCOME", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred, y_test), 3)}")
print(f"Precision: {round(precision_score(y_pred, y_test), 2)}")
print(f"F1: {round(f1_score(y_pred, y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred, y_test), 2)}")


# Accuracy: 0.79
# Recall: 0.711
# Precision: 0.67
# F1: 0.69
# Auc: 0.77

# Base Model
# Accuracy: 0.77
# Recall: 0.706
# Precision: 0.59
# F1: 0.64
# Auc: 0.75

##################################
# FEATURE IMPORTANCE
##################################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    print(feature_imp.sort_values("Value", ascending=False))
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_model, X)
