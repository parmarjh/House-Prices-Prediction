try:
    import warnings
    warnings.filterwarnings("ignore")
    import pandas as pd
    import numpy as np
    import xgboost as xgb
    from sklearn.preprocessing import RobustScaler
    import time
except Exception as ex:
    print("Could not import libraries please check if the numpy, pandas, sklearn, xgboost is installed in your env, error occured due to {}".format(ex))
    raise ex

start = time.time()
np.random.seed(3064) # set the random seed

out_file1_path ="mysubmission1.txt"
out_file2_path ="mysubmission2.txt"

#method to categorise each column to one of its datatype
def split_column_on_datatype(input_df):
    categorical = [var for var in input_df.columns if input_df[var].dtype=='O'] # get all categorical variable
    numerical = [var for var in input_df.columns if input_df[var].dtype!='O'] # get all numerical variable
    discrete = []
    continous=[]
    for var in numerical:
        if len(input_df[var].unique())<15:
            discrete.append(var)
        else:
            continous.append(var)
    
    discrete.append("Year_Built")
    discrete.append("Year_Remod_Add")
    discrete.remove("BsmtFin_SF_1")
    
    continous.remove("Year_Built")
    continous.remove("Year_Remod_Add")
    continous.append("BsmtFin_SF_1")
    
    ordinal=["Overall_Qual", "Overall_Cond", "Exter_Qual", "Exter_Cond","Bsmt_Qual","Bsmt_Cond", "Heating_QC","Kitchen_Qual","Fireplace_Qu","Garage_Qual","Garage_Cond",
         "Pool_QC","Lot_Shape","Land_Slope","Bsmt_Exposure","BsmtFin_Type_1","BsmtFin_Type_2","Electrical","Functional",
        "Paved_Drive","Fence"]
    nominal =[col for col in categorical if col not in ordinal]

    return categorical, numerical, discrete, continous, ordinal, nominal

#----method to encode the ordinal data based upon the levels
def integer_encoding(df_train, df_test):
    update_dict={"Bsmt_Cond" : {"No_Basement" : 0, "Poor" : 1, "Fair" : 2, "Typical": 3, "Good" : 4, "Excellent" : 5},
                       "Bsmt_Exposure" : {"No_Basement":0,"No" : 1, "Mn" : 2, "Av": 3, "Gd" : 4},
                       "BsmtFin_Type_1" : {"No_Basement" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 
                                         "ALQ" : 5, "GLQ" : 6},
                       "BsmtFin_Type_2" : {"No_Basement" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 
                                         "ALQ" : 5, "GLQ" : 6},
                       "Bsmt_Qual" : {"No_Basement" : 0, "Poor" : 1, "Fair" : 2, "Typical": 3, "Good" : 4, "Excellent" : 5},
                       "Exter_Cond" : {"Poor" : 1, "Fair" : 2, "Typical": 3, "Good": 4, "Excellent" : 5},
                       "Exter_Qual" : {"Poor" : 1, "Fair" : 2, "Typical": 3, "Good": 4, "Excellent" : 5},
                       "Fireplace_Qu" : {"No_Fireplace" : 0, "Poor" : 1, "Fair" : 2, "Typical" : 3, "Good" : 4, "Excellent" : 5},
                       "Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5, 
                                       "Min2" : 6, "Min1" : 7, "Typ" : 8},
                       "Garage_Cond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "Garage_Qual" : {"No_Garage" : 0, "Poor" : 1, "Fair" : 2, "Typical" : 3, "Good" : 4, "Excellent" : 5},
                       "Heating_QC" : {"Poor" : 1, "Fair" : 2, "Typical": 3, "Good": 4, "Excellent" : 5},
                       "Kitchen_Qual" : {"Poor" : 1, "Fair" : 2, "Typical": 3, "Good": 4, "Excellent" : 5},
                       "Land_Slope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3},
                       "Lot_Shape" : {"Irregular" : 1, "Moderately_Irregular" : 2, "Slightly_Irregular" : 3, "Regular" : 4},
                       "Paved_Drive" : {"Dirt_Gravel" : 0, "Paved" : 1, "Partial_Pavement" : 2},
                       "Pool_QC" : {"No_Pool" : 0, "Fair" : 1, "Typical" : 2, "Good" : 3, "Excellent" : 4},
                       "Overall_Qual":{"Very_Excellent": 10, "Excellent":9,"Very_Good":8, "Good":7, "Above_Average":6, "Average": 5, "Below_Average":4, "Fair": 3,"Poor":2, "Very_Poor":1},
                       "Overall_Cond" :{"Very_Excellent": 10, "Excellent":9,"Very_Good":8, "Good":7, "Above_Average":6, "Average": 5, "Below_Average":4, "Fair": 3,"Poor":2, "Very_Poor":1},
                       "Electrical ": {"Unknown": 0,"SBrkr ": 1,"FuseA": 2, "FuseF": 3, "FuseP": 4,"Mix": 5 },
                       "Fence":{"No_Fence":0, "Minimum_Wood_Wire":1,"Good_Wood":2,"Minimum_Privacy":3,"Good_Privacy":4},
                      }
    try:
        df_train = df_train.replace(update_dict)
        df_train["Electrical"].replace({"Unknown": 0,"SBrkr": 1,"FuseA": 2, "FuseF": 3, "FuseP": 4,"Mix": 5 }, inplace=True)
        df_test = df_test.replace(update_dict)
        df_test["Electrical"].replace({"Unknown": 0,"SBrkr": 1,"FuseA": 2, "FuseF": 3, "FuseP": 4,"Mix": 5 }, inplace=True)
        return df_train, df_test
    except Exception as ex:
        print("Error encoding the ordinal data due to {}".format(ex))
        raise ex

#---method to convert categorical data to discrete
def peroform_discretisation(train_df, test_df):
    try:
        Age_range = train_df['Year_Built'].max() - train_df['Year_Built'].min()
        min_value = int(np.floor(train_df['Year_Built'].min()))
        max_value = int(np.ceil(train_df['Year_Built'].max()))
         
        # let's round the bin width
        inter_value = int(np.round(Age_range/9))
        
        intervals = [i for i in range(min_value, max_value+inter_value, inter_value)]
        labels=[]
        for i in range(1, len(intervals)):
            labels.append(i)
        train_df['Year_Built'] = pd.cut(x = train_df['Year_Built'], bins=intervals, labels=labels, include_lowest=True)
        test_df['Year_Built'] = pd.cut(x = test_df['Year_Built'], bins=intervals, labels=labels, include_lowest=True)
        train_df['Year_Built']=train_df['Year_Built'].astype('int64', copy=False,errors="ignore")
        test_df['Year_Built']=test_df['Year_Built'].astype('int64', copy=False,errors="ignore")
        
        
        Age_range = train_df['Year_Remod_Add'].max() - train_df['Year_Remod_Add'].min()
        min_value = int(np.floor(train_df['Year_Remod_Add'].min()))
        max_value = int(np.ceil(train_df['Year_Remod_Add'].max()))
         
        # let's round the bin width
        inter_value = int(np.round(Age_range/5))
        
        intervals = [i for i in range(min_value, max_value+inter_value, inter_value)]
        labels=[]
        for i in range(1, len(intervals)):
            labels.append(i)
        
        train_df['Year_Remod_Add'] = pd.cut(x = train_df['Year_Remod_Add'], bins=intervals, labels=labels, include_lowest=True)
        test_df['Year_Remod_Add'] = pd.cut(x = test_df['Year_Remod_Add'], bins=intervals, labels=labels, include_lowest=True)
        train_df['Year_Remod_Add']=train_df['Year_Remod_Add'].astype('int64', copy=False,errors="ignore")
        test_df['Year_Remod_Add']=test_df['Year_Remod_Add'].astype('int64', copy=False,errors="ignore")
        return train_df, test_df
    except Exception as ex:
        print("Error occured in discretizing data due to {}".format(ex))
        raise ex

#-----encdoding of nominal data
def handle_nominal_variable_encoding(train_df, test_df, nominal):
    try:
        train_objs_num = len(train_df)
        trainY= train_df[["Sale_Price"]]
        dataset = pd.concat(objs=[train_df.drop(["Sale_Price"], axis=1), test_df], axis=0)
        
        #MS_Subclass
        top_5 = [x for x in dataset['MS_SubClass'].value_counts().sort_values(ascending=False).head(5).index]
        for label in top_5:
            dataset[label] = np.where(dataset['MS_SubClass']==label, 1, 0)
        
        dataset.drop("MS_SubClass", axis=1, inplace=True)
        
        nominal.remove("MS_SubClass")
        
        #MS_Zoning
        top_3 = [x for x in dataset['MS_Zoning'].value_counts().sort_values(ascending=False).head(3).index]
        for label in top_3:
            dataset[label] = np.where(dataset['MS_Zoning']==label, 1, 0)
        
        dataset.drop("MS_Zoning", axis=1, inplace=True)
        
        nominal.remove("MS_Zoning")
        
        #Lot_Config
        top_3 = [x for x in dataset['Lot_Config'].value_counts().sort_values(ascending=False).head(3).index]
        for label in top_3:
            dataset[label] = np.where(dataset['Lot_Config']==label, 1, 0)
        
        dataset.drop("Lot_Config", axis=1, inplace=True)
        
        nominal.remove("Lot_Config")
        
        #Neighborhood 
        top_8 = [x for x in dataset['Neighborhood'].value_counts().sort_values(ascending=False).head(8).index]
        for label in top_8:
            dataset[label] = np.where(dataset['Neighborhood']==label, 1, 0)
        
        dataset.drop("Neighborhood", axis=1, inplace=True)
        nominal.remove("Neighborhood")
        
        #Condition_1
        top_3 = [x for x in dataset['Condition_1'].value_counts().sort_values(ascending=False).head(3).index]
        for label in top_3:
            dataset[label] = np.where(dataset['Condition_1']==label, 1, 0)
        
        dataset.drop("Condition_1", axis=1, inplace=True)
        nominal.remove("Condition_1")
        
        #House_Style 
        top_3 = [x for x in dataset['House_Style'].value_counts().sort_values(ascending=False).head(3).index]
        for label in top_3:
            dataset[label] = np.where(dataset['House_Style']==label, 1, 0)
        
        dataset.drop("House_Style", axis=1, inplace=True)
        nominal.remove("House_Style")
        
        #Roof_Style
        top_2 = [x for x in dataset['Roof_Style'].value_counts().sort_values(ascending=False).head(2).index]
        for label in top_2:
            dataset[label] = np.where(dataset['Roof_Style']==label, 1, 0)
        
        dataset.drop("Roof_Style", axis=1, inplace=True)
        nominal.remove("Roof_Style")
        
        #Exterior_1st 
        top_5 = [x for x in dataset['Exterior_1st'].value_counts().sort_values(ascending=False).head(5).index]
        for label in top_5:
            dataset[label] = np.where(dataset['Exterior_1st']==label, 1, 0)
        
        dataset.drop("Exterior_1st", axis=1, inplace=True)
        nominal.remove("Exterior_1st")
        
        #Exterior_2nd 
        top_5 = [x for x in dataset['Exterior_2nd'].value_counts().sort_values(ascending=False).head(5).index]
        for label in top_5:
            dataset[label] = np.where(dataset['Exterior_2nd']==label, 1, 0)
        
        dataset.drop("Exterior_2nd", axis=1, inplace=True)
        nominal.remove("Exterior_2nd")
        
        #Mas_Vnr_Type
        top_3 = [x for x in dataset['Mas_Vnr_Type'].value_counts().sort_values(ascending=False).head(3).index]
        for label in top_3:
            dataset[label] = np.where(dataset['Mas_Vnr_Type']==label, 1, 0)
        
        dataset.drop("Mas_Vnr_Type", axis=1, inplace=True)
        nominal.remove("Mas_Vnr_Type")
        
        #Foundation
        top_3 = [x for x in dataset['Foundation'].value_counts().sort_values(ascending=False).head(3).index]
        for label in top_3:
            dataset[label] = np.where(dataset['Foundation']==label, 1, 0)
        
        dataset.drop("Foundation", axis=1, inplace=True)
        nominal.remove("Foundation")
        
        #Heating
        top_2 = [x for x in dataset['Heating'].value_counts().sort_values(ascending=False).head(2).index]
        for label in top_2:
            dataset[label] = np.where(dataset['Heating']==label, 1, 0)
        
        dataset.drop("Heating", axis=1, inplace=True)
        nominal.remove("Heating")
        
        #Garage_Type
        top_5 = [x for x in dataset['Garage_Type'].value_counts().sort_values(ascending=False).head(5).index]
        for label in top_5:
            dataset[label] = np.where(dataset['Garage_Type']==label, 1, 0)
        
        dataset.drop("Garage_Type", axis=1, inplace=True)
        nominal.remove("Garage_Type")
        
        #Misc_Feature
        top_2 = [x for x in dataset['Misc_Feature'].value_counts().sort_values(ascending=False).head(2).index]
        for label in top_2:
            dataset[label] = np.where(dataset['Misc_Feature']==label, 1, 0)
        
        dataset.drop("Misc_Feature", axis=1, inplace=True)
        nominal.remove("Misc_Feature")
        
        #Sale_Type
        top_3 = [x for x in dataset['Sale_Type'].value_counts().sort_values(ascending=False).head(3).index]
        for label in top_3:
            dataset[label] = np.where(dataset['Sale_Type']==label, 1, 0)
        
        dataset.drop("Sale_Type", axis=1, inplace=True)
        nominal.remove("Sale_Type")
        
        #Sale_Condition
        top_3 = [x for x in dataset['Sale_Condition'].value_counts().sort_values(ascending=False).head(3).index]
        for label in top_3:
            dataset[label] = np.where(dataset['Sale_Condition']==label, 1, 0)
        
        dataset.drop("Sale_Condition", axis=1, inplace=True)
        nominal.remove("Sale_Condition")

    
        dataset_preprocessed = pd.get_dummies(dataset)
        train_df = dataset_preprocessed[:train_objs_num]
        test_df = dataset_preprocessed[train_objs_num:]
        train_df=pd.concat([train_df, trainY], axis=1)
        return train_df, test_df, nominal
    except Exception as ex:
        print("Error occured in processing nominal data due to {}".format(ex))
        raise ex


# fix the issue of outliers
def clip_outliers(train_df,variables):
    try:
        for var in variables:
            IQR = train_df[[var]].quantile(0.75) - train_df[[var]].quantile(0.25)
            Lower_fence = float(train_df[[var]].quantile(0.25) - (IQR * 3))
            Upper_fence = float(train_df[[var]].quantile(0.75) + (IQR * 3))
            train_df[var].clip(Lower_fence, Upper_fence, inplace=True)
            return train_df
    except Exception as ex:
        print("Error occured in clipping outliers in data due to {}".format(ex))
        raise ex


#---method to check colinerity
def correlation(dataset, threshold):
    try:
        col_corr = set()  # Set of all the names of correlated columns
        corr_matrix = dataset.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                    colname = corr_matrix.columns[i]  # getting the name of column
                    col_corr.add(colname)
        return col_corr
    except Exception as ex:
        print("Error occured in identifying colineraity in data due to {}".format(ex))
        raise ex
    
    
#---method to perform feature scaling
def standardise_data(train_df, test_df):
    try:
        scaler = RobustScaler().fit(train_df)
        scaled_train_X=scaler.transform(train_df)
        scaled_test_X=scaler.transform(test_df)
        return scaled_train_X,scaled_test_X
    except Exception as ex:
        print("Error occured while standardizing the data due to {}".format(ex))
        raise ex



def perform_Xgbbost1_Regression(train_df, test_df, train_target,index):
    global out_file1_path
    try:
        xgb_reg_model2 = xgb.XGBRegressor(
                     colsample_bytree=0.3,
                     gamma=0.0,
                     learning_rate=0.02,
                     max_depth=4,
                     min_child_weight=1.5,
                     n_estimators=8000,                                                                  
                     reg_alpha=0.9,
                     reg_lambda=0.6,
                     subsample=0.2,
                     silent=1)
        xgb_reg_model2.fit(train_df, train_target)
        

        pred_test= xgb_reg_model2.predict(test_df)
        

        pred_test_original= pd.DataFrame(np.round(np.expm1(pred_test),1), index=index)
        pred_test_original.columns=['Sale_Price']
        pred_test_original.to_csv(out_file1_path, header=True, index=True, sep=',')
        print("Prediction from model 1 persisted to disk as mysubmission1.txt")
        return pred_test
    except Exception as ex:
        print("Error occured in running Xgboost1 due to {}".format(ex))
        raise ex



def perform_xgboosting2(train_df, test_df, train_target, index):
    global out_file2_path
    try:
        xgb_reg_model2 = xgb.XGBRegressor(
                     colsample_bytree=0.2,
                     gamma=0.0,
                     learning_rate=0.01,
                     max_depth=4,
                     min_child_weight=1.5,
                     n_estimators=8000,                                                                  
                     reg_alpha=0.9,
                     reg_lambda=0.6,
                     subsample=0.2,
                     silent=1)
        
        xgb_reg_model2.fit(train_df, train_target)
        
        y_pred_test = xgb_reg_model2.predict(test_df)
        

        pred_test_original= pd.DataFrame(np.round(np.expm1(y_pred_test),1), index=index)
        pred_test_original.columns=['Sale_Price']
        pred_test_original.to_csv(out_file2_path, header=True, index=True, sep=',')
        print("Prediction from model 2 persisted to disk as mysubmission2.txt")
        return y_pred_test
    except Exception as ex:
        print("Error occured in running xgboost regression 2 due to {}".format(ex))
        raise ex

try:
    try:
        #---------read in the training file
        train_df= pd.read_csv("train.csv", header=0, index_col="PID")
        train_df.drop("Garage_Yr_Blt", axis=1, inplace=True)
        train_df=train_df[train_df.columns[train_df.isnull().mean() < 0.5]]
        train_df=train_df.fillna(train_df.mean())
        train_df.dropna(inplace=True)
        trainY= train_df[["Sale_Price"]]
        trainX=train_df.drop(["Sale_Price","Condition_2","Utilities","Roof_Matl","Latitude", "Longitude"], axis=1)
        
        #-----read in the test file
        test_df=pd.read_csv("test.csv", header=0, index_col="PID")
        test_df.drop("Garage_Yr_Blt", axis=1, inplace=True)
        test_df.fillna(0)
        testX=test_df.drop(["Condition_2","Utilities","Roof_Matl","Latitude", "Longitude"], axis=1)
    
        Combine_df_X= pd.concat((trainX, testX))#combine the 2 df into one
        
        trainY=np.log1p(trainY) # transform the target variable
        
        full_train= pd.concat([trainX, trainY], axis=1) # combine the prediction and target for easy operation and indexing
        full_test=testX.copy()
    except Exception as ex:
        print("Error in importing data into memory due to {}".format(ex))
        raise ex
    
    
    categorical, numerical, discrete, continous, ordinal, nominal=split_column_on_datatype(trainX)
    
    try:
        #---remove outlier from Gr_Liv_Area
        out=full_train.Gr_Liv_Area.sort_values(ascending=False)[:3].index
        full_train.drop(out, axis=0, inplace=True)
        
        #---remove outlier from Total_Bsmt_SF
        out=full_train.Total_Bsmt_SF.sort_values(ascending=False)[:3].index
        full_train.drop(out, axis=0, inplace=True)
    except Exception as ex:
        print("Error in fixing Gr_Liv_Area and Total_Bsmt_SF due to {}".format(ex))
        raise ex
    
    full_train, full_test = integer_encoding(full_train, full_test) # encode the ordinal data
    
    full_train, full_test =peroform_discretisation(full_train, full_test) # discretisize the year variable
    
    full_train, full_test, nominal =handle_nominal_variable_encoding(full_train, full_test, nominal) # encode nominal variable
    
    full_train=clip_outliers(full_train,continous+discrete) # fix outliers
    
    try:
        corr_features = correlation(full_train.drop(["Sale_Price"], axis=1).copy(), 0.7) # identify colineraity
        full_train.drop(labels=corr_features, axis=1, inplace=True)
        full_test.drop(labels=corr_features, axis=1, inplace=True)
        
        #----remove contant features
        constant_features = [feat for feat in full_train.columns if full_train[feat].std() <= 0.1]
        full_train.drop(labels=constant_features, axis=1, inplace=True)
        full_test.drop(labels=constant_features, axis=1, inplace=True)
        
        #----remove any missing value if introduced
        full_test=full_test.fillna(0)
        full_train=full_train.fillna(0)
    
    except Exception as ex:
        print("Error occured during feature_selection due to {}".format(ex))
        raise ex
    #-------segment the predictor and target
    trainY= full_train[["Sale_Price"]] # get the target column
    trainX=full_train.drop(["Sale_Price"], axis=1) # get the predictors
    testX=full_test.copy() # get the test data
    
    
    scaled_train_X, scaled_test_X=standardise_data(trainX, testX) # standardise the features
    
    #----start modelling
    perform_Xgbbost1_Regression(scaled_train_X, scaled_test_X, trainY,testX.index) # model1
    print("-----")
    perform_xgboosting2(scaled_train_X, scaled_test_X, trainY,testX.index) # model 2
    print("-----")
    end = time.time()
    print("Script executed succesfully in {} seconds ".format(end - start))
except Exception as ex:
        print("Execution aborted due to {}".format(ex))
        raise ex
