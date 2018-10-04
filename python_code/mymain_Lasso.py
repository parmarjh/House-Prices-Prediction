try:
    import warnings
    warnings.filterwarnings("ignore")
    import pandas as pd
    import numpy as np
    from sklearn import preprocessing
    from sklearn.preprocessing import StandardScaler
    import time
except Exception as ex:
    print("Could not import libraries please check if numpy, pandas, sklearn is installed in your env, error occured due to {}".format(ex))
    raise ex

np.random.seed(3064) # set the random seed
start = time.time()


out_file3_path ="mysubmission3.txt"
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


#-----encdoding of nominal data
def handle_nominal_variable_encoding(train_df, test_df):
    try:
        train_objs_num = len(train_df)
        trainY= train_df[["Sale_Price"]]
        dataset = pd.concat(objs=[train_df.drop(["Sale_Price"], axis=1), test_df], axis=0)
        dataset_preprocessed = pd.get_dummies(dataset)
        train_df = dataset_preprocessed[:train_objs_num]
        test_df = dataset_preprocessed[train_objs_num:]
        train_df=pd.concat([train_df, trainY], axis=1)
        return train_df, test_df
    except Exception as ex:
        print("Error occured in processing nominal data due to {}".format(ex))
        raise ex

# fix the issue of outliers
def clip_outliers(train_df):
    try:
        for var in train_df.columns:
            IQR = train_df[[var]].quantile(0.75) - train_df[[var]].quantile(0.25)
            Lower_fence = float(train_df[[var]].quantile(0.25) - (IQR * 2))
            Upper_fence = float(train_df[[var]].quantile(0.75) + (IQR * 2))
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



def calculate_prediction(b0,b, X,Y,testX):
    predTest= b0+np.dot(testX,b)
    pred_test_original= pd.DataFrame(np.round(np.expm1(predTest),1), index=testX.index)
    pred_test_original.columns=['Sale_Price']
    pred_test_original.to_csv(out_file3_path, header=True, index=True, sep=',')
    print("Prediction from model 3 persisted to disk as mysubmission3.txt")


def one_step_lasso(r,x,lam):
   xx = (x*x).sum()
   xr = (r*x).sum()
   b = (np.abs(xr) -lam/2)/xx
   b = np.sign(xr)*np.where(b>0,b,0)
   return b

def standardise_data(X, y):
  scaler_X = StandardScaler()
  transform_X=scaler_X.fit_transform(X)
  scaler_Y=preprocessing.StandardScaler()
  scaler_Y.fit(y)
  transform_Y= y.Sale_Price-scaler_Y.mean_
  return transform_X, transform_Y,scaler_X,scaler_Y


def lasso_using_cordinate_descent(train_X, train_y, lam, iter_n = 100, standardize  = True):
  if standardize:
      X, y,scaler_X,scaler_Y=standardise_data(train_X, train_y)

  p=X.shape[1]
  b=np.zeros(p)
  r=y
  for step in range(0,iter_n):
      for j in range(0,p):
              r = r + X[:,j]*b[j]
              b[j] = one_step_lasso(r,X[:,j],lam)
              r = r - X[:,j]*b[j]
  b= b/np.sqrt(scaler_X.var_)
  b0= scaler_Y.mean_-((b*scaler_X.mean_).sum())
  return([b0, b])


try:
    #---------read in the training file
    train_df= pd.read_csv("train.csv", header=0, index_col="PID")
    train_df.drop("Garage_Yr_Blt", axis=1, inplace=True)
    train_df=train_df[train_df.columns[train_df.isnull().mean() < 0.5]]
    train_df=train_df.fillna(train_df.mean())
    train_df.dropna(inplace=True)
    trainY= train_df[["Sale_Price"]]
    trainX=train_df.drop(["Sale_Price","Condition_2","Utilities","Roof_Matl","Latitude", "Longitude",
                          "Street", "Land_Slope", "Heating","Pool_QC","Misc_Feature", "Low_Qual_Fin_SF", "Three_season_porch",
                          "Pool_Area", "Misc_Val"], axis=1)

    #-----read in the test file
    test_df=pd.read_csv("test.csv", header=0, index_col="PID")
    test_df.drop("Garage_Yr_Blt", axis=1, inplace=True)
    test_df.fillna(0)
    testX=test_df.drop(["Condition_2","Utilities","Roof_Matl","Latitude", "Longitude",
                        "Street","Land_Slope", "Heating", "Pool_QC","Misc_Feature", "Low_Qual_Fin_SF", "Three_season_porch",
                             "Pool_Area", "Misc_Val"], axis=1)



    trainY=np.log1p(trainY) # transform the target variable

    full_train= pd.concat([trainX, trainY], axis=1) # combine the prediction and target for easy operation and indexing
    full_test=testX


    full_train, full_test = integer_encoding(full_train, full_test) # encode the ordinal data
    full_train, full_test =handle_nominal_variable_encoding(full_train, full_test) # encode nominal variable
    full_train=clip_outliers(full_train) # fix outliers
    corr_features = correlation(full_train.drop(["Sale_Price"], axis=1).copy(), 0.8) # identify colineraity
    full_train.drop(labels=corr_features, axis=1, inplace=True)
    full_test.drop(labels=corr_features, axis=1, inplace=True)

    #----remove contant features
    constant_features = [feat for feat in full_train.columns if full_train[feat].std() <=0.1]
    full_train.drop(labels=constant_features, axis=1, inplace=True)
    full_test.drop(labels=constant_features, axis=1, inplace=True)

    #----remove any missing value if introduced
    full_test=full_test.fillna(0)
    full_train=full_train.fillna(0)

    #-------segment the predictor and target
    trainY= full_train[["Sale_Price"]] # get the target column
    trainX=full_train.drop(["Sale_Price"], axis=1) # get the predictors
    testX=full_test # get the test data

    #-----use the most imp variables
    trainX["3*Gr_Liv_Area"] = trainX["Gr_Liv_Area"] ** 3
    testX["3*Gr_Liv_Area"] = testX["Gr_Liv_Area"] ** 3
    trainX["4*Gr_Liv_Area"] = trainX["Gr_Liv_Area"] ** 4
    testX["4*Gr_Liv_Area"] = testX["Gr_Liv_Area"] ** 4
    trainX["2*Gr_Liv_Area"] = trainX["Gr_Liv_Area"] ** 2
    testX["2*Gr_Liv_Area"] = testX["Gr_Liv_Area"] ** 2

    trainX["3*Garage_Cars"] = trainX["Garage_Cars"] ** 3
    testX["3*Garage_Cars"] = testX["Garage_Cars"] ** 3
    trainX["4*Garage_Cars"] = trainX["Garage_Cars"] ** 4
    testX["4*Garage_Cars"] = testX["Garage_Cars"] ** 4
    trainX["2*Garage_Cars"] = trainX["Garage_Cars"] ** 2
    testX["2*Garage_Cars"] = testX["Garage_Cars"] ** 2
    
    trainX["3*Overall_Qual"] = trainX["Overall_Qual"] ** 3
    testX["3*Overall_Qual"] = testX["Overall_Qual"] ** 3
    trainX["4*Overall_Qual"] = trainX["Overall_Qual"] ** 4
    testX["4*Overall_Qual"] = testX["Overall_Qual"] ** 4
    trainX["2*Overall_Qual"] = trainX["Overall_Qual"] ** 2
    testX["2*Overall_Qual"] = testX["Overall_Qual"] ** 2
    
    trainX["3*Lot_Area"] = trainX["Lot_Area"] ** 3
    testX["3*Lot_Area"] = testX["Lot_Area"] ** 3
    trainX["4*Lot_Area"] = trainX["Lot_Area"] ** 4
    testX["4*Lot_Area"] = testX["Lot_Area"] ** 4
    trainX["2*Lot_Area"] = trainX["Lot_Area"] ** 2
    testX["2*Lot_Area"] = testX["Lot_Area"] ** 2
    
    trainX["3*Overall_Cond"] = trainX["Overall_Cond"] ** 3
    testX["3*Overall_Cond"] = testX["Overall_Cond"] ** 3
    trainX["4*Overall_Cond"] = trainX["Overall_Cond"] ** 4
    testX["4*Overall_Cond"] = testX["Overall_Cond"] ** 4
    trainX["2*Overall_Cond"] = trainX["Overall_Cond"] ** 2
    testX["2*Overall_Cond"] = testX["Overall_Cond"] ** 2
    
    trainX["3*Second_Flr_SF"] = trainX["Second_Flr_SF"] ** 3
    testX["3*Second_Flr_SF"] = testX["Second_Flr_SF"] ** 3
    trainX["4*Second_Flr_SF"] = trainX["Second_Flr_SF"] ** 4
    testX["4*Second_Flr_SF"] = testX["Second_Flr_SF"] ** 4
    trainX["2*Second_Flr_SF"] = trainX["Second_Flr_SF"] ** 2
    testX["2*Second_Flr_SF"] = testX["Second_Flr_SF"] ** 2
    
    trainX["3*Year_Built"] = trainX["Year_Built"] ** 3
    testX["3*Year_Built"] = testX["Year_Built"] ** 3
    trainX["4*Year_Built"] = trainX["Year_Built"] ** 4
    testX["4*Year_Built"] = testX["Year_Built"] ** 4
    trainX["2*Year_Built"] = trainX["Year_Built"] ** 2
    testX["2*Year_Built"] = testX["Year_Built"] ** 2
    
    trainX["3*Functional"] = trainX["Functional"] ** 3
    testX["3*Functional"] = testX["Functional"] ** 3
    trainX["4*Functional"] = trainX["Functional"] ** 4
    testX["4*Functional"] = testX["Functional"] ** 4
    trainX["2*Functional"] = trainX["Functional"] ** 2
    testX["2*Functional"] = testX["Functional"] ** 2
    
    trainX["3*Mas_Vnr_Area"] = trainX["Mas_Vnr_Area"] ** 3
    testX["3*Mas_Vnr_Area"] = testX["Mas_Vnr_Area"] ** 3
    trainX["4*Mas_Vnr_Area"] = trainX["Mas_Vnr_Area"] ** 4
    testX["4*Mas_Vnr_Area"] = testX["Mas_Vnr_Area"] ** 4
    trainX["2*Mas_Vnr_Area"] = trainX["Mas_Vnr_Area"] ** 2
    testX["2*Mas_Vnr_Area"] = testX["Mas_Vnr_Area"] ** 2

    b0, b= lasso_using_cordinate_descent(trainX, trainY, 6.31)
    calculate_prediction(b0,b, trainX, trainY, testX)
    end = time.time()
    print("Script executed succesfully in {} seconds ".format(end - start))
except Exception as ex:
        print("Execution aborted due to {}".format(ex))
        raise ex
