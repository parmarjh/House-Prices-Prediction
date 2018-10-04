import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np

#result = pd.DataFrame(index=range(0,10), columns=["model1", "model2"], dtype='float64')
#for i in range(0,10):
#    run_file_path ="run/test"+str(i)+".csv"
#    test_file_run = pd.read_csv(run_file_path, index_col="PID")
#    #print(test_file_run.Sale_Price)
#    
#    out_file_path1 ="sk/Output/mysubmission1_"+str(i)+".txt"
#    out_file_run = pd.read_csv(out_file_path1, index_col="PID")
#    
#    out_file_path2 ="sk/Output/mysubmission2_"+str(i)+".txt"
#    out_file_run2 = pd.read_csv(out_file_path2, index_col="PID")
#    #print(out_file_run.Sale_Price)
#    
#    pre=out_file_run[["Sale_Price"]].astype("float64")
#    true=test_file_run[["Sale_Price"]].astype("float64")
#
#    merge_df=pd.merge(out_file_run[["Sale_Price"]], test_file_run[["Sale_Price"]], left_index=True, right_index=True)
#    rms = np.sqrt(((np.log(merge_df.Sale_Price_y).astype("float64")-np.log(merge_df.Sale_Price_x).astype("float64"))**2).mean())
#    result.loc[i,"model1"]= round(rms,4)
#    
#    
#    merge_df=pd.merge(out_file_run2[["Sale_Price"]], test_file_run[["Sale_Price"]], left_index=True, right_index=True)
#    rms = np.sqrt(((np.log(merge_df.Sale_Price_y).astype("float64")-np.log(merge_df.Sale_Price_x).astype("float64"))**2).mean())
#    result.loc[i,"model2"]= round(rms,4)
    
    
# part 2
test_file= pd.read_csv("sk/Part2/test_full.csv",index_col="PID")   
output_file=pd.read_csv("sk/Part2/mysubmission3.txt",index_col="PID")
merge_df=pd.merge(test_file[["Sale_Price"]], output_file[["Sale_Price"]], left_index=True, right_index=True)
rmse=np.sqrt(((np.log(merge_df.Sale_Price_y).astype("float64")-np.log(merge_df.Sale_Price_x).astype("float64"))**2).mean())
print(rmse)
