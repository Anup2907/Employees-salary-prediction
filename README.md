# Using 4D Data Science framework to build ML models for predicting salaries of employees with low MSE from their qualifications and job locations.

![4D](https://user-images.githubusercontent.com/56169217/74596378-0c0d7d00-5014-11ea-96e7-99a717063311.png)


'''=============================================================================   
  Machine Learning template for Regression & Classification datasets that can be used by 
  calling the required functions from the respective classes:
 
  Class Data :          Contains functions load_file, consolidate_data, data_info, 
                        categorical_cols, numerical_cols, missing_data, potential_outlier, 
                        clean_data, obj_to_cat, label_encode, one_hot_encode

  Class Regr_assump :   Contains functions target_dist, vif, ols_summ (joint_plot, QQ_plot) 

  Class Descrip_stats : Contains functions describe, hist, log_transform, scatter_plot,
                        count_box_plot

  Class Feat_engg :     Contains group_mean, group_median, group_std, group_max, group_min

  Class Model :         Contains train_model, test_results, print_summary, save_results 
  =============================================================================='''
  
  **DISCOVER THE DATA**
  
Feature file has all the independent features and Target file has dependent feature. After merging the both the files on unique identifier "jobId", we get the salary data. It has 1M rows and 9 columns.
  
  The features of Salary data are:
  
  1. JobId --> unique identifier of every employee
  
  2. CompanyId --> the company id where the employee works
  
  3. Jobtype --> where the position of the job is junior, senior, CEO
  
  4. Degree --> if the employee has degree, master's or phD
  
  5. Major --> whether the employee's major is Chemistry or biology
  
  6. Industry --> if the employee's domain is Oil or Healthcare
  
  7. yearsExperience --> number of years of professional experience
  
  8. milesFromMetropolis --> how far the company is located from Metropolis
  
  9. Salary --> Salary in thousands (which is the target feature)
  
  ![salary_info](https://user-images.githubusercontent.com/56169217/74596533-4415bf80-5016-11ea-89ca-ae7f8db72f25.PNG)
  
  There is no missing data in the dataframe. After examining potential outliers there is bad data below the lower band of salaries which   I have removed.
  
  ![outlier](https://user-images.githubusercontent.com/56169217/74597555-df169580-5026-11ea-8467-5779feadb659.PNG)
  
  Identifying the numerical and categorical columns:
  
  ![cat_nums](https://user-images.githubusercontent.com/56169217/74597275-05860200-5022-11ea-8acf-3cf206880177.PNG)
  
  Now let's look at dependent variable description and it's distribution
  
  ![salary](https://user-images.githubusercontent.com/56169217/74597323-43cff100-5023-11ea-871f-1013633d3403.PNG)
  
  ![salary_dist](https://user-images.githubusercontent.com/56169217/74597324-46324b00-5023-11ea-8c49-a29d3d0a7641.PNG)

  Also, let's look at how numerical features years of experience and miles from metropolis are related to salary,
  
  ![years](https://user-images.githubusercontent.com/56169217/74597560-e8076700-5026-11ea-99e4-3d35e25c7264.PNG)
  
  ![miles](https://user-images.githubusercontent.com/56169217/74597569-23099a80-5027-11ea-9863-03dbaf340baa.PNG)

  From the plots we can infer that,
  
  1. Years of experience is linearly related to Salary (As YoE increases Salary increases) 
  
  2. Miles from metropolis is non-linearly related to Salary (As MfM increases Salary decreases)

  Let's look at what majors are in demand and see their distributions,
  
  ![major](https://user-images.githubusercontent.com/56169217/74597593-96131100-5027-11ea-89ba-7aad2b631bb4.PNG)
  
  The count of employees who have their degree "None" is pretty high but we can't just interpret those employees doesn't have a degree.   This is a good example of data collection is done improperly or mistakes are made in data entry.
  
  **Feature Engineering**
  
  Feature engineering is important to get the best results out of models. I have created new features as "group_mean", "group_median",     "group_std", "group_min" and "group_max" that are the salary mean, median, stand deviation, maximum and minimum grouping all the         categorical features. Please refer below snap shot.
  
  ![feature_engg](https://user-images.githubusercontent.com/56169217/74597664-6a445b00-5028-11ea-8d79-cbb411c710d0.PNG)
  
  **Modeling**
  
  Creating a pipeline for dataprocessing and model building

  Firstly models with default parameters are run with 4-fold Cross Validation and then the best model is selected and the hyperparameters are tuned by hand

  The models selected are:

  1. Linear Regression

  2. Linear Regression with feature scaling and PCA (pipeline)

  3. Random Forest Regressor

  4. CatBoost Regressor
  
  The first three model results are:
  
  ![models](https://user-images.githubusercontent.com/56169217/74597760-28b4af80-502a-11ea-9b44-ca1ec5e90283.PNG)
  
  Though RandomForest gave us good results, let's check how CatBoost Regressor performs on the data. We need to mention categorical    features explicitly while fitting the training model for CatBoost otherwise it won't know what are all the categorical features

Note: The below code is already hand tuned by using different set of hyperparameters. The below set gave us the best results.

![catBoost](https://user-images.githubusercontent.com/56169217/74597764-2f432700-502a-11ea-87f1-27962029bb9a.PNG)

Please refer Python notebook: https://github.com/Anup2907/salarypredictionportfolio/blob/master/Salary_predictions.ipynb


  
  

  




