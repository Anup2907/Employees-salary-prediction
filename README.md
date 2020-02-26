# 1. DEFINE
# Using 4D Data Science framework to build ML models for predicting salaries of employees with low MSE from their qualifications and job locations.

![4D](https://user-images.githubusercontent.com/56169217/74596378-0c0d7d00-5014-11ea-96e7-99a717063311.png)

 
# 2. DISCOVER THE DATA
  
There are three MS Excel files consisting of employees' data that is collected from famous job board:

1. Test Features (The dataset which was already split as 25% of the total data for prediction after trained by ML algorithms)

2. Train Features (The dataset which was already split as 75% of the total data for training into ML algorithms)

3. Train Salaries (The dataset which was already split as 75% of the total employee salaries for training into ML algorithms  


Feature file has all the independent features like Degree, Major etc and Target file has target feature, Salary. After merging the both the files on unique identifier "jobId", we get the whole salary data. It has 1M rows and 9 columns.
  
  The features of Salary data are:
  
  1. JobId --> unique identifier of every employee
  
  2. CompanyId --> the company id where the employee works
  
  3. Jobtype --> whether the position of the job is junior, senior, CEO
  
  4. Degree --> if the employee has degree, master's or phD
  
  5. Major --> whether the employee's major is Chemistry or biology
  
  6. Industry --> if the employee's domain is Oil or Healthcare
  
  7. yearsExperience --> number of years of professional experience
  
  8. milesFromMetropolis --> how far the company is located from Metropolis
  
  9. Salary --> Salary in thousands (which is the target feature)
  
 Identifying the numerical and categorical columns:
  
  ![cat_nums](https://user-images.githubusercontent.com/56169217/74597275-05860200-5022-11ea-8acf-3cf206880177.PNG)
  
 Identifying the missing values:

  ![salary_info](https://user-images.githubusercontent.com/56169217/74596533-4415bf80-5016-11ea-89ca-ae7f8db72f25.PNG)
  
 From the above screenshot, it is clear that there is no missing data in the dataframe but I could see the salary contain 0s in the       distribution from below snapshot which needs to be removed as employees donot work without salaries. But before removing the 0           salaries, let's look at potential outliers of lower band and upper band      
  
  ![potential outliers](https://user-images.githubusercontent.com/56169217/75195088-e886b880-571e-11ea-8b98-83866d53c870.PNG)

The upper band of outliers is above 220.5k and lower band is below 8.5k. Let's check who has above 220.5k salaries
  
  ![ouliers](https://user-images.githubusercontent.com/56169217/75193240-69dc4c00-571b-11ea-9271-61b3020da34e.PNG)

  After examining potential outliers it is evident that All the C level executives have salaries above 220.5k as expected but there are some Junior positions having salaries >220k. Let's check from which industries, Junior positions have above 220k salaries.
  
  ![oil](https://user-images.githubusercontent.com/56169217/75193458-ec650b80-571b-11ea-96c7-3b31254d2d4b.PNG)
  
  From the above snapshot, it is clear that Oil & Finance industries are known for higher salaries and hence the salaries looks legitimate.
  
  Now, let's look at the employees who have salaries below 8.5k, 
    
  ![outlier](https://user-images.githubusercontent.com/56169217/74597555-df169580-5026-11ea-8467-5779feadb659.PNG)
  
  From the above screenshot, we can say that this is a bad data and hence can be removed.
  
   Now let's look at dependent variable description and it's distribution
  
  ![salary](https://user-images.githubusercontent.com/56169217/74597323-43cff100-5023-11ea-871f-1013633d3403.PNG)
  
  ![salary_dist](https://user-images.githubusercontent.com/56169217/74597324-46324b00-5023-11ea-8c49-a29d3d0a7641.PNG)

  Also, let's look at how numerical features years of experience and miles from metropolis are related to salary,
  
  ![years](https://user-images.githubusercontent.com/56169217/74597560-e8076700-5026-11ea-99e4-3d35e25c7264.PNG)
  
  ![miles](https://user-images.githubusercontent.com/56169217/74597569-23099a80-5027-11ea-9863-03dbaf340baa.PNG)

  From the plots we can infer that,
  
  1. Employees' years of experience is linearly (positive) related to Salary (As YoE increases Salary increases) 
  
  2. Miles from metropolis is linearly (negative) related to Salary (As MfM increases Salary decreases)

  Let's look at what majors are in demand and see their distributions,
  
  ![major](https://user-images.githubusercontent.com/56169217/74597593-96131100-5027-11ea-89ba-7aad2b631bb4.PNG)
  
  The count of employees who have their degree "None" is pretty high but we can't just interpret those employees doesn't have a degree.   This is a good example of data collection is done improperly or mistakes are made in data entry.
  
  # 3. DEVELOP
  
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

**Feature Importance**

![featimp](https://user-images.githubusercontent.com/56169217/75199464-c85bf700-5728-11ea-82a1-0c05aba24e70.PNG)

From the above screenshot, we can say that **Employees' work experince** and **Employees' work location distance from metropolis** are the contributing factors in predicting the salary.

# 4. DEPLOY

Now the model is ready to be deployed to Production and other environments which could be scheduled to work either weekly or monthly depending on the volume of inflow data coming into the system.

  
  

  




