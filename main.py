import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import scipy.stats as stats

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Student Performance Metrics Analysis
# https://data.mendeley.com/datasets/5b82ytz489/1

df = pd.read_csv("ResearchInformation3.csv")

#-----------------------------------------------------------------------------------------------
### Data Cleaning

# All columns are already in the correct datatypes, with strings being objects,
# and numbers being in float64 (all numbers use decimals so that is appropriate)
# except for the ID, which is an int64
print("\n Datatypes:")
display(df.dtypes)

#Quick check to see if dataset is clean.
print("\n Isnull:")
display(df.isnull().sum())

df['Income'] = df['Income'].str.strip()

## Overall Summary Statistics   

### Summary Statistics for Entire Dataset
display(df.describe())

incomes = df.groupby('Income')
home_towns = df.groupby('Hometown')
overall_gpas = df.groupby('Overall')

### Summary Statistics by Income (Low, Lower Middle, Upper Middle, High)

for income,group in incomes:
    sorted = group.sort_values("Income", ascending=True)
    print(f"\nIncome Level: {income}")
    display(sorted.describe())

### Summary Statistics by Hometown (Village, City)
for home,group in home_towns:
    sorted = group.sort_values("Hometown", ascending=True)
    print(f"\nType:: {home}")
    display(sorted.describe())

### Summary Statistics by GPA
bins = [0, 1.99, 2.99, 3.99, 4.01]
labels = ["1.00–1.99", "2.00–2.99", "3.00–3.99", "4.00"]

df['GPA_Range'] = pd.cut(df['Overall'], bins=bins, labels=labels, include_lowest=True)

overall_gpas = df.groupby('GPA_Range')

for gpa, group in overall_gpas:
    print(f"\nGPA Range: {gpa}")
    display(group.describe())


#--------------------------------------------------------------------------------------------------
## Hypothesis Testing

### 1. Does time spent on gaming, jobs, and extracurriculurs affect time spent on preparation?

# $H_0$: There is no significant relationship between the amount of time spent on gaming, jobs, and extracurriculurs and the amount of time spent on preparation.
# $H_A$: There is a significant relationship between the amount of time spent on gaming, jobs, and extracurriculurs and the amount of time spent on preparation.

'''
While there are only so many hours in a week, it may be the case that participation in various activihies is correlated in different ways with the amount of time that students spend preparing for class.
This analysis seeks to determine how different types of activities (gaming, having a job, and participating in extracurriculurs) each affect preparation time.
'''

# Creates a Profile column with a combined set of Gaming, Job, and Extracurriculurs
df['Profile'] = ('Gaming: ' +
    df['Gaming'].astype(str) + ' | ' +
    'Job: ' + df['Job'].astype(str) + ' | ' +
    'Extra: ' + df['Extra'].astype(str)
)

prep_dist = df.groupby('Profile')['Preparation'].value_counts(normalize=True).mul(100).rename('Percentage').reset_index()

contingency = pd.crosstab(df['Profile'], df['Preparation'])

# Chi-square test
chi2, p, dof, expected = stats.chi2_contingency(contingency)

print(f"Chi-square: {chi2}")
print(f"Degrees of freedom: {dof}")
print(f"P-value: {p}")

# Heatmap
heatmap_data = prep_dist.pivot(index='Profile', columns='Preparation', values='Percentage')
heatmap_data = heatmap_data.fillna(0)  # Replace NaN with 0

plt.figure(figsize=(14, 8))
sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', fmt=".1f")
plt.title("Combined Effects of Gaming, Job, and Extracurriculars on Preparation (%)")
plt.xlabel("Preparation Level")
plt.ylabel("Gaming + Job + Extracurricular Profile")
plt.xticks(rotation=0)
plt.tick_params(axis='x', length=0)
plt.yticks(rotation=0)
plt.tick_params(axis='y', length=0)
plt.tight_layout()
plt.show()

'''
#### Conclusion
The p-value of the Chi-squared test is less than .01, the value of α, which suggests that there is a significant relationship between the amount of time spent on gaming, jobs, and extracurriculurs and the amount of time spent on preparation.
Based off the heatmap, those who have an extracurricular tend to prepare for 2-3 hours most of the time. However, those who game for over 3 hours tend to prepare the least at around 0-1 
hours, even if they have an extracurricular.

Otherwise, there does not appear to be a significant relation between time spent on preparation and time spent on gaming or having a job, as across all groups who devote time to gaming, 
a job, or an extracurricular, but game less than 2-3 hours, the time spent on preparation tends to be 2-3 hours.
'''

### 2. Does the level of income have an impact on the computer proficiency level of a student? (α = 0.10)

# $H_0$: There is no difference between the computer proficiencies of any of the students based on their levels of income.
# $H_A$: There is a difference between at least one of the level of incomes due to computer proficiency.
'''
#### Hypothesis 2 Introduction

While many attribute performance in school to talent or effort, many students are disadvantaged when coming into higher education as a result of their economic background.
There are some skills which are not taught in higher education, but still necessary for success.
In the digital age, chief among these skills is computer proficiency.
We were concerned with how income may impact the computer proficiency that students bring into their college education.
The dataset we selected stratified income into four categories: Low (Below 15,000 RM), Lower middle (15,000 - 30,000 RM), Upper middle (30,000 - 50,000 RM), and High (Above 50,000 RM).

We have displayed a box plot to provide a visual representation of how these income groups differ in terms of computer proficiency, but will explore the difference more concretely with a hypothesis test.

This hypothesis explores if there are significant differences in computer proficiency between the provided levels of income.
We will use an ANOVA test to compare the mean computer proficiency of the four income groups to determine whether there is a significant difference between them.
Then, we will perform a post-hoc analysis on the results of the ANOVA test, conducting Tukey's Honest Significant Difference test to identify which income groups differ from one another.
'''

groups = 'Income'
value = 'Computer'

summary = df.groupby(groups)[value].mean()
print(summary)


low = df[df[groups] == 'Low (Below 15,000)'][value]
low_mid  = df[df[groups] == 'Lower middle (15,000-30,000)'][value]
upper_mid = df[df[groups] == 'Upper middle (30,000-50,000)'][value]
high= df[df[groups] == 'High (Above 50,000)'][value]

# Box Plot & ANOVA Test - Post Hoc (Tukey's HSD)

# Boxpolot
sns.boxplot(x=groups, y=value, data=df, palette='deep')
plt.title("Computer Proficiency Distribution by Level of Income")
plt.xlabel("Level of Income")
plt.ylabel("Computer Proficiency (1-5)")
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.show()

# ANOVA
p_value = stats.f_oneway(low, low_mid, upper_mid, high).pvalue
print(f"P-value for one-way ANOVA test: {p_value}")

# Post Hoc (Tukey's HSD)
tukey = stats.tukey_hsd(low, low_mid, upper_mid, high)
print(f"Post-Hoc (Tukey's HSD): {tukey}")


#### Conclusion
'''
The p-value of our ANOVA test is less than 0.1, the value of α. 
Since the p-value ≤α, it is unlikely that we would observe the results of this test if there was no difference between the computer proficiencies of students from different income level. 
Thus, we reject the null hypothesis.

In regards to Tukey's Post-Hoc test, the p-values of all pairings are ≤α, 
so we can conclude that there are differences between each income level in regards to computer proficiency.
'''


### 3. Does the Highschool GPA correlate to College Overall GPA? (α = 0.05)


#### Hypothesis 3 Introduction
'''
Common sense would typically be that students who are more academically proficient in high school would be the same in college,
but do we have proof? Therefore, this hypothesis explores whether High School GPA is associated with the College Overall GPA.
Since these values are restricted to fixed ranges and not normally distributed, Spearman's Rank Coefficient is used to
evaluate the strength and direction of this relationship. This is then displayed on a Scatter Plot because it effectively
shows the relationshp between two variables that combine to academically make up one student.
'''

# $H_0$: There is no a significant relationship between High School GPA and College GPA.
# $H_A$: There is a significant relationship between High School GPA and College GPA.

x_val = 'HSC'
y_val = 'Overall'
df_hyp3 = df[[x_val, y_val]]

rho, p = stats.spearmanr(df_hyp3[x_val], df_hyp3[y_val])
print("Spearman's Rank Coefficient (rho) = ", rho)
print("p-value", p)

#Scatter Plot & Regression Analysis & Spearman's Rank Correlation
sns.regplot(x=x_val, y=y_val, data=df_hyp3, line_kws={'color':'red'})
plt.title('Highschool GPA Versus College Overall GPA')
plt.xlabel('Highschool GPA (5.0 Scale)')
plt.ylabel('College Cumulat vGPA (4.0 Scale)')
plt.show()

#### Hypothesis 3 Conclusion
'''
The completion of the the Spearman's Rank Correlation test between High School GPA and College GPA 
shows us that there is a somewhat weak but statistically significant relationship between the two features.
The p-value of approximately 9.0614e-11 is significantly below the alpha value of 0.05, which proves
that the two features are definatley related to each other. The Spearman's Rank Correltation Coefficient
of approximately 0.287 is positive, which indicates that higher High School GPAs lead to higher College GPAs.
Relatively speaking, 0,287 is weak, but the range of GPAs is overall not that large, mostly lying between
2 and 4, so this makes sense. Overall, we can reject the null hypothesis and conclude that students with higher
High School GPAs are more likely to have higher College GPAs and vice versa.
'''

#### Machine Learning
'''

Goal: Using highschool information, predicting the cumulative college GPA.

Up to this point, we have seen how there are many different ways that the features in this dataset are correlated.
Therefore, it is feasible that we could use these features to predict useful information about students. Specificaliy,
we could predict students overall college GPA based off of all of the features they would have from high school. This would
allow students to have an idea of what kind of GPA they would have in college, or college admissions people to have an expectation
of what the students they will admit will have.

''''

#### Machine Learning Data Preparation
''''
Before we train the models, we need to prepare the features to be accessible by the models we will test for 
training. This will be done by encoding them all in to numbers. Overall, we have three different kinds of features.

First, we have numerical data, where the data is already float64s so no preperation needs to be done. 

Next is the ordinal data, where they are categorical objects that have a hierarchy. For example, 3+ Hours of 
preparation is greater than 1 hour. To prepare these, we use Label Encoding with maps to give tiered values
to each of the Categories.

Finally, there is categorical data, where they are categorical objects that don't have any type of hierarchy.
For example, we don't know if students in the Department of Busienss Administration are necessarily better
than those in Computer Science and Engineering. Therefore for those, we use One-Hot enconding to give them all
unique binary values that don't represent a specific hierarchy.
'''

num_cols = df[['HSC', 'SSC']].copy()
cat_cols = df[['Department', 'Gender', 'Hometown', 'Job', 'Extra']].copy()
ord_cols = df[['Income', 'Computer', 'Preparation', 'Gaming', 'Attendance', 'English']].copy()

# One-hot encoding
encoder = OneHotEncoder()
cat_cols = encoder.fit_transform(cat_cols)
cat_cols = pd.DataFrame(cat_cols.toarray(),columns=encoder.get_feature_names_out(["Department", "Gender", "Hometown", "Job", "Extra"]))

scaler = StandardScaler()
num_cols = scaler.fit_transform(num_cols)
num_cols = pd.DataFrame(num_cols, columns=['HSC','SSC'])

income_map = {
    'Low (Below 15,000)':1, 
    'Lower middle (15,000-30,000)':2,
    'Upper middle (30,000-50,000)':3, 
    'High (Above 50,000)':4
    }

preparation_map = {
    '0-1 Hour':1, 
    '2-3 Hours':2,
    'More than 3 Hours':3, 
    }

gaming_map = {
    '0-1 Hour':1, 
    '2-3 Hours':2,
    'More than 3 Hours':3, 
    }

attendance_map = {
    'Below 40%':1, 
    '40%-59%':2,
    '60%-79%': 3,
    '80%-100%':4, 
    }


ord_cols["Income"] = ord_cols["Income"].map(income_map)
ord_cols["Preparation"] = ord_cols["Preparation"].map(preparation_map)
ord_cols["Gaming"] = ord_cols["Gaming"].map(gaming_map)
ord_cols["Attendance"] = ord_cols["Attendance"].map(attendance_map)

X = pd.concat([cat_cols, num_cols, ord_cols], axis=1)

# Overall is the numerical amount that we are trying to predict
Y = df['Overall']

# Splitting of data into testing and training
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

def draw_scatter(Y_train, Y_train_pred, Y_test, Y_test_pred):
        plt.scatter(Y_train, Y_train_pred, color='blue', label='Train')
        plt.scatter(Y_test, Y_test_pred, color='orange', label='Test')
        plt.title("Predicted GPA vs Actual GPA")
        plt.xlabel("Actual GPA")
        plt.ylabel("Predicted GPA")
        plt.legend()
        plt.show()


# Testing out the each of the models
'''
We initially tried the linear regression model for the data, however it gave subpar performance with 
a MSE of _____ and an R2 value of _________. This was unsatisfatory, and was most likely because of the 
categorical data. Thus we decided to try the Random Forest Tree model instead.
With the Random Forest Model, the MSE and the R2 values were much more satisfactory, with the MSE = _____
and the R2 = _______. 
Finally to find the optimal number of trees, we used the method of generating hundreds of trees, 
and finding the one with the best result. We found the ideal number of trees to predict was n_estimators = 125.
'''
# Linear Regression Model
model = LinearRegression().fit(X_train, Y_train)

# Predictions from the Model
Y_train_pred = model.predict(X_train)
Y_test_pred = model.predict(X_test)

# Evaluations of model
mse_train = mean_squared_error(Y_train, Y_train_pred)
mse_test = mean_squared_error(Y_test, Y_test_pred)
r2_train = r2_score(Y_train, Y_train_pred)
r2_test = r2_score(Y_test, Y_test_pred)

print(f"Current Model: LinearRegression")
print(f"Train - MSE: {mse_train:.4f}, R^2: {r2_train:.4f}")
print(f"Test - MSE: {mse_test:.4f}, R^2: {r2_test:.4f}\n")

draw_scatter(Y_train, Y_train_pred, Y_test, Y_test_pred)

# Random Forest Tree Model
model = RandomForestRegressor(n_estimators=125, max_depth=None, random_state=42).fit(X_train, Y_train)

# Predictions from the Model
Y_train_pred = model.predict(X_train)
Y_test_pred = model.predict(X_test)

# Evaluations of model
mse_train = mean_squared_error(Y_train, Y_train_pred)
mse_test = mean_squared_error(Y_test, Y_test_pred)
r2_train = r2_score(Y_train, Y_train_pred)
r2_test = r2_score(Y_test, Y_test_pred)

print(f"Model: RandomForestRegressor; Number of trees: {125}")
print(f"Train - MSE: {mse_train:.4f}, R^2: {r2_train:.4f}")
print(f"Test - MSE: {mse_test:.4f}, R^2: {r2_test:.4f}\n")

draw_scatter(Y_train, Y_train_pred, Y_test, Y_test_pred)

'''
Here we see that the Random Forest model shows a strong performance with a Mean Squared  Error (MSE) of 0.0201
and an R^2 of 0.9416, showing that the model fits the training data well. At the same time, there's a performance drop
on the test set where our MSE is 0.1445 and our R^2 is 0.6128. This means that the model is most likely overfitting
and is memorizing the patterns rather than learning them effectively.
'''


'''
Conclusive statements:
Ultimately, the analyses show that academic background and socioeconomic factors are meaningfully connected towards college performance.
Based on our hypothesis and ML testing, we can say that a moderately accurate GPA prediction model can be built from high school and demographic features.
When combined together, the hypothesis stests as well as the models show that despite some relationships being weak in magnitude, they still are 
statistically significang and interpretable.

We noticed that the time being spent on gaming, jobs, and extracurriculuars significantly correlate with preparation time, with a Chi-squared value of p<0.01.
Additionally, heavy gaming (more than 3 hours) seems to negatively impact preparation time, while extracurriculars align with moderate preparation (2-3 hours).

When discussing income levels, it correlates to computer proficiency, with higher income students (above >50000 RM) showing significantly better computer skills (p<0.10).
High School GPA also positively correlates to college GPA, although weakly. This indicates prior achievement as a reliable predictor, despite mediating influences such as environment.

Our Maching Learning reinforces these mpatterns, especially through our Random Forest Regressor model which involves 125 trees. It outperforms Linear Regression, achieving a test R^2 of 0.6128
through testing data, which possible signals overfitting as the training data showed much better results of R^2=0.9416; however, it still indicates that high school and demographic features can moderately predict college GPA.

As shown above, foundational advantages of income-linked proficency and high school performance, combined with daily choices like gaming and preparation time imply that college success is multifaceted.
In conclusion, college success emerges as a complex interplay of socioeconomic background, prior academic achievement, and personal habits, and having holistic interventions to address both structural inequalities and individual habits 
can produce better educational outcomes for all students.


'''