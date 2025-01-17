# Libraries
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.outliers_influence import OLSInfluence
import seaborn as sns
import matplotlib.pyplot as plt

# ANOVA Model
anova_model = smf.ols(formula='count ~ listed_in + country', data=filtered_data).fit()

# AIC
print(f"AIC: {anova_model.aic}")

# Durbin-Watson (Independence)
durbin_watson_value = sms.durbin_watson(anova_model.resid)
print(f"Durbin-Watson: {durbin_watson_value}")

# Plot of Fitted Values vs Residuals
plt.figure(figsize=(10, 6))
sns.scatterplot(x=anova_model.fittedvalues, y=anova_model.resid)
plt.axhline(0, color='red', linestyle='--', linewidth=1)
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Fitted Values vs Residuals')
plt.show()

# Breusch-Pagan (Equal Variance)
homo_breusche_pagan = sms.het_breuschpagan(anova_model.resid, anova_model.model.exog)
print(f"Breusch-Pagan Test: {homo_breusche_pagan}")

# Plot of Residuals
plt.figure(figsize=(10, 6))
anova_model.resid.hist(bins=20)
plt.title('Histogram of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

# Normality
normal_ad_test = sms.normal_ad(anova_model.resid)
print(f"Normality Test: {normal_ad_test}")

# Influential Observations
influential_observations = OLSInfluence(anova_model).summary_frame().describe()
print("Influential Observations Summary:")
print(influential_observations)

# Summary
print(anova_model.summary())
