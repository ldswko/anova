# Libraries
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

# Dataset
netflix_data = pd.read_csv('netflix_titles.csv')

# Cleaning/Processing Data
netflix_data_cleaned = netflix_data.dropna(subset=['type', 'country', 'listed_in'])
netflix_data_exploded = netflix_data_cleaned.assign(listed_in=netflix_data_cleaned['listed_in'].str.split(', ')).explode('listed_in')

# Dataframes
genre_country_counts = netflix_data_exploded.groupby(['listed_in', 'country']).size().reset_index(name='count')

# Filtering to Top Genres
top_genres = genre_country_counts['listed_in'].value_counts().head(20).index
top_countries = genre_country_counts['country'].value_counts().head(20).index

# Filtering Dataframes
filtered_data = genre_country_counts[
    genre_country_counts['listed_in'].isin(top_genres) &
    genre_country_counts['country'].isin(top_countries)
]

# ANOVA Model
formula = 'count ~ listed_in + country'

# Fitting using smf.ols
anova_model = smf.ols(formula=formula, data=filtered_data).fit()

# ANOVA
anova_results = anova_lm(anova_model)

# ANOVA Results
print("ANOVA Results:")
print(anova_results)
