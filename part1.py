import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder

# Read the CSV file
df = pd.read_csv("car_details_from_car_dekho.csv")

# Create a new column "year old" by subtracting the "year" column from the current year
df['year old'] = pd.Timestamp.now().year - df['year']

# Drop the "year" column as it is no longer required
df.drop('year', axis=1, inplace=True)

# Remove columns with only one unique value
for col in df.columns:
    if len(df[col].unique()) == 1:
        df.drop(col, axis=1, inplace=True)

# Set "selling_price" as the target column
target_col = 'selling_price'

# Split the data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

# Perform ordinal encoding on the categorical columns
cat_cols = train_df.select_dtypes(include=['object']).columns.tolist()
ordinal_encoder = OrdinalEncoder()
train_df[cat_cols] = ordinal_encoder.fit_transform(train_df[cat_cols])
test_df[cat_cols] = ordinal_encoder.transform(test_df[cat_cols])

# Fit a Random Forest Regressor to the training data
features = train_df.drop(target_col, axis=1)
target = train_df[target_col]
rf = RandomForestRegressor(random_state=42)
rf.fit(features, target)

# Calculate Mean Absolute Error for test data
test_features = test_df.drop(target_col, axis=1)
test_target = test_df[target_col]
preds = rf.predict(test_features)
mae = mean_absolute_error(test_target, preds)
print("Mean Absolute Error:", mae)
