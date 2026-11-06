import pandas as pd
import numpy as np

# Load Fast Food Dataset
df = pd.read_csv("fastfood.csv")

print("Dataset Loaded Successfully!")
print("Total Rows:", df.shape[0])
print("Total Columns:", df.shape[1])


# Q1: Restaurant with the Highest Number of Food Items

restaurant_analysis = df["restaurant"].value_counts().reset_index()

restaurant_analysis.columns = [
    "Restaurant",
    "Total_Food_Items"
]

restaurant_analysis = restaurant_analysis.sort_values(
    by="Total_Food_Items",
    ascending=False
)

print("\nQ1: Number of Food Items in Each Restaurant")
print(restaurant_analysis.to_string(index=False))

# Find the restaurant with highest food items
highest_restaurant = restaurant_analysis.iloc[0]

print("\nRestaurant with Highest Food Items:")
print(highest_restaurant["Restaurant"])
print("Total Food Items:", highest_restaurant["Total_Food_Items"])

restaurant_analysis.to_csv(
    "Q1_Restaurant_Food_Count.csv",
    index=False
)

print("\nResult saved successfully!")




# Q2: Top 10 Food Items with the Highest Calories

calorie_analysis = df[
    ["restaurant", "item", "calories"]
].copy()

# Sort food items by calories
calorie_analysis = calorie_analysis.sort_values(
    by="calories",
    ascending=False
).head(10)

print("\nQ2: Top 10 Food Items with Highest Calories")
print(calorie_analysis.to_string(index=False))

# Save result
calorie_analysis.to_csv(
    "Q2_Top_10_Highest_Calories.csv",
    index=False
)

print("\nResult saved successfully!")



# Q3: Average Calorie Count for Each Restaurant

calorie_analysis = df.groupby(
    "restaurant"
).agg(
    Total_Food_Items=("item", "count"),
    Average_Calories=("calories", "mean"),
    Maximum_Calories=("calories", "max"),
    Minimum_Calories=("calories", "min")
).reset_index()

# Round average calories
calorie_analysis["Average_Calories"] = (
    calorie_analysis["Average_Calories"].round(2)
)

calorie_analysis = calorie_analysis.sort_values(
    by="Average_Calories",
    ascending=False
)

print("\nQ3: Average Calories for Each Restaurant")
print(calorie_analysis.to_string(index=False))

# Save result
calorie_analysis.to_csv(
    "Q3_Average_Calories_By_Restaurant.csv",
    index=False
)

print("\nResult saved successfully!")



# Q4: Restaurant with the Highest Average Protein Content

protein_analysis = df.groupby(
    "restaurant"
).agg(
    Average_Protein=("protein", "mean"),
    Maximum_Protein=("protein", "max"),
    Minimum_Protein=("protein", "min")
).reset_index()

# Round protein values
protein_analysis = protein_analysis.round(2)

protein_analysis = protein_analysis.sort_values(
    by="Average_Protein",
    ascending=False
)

print("\nQ4: Average Protein Content by Restaurant")
print(protein_analysis.to_string(index=False))

# Find the restaurant with highest average protein
highest_protein = protein_analysis.iloc[0]

print("\nRestaurant with Highest Average Protein:")
print(highest_protein["restaurant"])
print("Average Protein:", highest_protein["Average_Protein"], "g")

# Save result
protein_analysis.to_csv(
    "Q4_Highest_Average_Protein.csv",
    index=False
)

print("\nResult saved successfully!")




# Q5: Restaurant with the Highest and Lowest Average Total Fat

fat_analysis = df.groupby(
    "restaurant"
).agg(
    Average_Total_Fat=("total_fat", "mean")
).reset_index()

fat_analysis["Average_Total_Fat"] = (
    fat_analysis["Average_Total_Fat"].round(2)
)

fat_analysis = fat_analysis.sort_values(
    by="Average_Total_Fat",
    ascending=False
)

print("\nQ5: Average Total Fat by Restaurant")
print(fat_analysis.to_string(index=False))

# Find highest and lowest fat restaurants
highest_fat = fat_analysis.iloc[0]
lowest_fat = fat_analysis.iloc[-1]

print("\nRestaurant with Highest Average Total Fat:")
print(highest_fat["restaurant"])
print("Average Total Fat:", highest_fat["Average_Total_Fat"], "g")

print("\nRestaurant with Lowest Average Total Fat:")
print(lowest_fat["restaurant"])
print("Average Total Fat:", lowest_fat["Average_Total_Fat"], "g")

# Save result
fat_analysis.to_csv(
    "Q5_Average_Total_Fat_Analysis.csv",
    index=False
)

print("\nResult saved successfully!")


# Q6: Food Items with the Highest Sugar Content

sugar_analysis = df[
    ["restaurant", "item", "sugar", "calories"]
].copy()

# Sort by sugar content
sugar_analysis = sugar_analysis.sort_values(
    by="sugar",
    ascending=False
).head(10)

print("\nQ6: Top 10 Food Items with Highest Sugar")
print(sugar_analysis.to_string(index=False))

# Find highest sugar item
highest_sugar = sugar_analysis.iloc[0]

print("\nFood Item with Highest Sugar:")
print(highest_sugar["item"])
print("Restaurant:", highest_sugar["restaurant"])
print("Sugar:", highest_sugar["sugar"], "g")
print("Calories:", highest_sugar["calories"])

# Save result
sugar_analysis.to_csv(
    "Q6_Top_10_Highest_Sugar.csv",
    index=False
)

print("\nResult saved successfully!")



# Q7: Nutritional Comparison of Salad and Other Food Items

salad_analysis = df.groupby(
    "salad"
).agg(
    Total_Food_Items=("item", "count"),
    Average_Calories=("calories", "mean"),
    Average_Fat=("total_fat", "mean"),
    Average_Carbohydrates=("total_carb", "mean"),
    Average_Protein=("protein", "mean")
).reset_index()

# Round numerical values
numeric_columns = [
    "Average_Calories",
    "Average_Fat",
    "Average_Carbohydrates",
    "Average_Protein"
]

salad_analysis[numeric_columns] = (
    salad_analysis[numeric_columns].round(2)
)

print("\nQ7: Salad vs Other Food Items")
print(salad_analysis.to_string(index=False))

# Save result
salad_analysis.to_csv(
    "Q7_Salad_Other_Nutritional_Analysis.csv",
    index=False
)

print("\nResult saved successfully!")


# Q8: Best Combination of Low Calories and High Protein

nutrition_analysis = df[
    ["restaurant", "item", "calories", "protein"]
].copy()

# Min-Max Normalization
def normalize(series):
    if series.max() == series.min():
        return pd.Series(0, index=series.index)

    return (
        (series - series.min()) /
        (series.max() - series.min())
    )

# Lower calories get a higher score
nutrition_analysis["Calorie_Score"] = (
    1 - normalize(nutrition_analysis["calories"])
)

# Higher protein gets a higher score
nutrition_analysis["Protein_Score"] = normalize(
    nutrition_analysis["protein"]
)

# Calculate final score
nutrition_analysis["Nutrition_Score"] = (
    0.50 * nutrition_analysis["Calorie_Score"] +
    0.50 * nutrition_analysis["Protein_Score"]
) * 100

# Restaurant-wise analysis
restaurant_nutrition = nutrition_analysis.groupby(
    "restaurant"
).agg(
    Average_Calories=("calories", "mean"),
    Average_Protein=("protein", "mean"),
    Average_Nutrition_Score=("Nutrition_Score", "mean")
).reset_index()

restaurant_nutrition = restaurant_nutrition.round(2)

restaurant_nutrition = restaurant_nutrition.sort_values(
    by="Average_Nutrition_Score",
    ascending=False
)

print("\nQ8: Best Low Calories and High Protein Restaurant")
print(restaurant_nutrition.to_string(index=False))

# Find best restaurant
best_restaurant = restaurant_nutrition.iloc[0]

print("\nBest Restaurant:")
print(best_restaurant["restaurant"])
print("Average Calories:", best_restaurant["Average_Calories"])
print("Average Protein:", best_restaurant["Average_Protein"])
print("Nutrition Score:", best_restaurant["Average_Nutrition_Score"])

# Save result
restaurant_nutrition.to_csv(
    "Q8_Best_Low_Calorie_High_Protein.csv",
    index=False
)

print("\nResult saved successfully!")




# Q9: Top 10 Least Healthy Food Items

risk_analysis = df[
    [
        "restaurant",
        "item",
        "calories",
        "total_fat",
        "sat_fat",
        "trans_fat",
        "cholesterol",
        "sodium"
    ]
].copy()

# Risk factors
risk_columns = [
    "calories",
    "total_fat",
    "sat_fat",
    "trans_fat",
    "cholesterol",
    "sodium"
]

# Convert columns into numeric format
for col in risk_columns:
    risk_analysis[col] = pd.to_numeric(
        risk_analysis[col],
        errors="coerce"
    )

# Min-Max Normalization
def normalize(series):
    if series.max() == series.min():
        return pd.Series(0, index=series.index)

    return (
        (series - series.min()) /
        (series.max() - series.min())
    )

# Calculate individual risk scores
for col in risk_columns:
    risk_analysis[col + "_Score"] = normalize(
        risk_analysis[col]
    )

# Calculate Health Risk Score
score_columns = [
    col + "_Score" for col in risk_columns
]

risk_analysis["Health_Risk_Score"] = (
    risk_analysis[score_columns].mean(axis=1) * 100
)

# Sort by highest risk score
top_10_unhealthy = risk_analysis.sort_values(
    by="Health_Risk_Score",
    ascending=False
).head(10)

result = top_10_unhealthy[
    [
        "restaurant",
        "item",
        "calories",
        "total_fat",
        "sat_fat",
        "trans_fat",
        "cholesterol",
        "sodium",
        "Health_Risk_Score"
    ]
].copy()

result["Health_Risk_Score"] = (
    result["Health_Risk_Score"].round(2)
)

print("\nQ9: Top 10 Least Healthy Food Items")
print(result.to_string(index=False))

# Save result
result.to_csv(
    "Q9_Top_10_Least_Healthy_Food_Items.csv",
    index=False
)

print("\nResult saved successfully!")


# Q10: Healthiest Restaurant Overall Menu

import matplotlib.pyplot as plt
import seaborn as sns

health_analysis = df[
    [
        "restaurant",
        "item",
        "calories",
        "total_fat",
        "sugar",
        "sodium",
        "fiber",
        "protein",
        "vit_a",
        "vit_c",
        "calcium"
    ]
].copy()

# Nutritional factors
low_is_good = [
    "calories",
    "total_fat",
    "sugar",
    "sodium"
]

high_is_good = [
    "fiber",
    "protein",
    "vit_a",
    "vit_c",
    "calcium"
]

all_columns = low_is_good + high_is_good

# Convert columns into numeric format
for col in all_columns:
    health_analysis[col] = pd.to_numeric(
        health_analysis[col],
        errors="coerce"
    )

# Min-Max Normalization
def normalize(series):
    if series.max() == series.min():
        return pd.Series(0, index=series.index)

    return (
        (series - series.min()) /
        (series.max() - series.min())
    )

# Lower values get higher scores
low_scores = pd.DataFrame(index=health_analysis.index)

for col in low_is_good:
    low_scores[col + "_Score"] = (
        1 - normalize(health_analysis[col])
    )

# Higher values get higher scores
high_scores = pd.DataFrame(index=health_analysis.index)

for col in high_is_good:
    high_scores[col + "_Score"] = normalize(
        health_analysis[col]
    )

# Calculate average scores
health_analysis["Low_Nutrition_Score"] = (
    low_scores.mean(axis=1)
)

health_analysis["High_Nutrition_Score"] = (
    high_scores.mean(axis=1)
)

# Final Health Score
health_analysis["Health_Score"] = (
    0.50 * health_analysis["Low_Nutrition_Score"] +
    0.50 * health_analysis["High_Nutrition_Score"]
) * 100

# Compare all restaurants
restaurant_health = health_analysis.groupby(
    "restaurant"
).agg(
    Total_Food_Items=("item", "count"),
    Average_Calories=("calories", "mean"),
    Average_Fat=("total_fat", "mean"),
    Average_Sugar=("sugar", "mean"),
    Average_Sodium=("sodium", "mean"),
    Average_Fiber=("fiber", "mean"),
    Average_Protein=("protein", "mean"),
    Average_Vitamin_A=("vit_a", "mean"),
    Average_Vitamin_C=("vit_c", "mean"),
    Average_Calcium=("calcium", "mean"),
    Overall_Health_Score=("Health_Score", "mean")
).reset_index()

# Round numerical values
restaurant_health = restaurant_health.round(2)

# Rank restaurants by health score
restaurant_health = restaurant_health.sort_values(
    by="Overall_Health_Score",
    ascending=False
).reset_index(drop=True)

restaurant_health.insert(
    0,
    "Rank",
    range(1, len(restaurant_health) + 1)
)

print("\nQ10: Overall Nutritional Comparison of Restaurants")
print(restaurant_health.to_string(index=False))

# Find healthiest restaurant
healthiest = restaurant_health.iloc[0]

print("\nHealthiest Restaurant:")
print(healthiest["restaurant"])

print("Overall Health Score:",
      healthiest["Overall_Health_Score"])

# Save result
restaurant_health.to_csv(
    "Q10_Healthiest_Restaurant_Analysis.csv",
    index=False
)

# Create Health Score Chart
plt.figure(figsize=(12, 6))

sns.barplot(
    data=restaurant_health,
    x="restaurant",
    y="Overall_Health_Score"
)

plt.title("Overall Health Score of Fast Food Restaurants")
plt.xlabel("Restaurant")
plt.ylabel("Health Score")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "Q10_Healthiest_Restaurant_Chart.png"
)

plt.show()

print("\nAll Q10 results saved successfully!")


