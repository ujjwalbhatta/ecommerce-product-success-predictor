# Amazon Rating Prediction Project
A product manager wants to evaluate a NEW product in an EXISTING category/brand. Using historical performance data of similar products (same brand tier, category), predict how this new product will perform.

**Example Use Case:**
- **Input:** price=$25, category="Electronics/Accessories", brand="GenericBrand", description="Durable silicone case with kickstand"
- **Output:** "⚠️ 2.8/5 star rating expected - HIGH RISK product"

## 📊 Dataset
Using **Amazon Reviews 2023** dataset from McAuley Lab:
- **Source:** https://amazon-reviews-2023.github.io/
- **Category:** Electronics (43.9M reviews, 18.3M users, 1.6M items)
- **Sample Size:** 30,000 merged records for this project

## 📁 Data Files Required
Download these files from the website:
1. `Electronics.jsonl` - Review data (ratings, text, user info)
2. `meta_Electronics.jsonl` - Product metadata (title, price, features, etc.)

Run file data_extraction.ipynb file to save the new data in csv format with everything you need.


## 📋 Output Dataset Features

### From Product Metadata:
- `main_category` - Product category
- `product_title` - Product name
- `average_rating` - Overall product rating
- `rating_number` - Number of ratings
- `price` - Product price in USD
- `description` - Product description (list format)
- `parent_asin` - Unique product ID
- `details` - Product details (contains brand, size, etc.)

### From Reviews:
- `rating` - Individual review rating (**Target variable**)
- `review_title` - Review title
- `text` - Review content
- `helpful_vote` - Review helpfulness votes

🛠️ Project Pipeline
1. data_cleaning.ipynb
Handle missing values and outliers (e.g., drop products with missing price, filter unrealistic values).
Normalize text fields (lowercasing, removing special characters, etc.).
Save cleaned and merged dataset into a CSV file for downstream use.
Output: [cleaned_data.csv](data/processed/amazon_electronics_clean.csv)

2. feature_engineering.ipynb
This notebook transforms the cleaned dataset into machine learning–ready features.

