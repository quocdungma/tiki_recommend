
import time
import pandas as pd
import pickle
import nltk

nltk.download('punkt')
from nltk.tokenize import word_tokenize
import re

# Lấy danh sách stop words từ file
with open('vietnamese-stopwords.txt', 'r', encoding='utf-8') as file:
    stop_words = file.read().split('\n')

# Load pre-saved objects
with open('dictionary.pkl', 'rb') as f:
    dictionary = pickle.load(f)

with open('tfidf_model.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('similarity_index.pkl', 'rb') as f:
    index = pickle.load(f)

# Load product data
product_data = pd.read_csv('product_data.csv')

def recommender_gensim(product_id, num_recommendations):
    start_time = time.time()  # Start the timer

    # Retrieve product details based on product_id
    product_selection = product_data[product_data['item_id'] == product_id]

    # Check if the product is not found
    if product_selection.empty:
        print("Không tìm thấy sản phẩm với ID:", product_id)
        return None

    # Extract the product description
    name_description_pre = product_selection['content_wt'].to_string(index=False)

    # Convert search word into Sparse Vectors
    view_product = name_description_pre.lower().split()
    kw_vector = dictionary.doc2bow(view_product)

    # Similarity calculation
    sim = index[tfidf[kw_vector]]

    # Create dataframe to store results
    df_result = pd.DataFrame({'id': range(len(sim)), 'score': sim})

    # Get the highest scores
    highest_scores = df_result.sort_values(by='score', ascending=False).head(num_recommendations+1)

    # Extract the corresponding rows from product_data
    products_find = product_data[product_data.index.isin(highest_scores['id'])]
    results = products_find[['item_id', 'name', 'image', 'price', 'rating', 'description']]

    # Merge results and scores, and sort by score
    merged_results = pd.merge(results, highest_scores, left_index=True, right_on='id').sort_values(by='score', ascending=False)
    merged_results = merged_results[merged_results['item_id'] != product_id]


    # Extract recommended items (excluding the searched product)
    recs = merged_results[merged_results['item_id'] != product_id].values

    # Print the results
    print('Recommending ' + str(num_recommendations) + ' products similar to ' + product_selection['name'].values[0] + '...')
    print('*' * 96)
    for rec in recs:
        print('Recommended: product id: ' + str(int(rec[0])) + ', ' + rec[1] + ' (score: ' + str(rec[6]) + ')')

    end_time = time.time()  # End the timer

    print("\nTime taken for recommendation: {:.2f} seconds".format(end_time - start_time))

    return merged_results

# Hàm tiền xử lý đầu vào
def preprocess_input(input_text):
    input_text = re.sub('[^a-zA-Z0-9 \n\.]', '', input_text)
    input_text = input_text.lower()
    input_text = word_tokenize(input_text)
    input_text = ' '.join([word for word in input_text if word not in stop_words])
    return input_text

def recommend_for_new_product_gensim(product_name, num_recommendations=5):
    try:
        # Tiền xử lý đầu vào
        processed_input = preprocess_input(product_name)

        # Chuyển đổi đầu vào thành vector đặc trưng
        query_bow = dictionary.doc2bow(processed_input.split())

        # Tính toán độ tương tự sử dụng mô hình Gensim
        query_tfidf = tfidf[query_bow]
        sims = index[query_tfidf]

        # Lấy ra các sản phẩm tương tự
        similar_items = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[:num_recommendations]

        # Tạo một danh sách để lưu trữ thông tin sản phẩm
        product_info_list = []

        # Hiển thị kết quả
        print(f"Recommending {num_recommendations} products similar to {product_name}:")
        for item in similar_items:
            product_info = product_data.iloc[item[0]][['item_id', 'name', 'image', 'price', 'rating', 'description']]
            product_info_list.append(product_info)
            print(f"Product ID: {product_info['item_id']}, Product Name: {product_info['name']}, Score: {item[1]}")

        # Chuyển đổi danh sách thông tin sản phẩm thành DataFrame
        recommended_df = pd.DataFrame(product_info_list)
        return recommended_df
    except Exception as e:
        print(f"Có lỗi xảy ra: {e}")
        return None


import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-3.3.0-bin-hadoop3"
import findspark
findspark.init()

import pyspark
from pyspark import SparkContext
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexerModel, IndexToString, StringIndexer
from pyspark.sql import SQLContext
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import col, explode, udf, isnan, when, count, col, avg
import pyspark.sql.functions as F
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql.types import StringType, IntegerType, DoubleType

# Khởi tạo SparkSession
spark = SparkSession.builder \
    .appName("Product Recommendation System") \
    .getOrCreate()

# Lấy SparkContext từ SparkSession
sc = spark.sparkContext

model = ALSModel.load('ALS')
customer_indexer_model = StringIndexerModel.load('customer_indexer_model')
product_indexer_model = StringIndexerModel.load('product_indexer_model')

product_data_1 = pd.read_csv('ProductRaw.csv')
for col in product_data_1.columns:
    if col not in ['item_id', 'rating', 'price']:
        product_data_1[col] = product_data_1[col].astype(str)
product_data_spark = spark.createDataFrame(product_data_1)

product_data_spark = product_data_spark.withColumnRenamed("rating", "product_rating")
reviews_data = spark.read.csv("ReviewRaw.csv", header=True, inferSchema=True)

def get_purchased_products(customer_id, reviews_data, product_data_spark):
    # Bước 1: Lọc dữ liệu
    purchased_products = reviews_data.filter(F.col('customer_id') == customer_id).select('product_id', 'rating').distinct()

    # Bước 2: Join DataFrames
    purchased_products_details = purchased_products.join(product_data_spark, purchased_products.product_id == product_data_spark.item_id, how='inner')

    # Bước 3: Chọn các cột cần thiết
    purchased_products_details = purchased_products_details.select(
        'item_id', 'name', 'image', 'price', 'product_rating', 'description', 'rating'
    )


    # Chuyển đổi kết quả thành pandas DataFrame để hiển thị đẹp hơn
    purchased_products_details = purchased_products_details.toPandas()
    purchased_products_details['image'] = '<img src="' + purchased_products_details['image'].astype(str) + '" width="100" height="100"/>'

    return purchased_products_details

def recommend_products_for_user(customer_id, customer_indexer_model, product_indexer_model, model, product_data, num_recommendations=10):

    # Convert customer_id to its indexed form
    user_subset = spark.createDataFrame([(customer_id, )], ["customer_id"])
    user_subset_idx = customer_indexer_model.transform(user_subset)

    # Fetch recommendations
    userRecs = model.recommendForUserSubset(user_subset_idx, numItems=num_recommendations)  # Recommend top 10 items

    # Explode the recommendations
    userRecs_exploded = userRecs.withColumn("rec_exp", explode("recommendations")) \
                               .select("customer_id_idx", "rec_exp.product_id_idx", "rec_exp.rating")

    # Convert indexed product_id back to original form
    converter = IndexToString(inputCol="product_id_idx", outputCol="product_id_original", labels=product_indexer_model.labels)
    userRecs_converted = converter.transform(userRecs_exploded)

    # Join with product_data to get product names
    recommended_products = userRecs_converted.join(
        product_data_spark, userRecs_converted.product_id_original == product_data_spark.item_id).select('item_id','name','image','price','product_rating','description')
    recommended_products_pandas = recommended_products.toPandas()

    return recommended_products_pandas
