
import streamlit as st
from recommender import recommender_gensim, recommend_for_new_product_gensim, preprocess_input, recommend_products_for_user, customer_indexer_model, product_indexer_model, model, get_purchased_products
import pandas as pd

!apt update
!apt-get install openjdk-11-jdk-headless -qq > /dev/null
!wget -q http://archive.apache.org/dist/spark/spark-3.3.0/spark-3.3.0-bin-hadoop3.tgz
!tar -xvf spark-3.3.0-bin-hadoop3.tgz
!pip install -q findspark

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

# Load data
product_data = pd.read_csv('product_data.csv')

product_data_1 = pd.read_csv('ProductRaw.csv')
for col in product_data_1.columns:
    if col not in ['item_id', 'rating', 'price']:
        product_data_1[col] = product_data_1[col].astype(str)
product_data_spark = spark.createDataFrame(product_data_1)

product_data_spark = product_data_spark.withColumnRenamed("rating", "product_rating")
reviews_data = spark.read.csv("ReviewRaw.csv", header=True, inferSchema=True)

st.title('Product Recommendation System')

# Tạo thanh lựa chọn số lượng khuyến nghị
num_recommendations = st.sidebar.slider('Select number of products to display', 5, 50, 5)

# Tạo nút lựa chọn giữa 3 mô hình
recommend_option = st.sidebar.radio("Choose an option to recommend products:", ('Recommend by ID', 'Recommend by Name', 'Recommend for Customer ID'))

if recommend_option == 'Recommend by ID':
    # Allow user to input a product ID
    product_id = st.text_input('Nhập ID sản phẩm:', '')
    st.write('Ví dụ: 38458616, 14497425, 60228865,...')

    if product_id:
        try:
          product_id = int(product_id)
          # Get product details
          product_details = product_data[product_data['item_id'] == product_id]
          if not product_details.empty:
            
            st.markdown(f"**Product Name:** {product_details['name'].values[0]}")
            st.markdown(f"**Price:** {product_details['price'].values[0]}")
            st.markdown(f"**Rating:** {product_details['rating'].values[0]}")

            # Display product image if the URL is available
            image_url = product_details['image'].values[0]
            if image_url:
                st.image(image_url)

            # Get recommendations
            recommendations = recommender_gensim(product_id, num_recommendations)

            if recommendations is not None:
                st.write('Recommendations:')

                # Create a grid to display products in rows of 5
                for start in range(0, len(recommendations), 5):
                    end = start + 5
                    cols = st.columns(5)
                    for i, row in enumerate(recommendations.iloc[start:end].iterrows()):
                        index, data = row
                        # Display product image if the URL is available
                        if data['image']:
                            cols[i].image(data['image'])
                        
                        cols[i].markdown(f"**Price:** {data['price']}")
                        cols[i].markdown(f"**Rating:** {data['rating']}")
                        cols[i].markdown(f"**Name:** {data['name']}")

                        # Create a button to show description
                        if cols[i].button('Show Description', key=data['item_id']):
                            cols[i].write(f"Description: {data['description']}")
          else:
            st.write('Product not found.')
        except ValueError:
          st.write('Please enter a valid product ID.')
elif recommend_option == 'Recommend by Name':
    # Allow user to input a product name
    product_name = st.text_input('Nhập mô tả sản phẩm:', '')

    if product_name:
      # Tiền xử lý product_name trước khi gọi hàm recommend_for_new_product_gensim
      processed_product_name = preprocess_input(product_name)

      # Get recommendations
      recommendations = recommend_for_new_product_gensim(processed_product_name, num_recommendations)

      st.write('Recommendations:')
        # Create a grid to display products in rows of 5
      for start in range(0, len(recommendations), 5):
          end = start + 5
          cols = st.columns(5)
          for i, row in enumerate(recommendations.iloc[start:end].iterrows()):
                index, data = row
                # Display product image if the URL is available
                if data['image']:
                    cols[i].image(data['image'])
                
                cols[i].markdown(f"**Price:** {data['price']}")
                cols[i].markdown(f"**Rating:** {data['rating']}")
                cols[i].markdown(f"**Name:** {data['name']}")

                # Create a button to show description
                if cols[i].button('Show Description', key=data['item_id']):
                    cols[i].write(f"Description: {data['description']}")

elif recommend_option == 'Recommend for Customer ID':
    customer_id = st.text_input('Điền thông tin khách hàng:', '')
    st.write('ví dụ: 9909549, 10701688, 709310')
    if customer_id:
        try:
            customer_id = int(customer_id)

            # Gọi hàm để lấy thông tin về sản phẩm đã mua
            purchased_products_details = get_purchased_products(customer_id, reviews_data, product_data_spark)

            # Hiển thị thông tin về sản phẩm đã mua
            if not purchased_products_details.empty:
                st.write("Danh sách sản phẩm đã mua:")
                st.write(purchased_products_details.to_html(columns=['item_id', 'name', 'image', 'price', 'product_rating'], index=False, escape=False), unsafe_allow_html=True)
            else:
                st.write("Không tìm thấy thông tin về sản phẩm đã mua cho ID khách hàng này.")

            # Get recommendations
            recommendations = recommend_products_for_user(customer_id, customer_indexer_model, product_indexer_model, model, product_data_spark, num_recommendations)

            if recommendations is not None:
                st.write('Recommendations:')

                # Create a grid to display products in rows of 5
                for start in range(0, len(recommendations), 5):
                    end = start + 5
                    cols = st.columns(5)
                    for i, row in enumerate(recommendations.iloc[start:end].iterrows()):
                        index, data = row
                        # Display product image if the URL is available
                        if data['image']:
                            cols[i].image(data['image'])                        
                        
                        cols[i].markdown(f"**Price:** {data['price']}")
                        cols[i].markdown(f"**Rating:** {data['product_rating']}")
                        cols[i].markdown(f"**Name:** {data['name']}")

                        # Create a button to show description
                        with cols[i].expander("Description"):
                             st.write(f"{data['description']}")
        except ValueError:
            st.write('Please enter a valid customer ID.')
