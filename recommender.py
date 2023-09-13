
import time
import pandas as pd
import pickle
import nltk

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
import json
import pandas as pd

def get_purchased_products(customer_id, reviews_data, product_data):
    # Step 1: Filter data
    purchased_products = reviews_data[reviews_data['customer_id'] == customer_id][['product_id', 'rating']].drop_duplicates()

    # Step 2: Join DataFrames
    purchased_products_details = purchased_products.merge(product_data, left_on='product_id', right_on='item_id', how='inner')

    # Step 3: Select necessary columns
    purchased_products_details = purchased_products_details[['item_id', 'name', 'image', 'price', 'product_rating', 'description', 'rating']]
    
    # Step 4: Embed images in DataFrame
    purchased_products_details['image'] = purchased_products_details['image'].apply(lambda x: f'<img src="{x}" width="100px" />')

    return purchased_products_details

def recommend_products_user(customer_id, df, products_df, num_recommendations=10):

    customer_id = int(customer_id) 

    # Bước 1: Xác định các sản phẩm đã được khuyến nghị
    recommended_products_already = df[df['customer_id'] == customer_id][['product_id', 'rating']].head(num_recommendations)

    # Bước 2: Tìm sản phẩm tương tựht
    # Ở đây, chúng ta chỉ đơn giản lấy những sản phẩm chưa được khuyến nghị từ tập sản phẩm
    similar_products = products_df[products_df['item_id'].isin(recommended_products_already['product_id'])]

    # Bước 3: Sắp xếp sản phẩm dựa trên đánh giá từ df
    # Ở đây, chúng ta sáp nhập cột đánh giá từ df vào similar_products và sau đó sắp xếp chúng
    similar_products = similar_products.merge(recommended_products_already, left_on='item_id', right_on='product_id', how='left')
    recommended_products = similar_products.sort_values(by='rating', ascending=False)
    
    # Bước 4: Trả về danh sách khuyến nghị
    return recommended_products[['name','description','product_rating','price','image']].head(num_recommendations)
