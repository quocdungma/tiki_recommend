
import streamlit as st
from recommender import recommender_gensim, recommend_for_new_product_gensim, preprocess_input, get_purchased_products, recommend_products_user
import pandas as pd
import os
import json

# Load data
product_data = pd.read_csv('product_data.csv')
product_data = product_data.rename(columns={'rating': 'product_rating'})
reviews_data = pd.read_csv('ReviewRaw.csv')
product_list = product_data[['item_id', 'name']].drop_duplicates().values.tolist()


#Đọc dữ liệu

df = pd.read_csv('recommendations_new.csv')

st.title('Product Recommendation System')

if st.sidebar.button('Giới Thiệu Mô Hình Khuyến Nghị'):
    st.header('Giới Thiệu Mô Hình Khuyến Nghị')

    st.markdown("""
    Mô hình khuyến nghị sản phẩm của chúng tôi dựa trên hai phương pháp chính:

    1. **Content-Based Filtering**: Phương pháp này khuyến nghị sản phẩm dựa trên đặc điểm của sản phẩm (Sử dụng thuật toán Gensim). Trong ứng dụng này, chúng tôi sử dụng phương pháp này để khuyến nghị sản phẩm dựa trên ID hoặc tên sản phẩm.

    2. **Collaborative Filtering**: Phương pháp này khuyến nghị sản phẩm dựa trên sự tương tác giữa người dùng và sản phẩm (Sử dụng mô hình ALS). Trong ứng dụng này, chúng tôi sử dụng phương pháp này để khuyến nghị sản phẩm dựa trên ID khách hàng.

    Dưới đây là một hình ảnh minh họa về hai phương pháp này:
    """)

    st.image('https://www.researchgate.net/profile/Marwa-Mohamed-54/publication/331063850/figure/fig3/AS:729493727621125@1550936266704/Content-based-filtering-and-Collaborative-filtering-recommendation.ppm')

# Tạo thanh lựa chọn số lượng khuyến nghị
num_recommendations = st.sidebar.slider('Select number of products to display', 5, 50, 5)

# Tạo nút lựa chọn giữa 3 mô hình
recommend_option = st.sidebar.radio("Choose an option to recommend products:", ('Recommend by ID', 'Recommend by Name', 'Recommend for Customer ID'))

if recommend_option == 'Recommend by ID':
    # Allow user to input a product ID
    selected_product = st.selectbox(
      'Chọn hoặc gõ tên sản phẩm:',
      options=product_list,
      format_func=lambda x: x[1])

    if selected_product:
      product_id = selected_product[0]  # Lấy product_id từ phần tử đã chọn

    if product_id:
        try:
          product_id = int(product_id)
          # Get product details
          product_details = product_data[product_data['item_id'] == product_id]
          if not product_details.empty:

            st.markdown(f"**Product Name:** {product_details['name'].values[0]}")
            st.markdown(f"**Price:** {product_details['price'].values[0]}")
            st.markdown(f"**Rating:** {product_details['product_rating'].values[0]}")

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
    customer_id = st.text_input('Enter customer ID:', '')
    st.write('Ví dụ: 9999553, 1064154, 1827148,...')

    if customer_id:
        try:
            customer_id = int(customer_id)

            # Display purchased products
            st.write('Purchased Products:')
            purchased_products = get_purchased_products(customer_id, reviews_data, product_data)
            if purchased_products is not None and not purchased_products.empty:
                st.markdown(purchased_products.to_html(escape=False), unsafe_allow_html=True)
            else:
                st.write('No purchased products found.')

            # Get recommendations using recommend_products function
            recommendations = recommend_products_user(customer_id, df, product_data, num_recommendations)

            
            if recommendations is not None:
                st.write('Recommendations:')
                cols = st.columns(5)

                for i, row in enumerate(recommendations.iterrows()):
                    index, data = row

                    # Display product image if available
                    if not pd.isnull(data['image']):
                        cols[i % 5].image(data['image'])

                    cols[i % 5].markdown(f"**Price:** {data['price']}")
                    cols[i % 5].markdown(f"**Rating:** {data['product_rating']}")
                    cols[i % 5].markdown(f"**Name:** {data['name']}")

                    with cols[i % 5].expander("Description"):
                        st.write(data['description'])

        except ValueError:
            st.write('Please enter a valid customer ID.')
