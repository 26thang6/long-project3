import pandas as pd
import streamlit as st
import pickle
import random
from pre_process import load_files, process_text, loaddicchar, process_special_word, normalize_repeated_characters, process_postag_thesea, covert_unicode, analyze_general

bg = """
<style>
[class="main st-emotion-cache-bm2z3a ea3mdgi8"] {
background-color: #ffe0cc;
}

[class="st-emotion-cache-12fmjuu ezrtsby2"]  {
background-color: #ffe0cc;
}

[class="st-emotion-cache-6qob1r eczjsme8"] {
background-color: #ff4d2a;
}
</style>
"""

st.markdown(bg, unsafe_allow_html=True)

restaurant_ids = [971, 178, 183, 184, 192, 193, 195, 196, 198, 987, 973, 953, 778, 948, 941, 917, 868, 858, 339, 840, 829, 794, 786, 1056, 169, 161, 159, 111, 1148]
random.seed(42)
suggested_ids = random.sample(restaurant_ids, 5)
suggested_ids = sorted(suggested_ids)

df_rev_resam = pd.read_csv('df_rev_resam.csv')
df = pd.read_csv('merged_df.csv').fillna('')
data = load_files()

pickle_file = "log_model_word_balance.pkl"
with open(pickle_file, 'rb') as file:
    loaded_log_model_word_balance = pickle.load(file)

pickle_file = "vectorizer.pkl"
with open(pickle_file, 'rb') as file:
    loaded_vectorizer = pickle.load(file)

st.image('shopeefood.png', use_column_width=True)
menu = ['Home Page', 'Review Classification', 'Restaurant Information', 'About Us']
choice = st.sidebar.selectbox('Menu', menu)

if choice == 'Home Page':
    st.markdown('''
    **ShopeeFood** is an app that provides food delivery services primarily in major cities in Vietnam. It was formerly known as the Now app before being rebranded. ShopeeFood allows users to order food from a variety of restaurants and have it delivered directly to their doorstep. The app offers a user-friendly interface, extensive merchant selections, and is designed to handle group orders efficiently.

    **Key features of ShopeeFood include:**

    - **Extensive Merchant Options**: Users can choose from a wide range of food options from various restaurants.
    - **Ease of Use**: The app is designed to be intuitive and easy to use.
    - **Group Ordering**: Facilitates ordering food for groups.
    - **Incentives and Utilities**: Provides various incentives and optimal utilities for users.
    ''')

    st.image('shopee.png', use_column_width=True)

    st.markdown('''    
    The app aims to enhance the food delivery experience by optimizing product search, improving battery consumption, and enhancing product uploads and edits.

    For more information, you can visit the [website](https://shopeefood.vn/).

    Additionally, ShopeeFood is part of the larger trend of online food delivery services similar to platforms like ShopRite From Home, which offers grocery delivery and curbside pickup services in certain regions, allowing customers to order groceries online and have them delivered or ready for pickup at their convenience.
    ''')


    st.write("<br><br><br>", unsafe_allow_html=True)
    st.markdown("### Feedback Form ###")
    name = st.text_input("Your Name")
    email = st.text_input("Your Email")
    feedback = st.text_area("Your Feedback")

    if st.button("Submit"):
        st.success("Feedback sent successfully!")

elif choice == 'Review Classification':
    st.subheader("Sentiment Analysis")
    type = st.radio("", options=["Input one review", "Input multiple reviews", "Upload review file"])

    if type == "Input one review":
        st.markdown("### Input review to text area ###")
        customer_review = st.text_input('Enter content of courses')

        if st.button('Predict'):
            customer_review = process_text(customer_review, data['emojicon'], data['teencode'], data['wrong_words'])
            customer_review = covert_unicode(customer_review)
            customer_review = process_special_word(customer_review)
            customer_review = normalize_repeated_characters(customer_review)
            customer_review = process_postag_thesea(customer_review)
            new_comment = loaded_vectorizer.transform([customer_review])
            pred = loaded_log_model_word_balance.predict(new_comment)
            st.markdown(f'**Prediction:** {pred[0]}')

    elif type == "Input multiple reviews":
        st.markdown("### Input reviews to text area ###")
        comment_df = pd.DataFrame(columns=["Comment"])

        reviews = []
        for i in range(5):
            review = st.text_area(f"Review {i+1}:")
            reviews.append({"Comment": review})
        comment_df = pd.concat([comment_df, pd.DataFrame(reviews)], ignore_index=True)
        comment_df = comment_df[comment_df["Comment"].str.len() >= 2]
        if st.button('Predict'):
            comment_list = comment_df["Comment"].tolist()
            comment_list = [process_text(comment, data['emojicon'], data['teencode'], data['wrong_words']) for comment in comment_list]
            comment_list = [covert_unicode(comment) for comment in comment_list]
            comment_list = [process_special_word(comment) for comment in comment_list]
            comment_list = [normalize_repeated_characters(comment) for comment in comment_list]
            comment_list = [process_postag_thesea(comment) for comment in comment_list]
            new_comments = loaded_vectorizer.transform(comment_list)
            preds = loaded_log_model_word_balance.predict(new_comments)
            comment_df['Predict'] = preds
            st.write(comment_df)

    elif type == "Upload review file":
        st.markdown("###  Upload review file ###")
        uploaded_file = st.file_uploader("Please upload 'csv' or 'txt' file", type=["csv", "txt"])

        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                comment_df = pd.read_csv(uploaded_file)
            else:
                comment_df = pd.read_csv(uploaded_file, delimiter="\t", header=None, names=["Comment"])

            first_column_name = comment_df.columns[0]
            comment_df.rename(columns={first_column_name: "Comment"}, inplace=True)

            comment_list = comment_df["Comment"].tolist()
            comment_list = [process_text(comment, data['emojicon'], data['teencode'], data['wrong_words']) for comment in comment_list]
            comment_list = [covert_unicode(comment) for comment in comment_list]
            comment_list = [process_special_word(comment) for comment in comment_list]
            comment_list = [normalize_repeated_characters(comment) for comment in comment_list]
            comment_list = [process_postag_thesea(comment) for comment in comment_list]
            new_comments = loaded_vectorizer.transform(comment_list)
            preds = loaded_log_model_word_balance.predict(new_comments)
            comment_df['Predict'] = preds
            st.write(comment_df)

elif choice == 'Restaurant Information':
    st.subheader("Restaurant Information")
    type = st.radio("", options=["Search Information", "Compare Information"])

    if type == "Search Information":
        st.markdown("###  Input Restaurant ID ###")

        suggestion = st.selectbox(
            'Choose a suggested ID or "Manual Input"',
             options=[str(id) for id in suggested_ids] + ["Manual Input"]
        )
        
        if suggestion == "Manual Input":
            id = st.text_input('Restaurant ID')
        else:
            id = suggestion

        if st.button('Search'):
            try:
                analyze_general(df, int(id))
            except ValueError:
                st.error("Please enter a valid number for the Restaurant ID.")

    if type == "Compare Information":
        st.markdown("### Input Restaurant ID ###")

        suggestion1 = st.selectbox(
            'Choose a suggested ID 1 or "Manual Input"',
             options=[str(id) for id in suggested_ids] + ["Manual Input"]
        )
        
        if suggestion1 == "Manual Input":
            id1 = st.text_input('Restaurant 1 ID')
        else:
            id1 = suggestion1

        suggestion2 = st.selectbox(
            'Choose a suggested ID 2 or "Manual Input"',
             options=[str(id) for id in suggested_ids] + ["Manual Input"]
        )
        
        if suggestion2 == "Manual Input":
            id2 = st.text_input('Restaurant 2 ID')
        else:
            id2 = suggestion2

        if st.button('Compare'):
            try:
                col1, col2 = st.columns(2)

                with col1:
                    analyze_general(df, int(id1))

                with col2:
                    analyze_general(df, int(id2))
            except ValueError:
                st.error("Please enter a valid number for the Restaurant ID.")

elif choice == 'About Us':
    st.subheader("About Us")

    st.write("<br>", unsafe_allow_html=True)
    
    st.markdown('''
    **Project Sentiment Analysis

    - Nguyen Thanh Long - 26thang6@gmail.com
    ''')

    st.write("<br>", unsafe_allow_html=True)

    st.markdown('''
    The project uses a Logistic Regression model.
    ''')

    st.image('confusion_matrix.png')

    
    st.write("<br><br><br>", unsafe_allow_html=True)
    
    st.image('thankyou.png', use_column_width=True)

    st.markdown('''

    Any feedback can be filled out on the Home Page or sent to our email.

    Refer to the source code at the [GitHub link](https://github.com/26thang6/long-project3.git).
    ''')
