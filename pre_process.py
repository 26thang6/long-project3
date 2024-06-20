import pandas as pd
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import regex
from nltk.tokenize import sent_tokenize
from underthesea import pos_tag, word_tokenize
import nltk
nltk.download('punkt')

def analyze_general(df, restaurant_id):
    # Filter the data for the given restaurant ID
    restaurant_data = df[df['IDRestaurant'] == restaurant_id]
    
    if restaurant_data.empty:
        st.error("Restaurant ID not found.")
        return
    
    # Extract basic information
    if restaurant_data['Restaurant'].isna().any() or restaurant_data['Address'].isna().any() or \
       restaurant_data['Time'].isna().any() or restaurant_data['Price'].isna().any():
        st.error("This restaurant does not have enough basic information to show detailed data.")
        return
    
    name = restaurant_data['Restaurant'].iloc[0]
    address = restaurant_data['Address'].iloc[0]
    time = restaurant_data['Time'].iloc[0]
    price = restaurant_data['Price'].iloc[0]
    average_rating = round(restaurant_data['Rating'].mean(), 2)
    
    st.markdown(f"**Name:** {name}")
    st.markdown(f"**Address:** {address}")
    st.markdown(f"**Opening time:** {time}")
    st.markdown(f"**Price:** {price}")
    st.markdown(f"**Rating:** {average_rating} ⭐")
    
    # Analyze reviews
    positive_reviews = restaurant_data[restaurant_data['label'] == 'positive']
    negative_reviews = restaurant_data[restaurant_data['label'] == 'negative']
    
    num_positive_reviews = len(positive_reviews)
    num_negative_reviews = len(negative_reviews)
    
    st.markdown(f"**Number of Positive Reviews:** {num_positive_reviews}")
    st.markdown(f"**Number of Negative Reviews:** {num_negative_reviews}")
    
    if num_positive_reviews == 0 and num_negative_reviews == 0:
        st.error("This restaurant does not have enough review data to show detailed analysis.")
        return

    # Generate word clouds
    if not positive_reviews['clean_Comment'].empty:
        positive_text = " ".join(positive_reviews['clean_Comment'])
        wordcloud_positive = WordCloud(width=700, height=400, background_color='white').generate(positive_text)
        st.write("<div style='text-align: center; font-size: 14px; font-weight: bold;'>Positive Reviews Word Cloud</div>", unsafe_allow_html=True)
        st.image(wordcloud_positive.to_array(), use_column_width=True)
    
    if not negative_reviews['clean_Comment'].empty:
        negative_text = " ".join(negative_reviews['clean_Comment'])
        wordcloud_negative = WordCloud(width=700, height=400, background_color='black').generate(negative_text)
        st.write("<br>", unsafe_allow_html=True)  # Adding spacing between charts
        st.write("<div style='text-align: center; font-size: 14px; font-weight: bold;'>Negative Reviews Word Cloud</div>", unsafe_allow_html=True)
        st.image(wordcloud_negative.to_array(), use_column_width=True)

    # Generate bar charts for reviews by year and month
    if 'date' in restaurant_data.columns:
        restaurant_data['Year'] = pd.to_datetime(restaurant_data['date']).dt.year
        restaurant_data['Month'] = pd.to_datetime(restaurant_data['date']).dt.month
        
        yearly_reviews = restaurant_data.groupby(['Year', 'label']).size().unstack(fill_value=0)
        monthly_reviews = restaurant_data.groupby(['Month', 'label']).size().unstack(fill_value=0)
        
        if not yearly_reviews.empty:
            st.write("<br>", unsafe_allow_html=True)  # Adding spacing between charts
            fig, ax = plt.subplots(figsize=(10, 4))
            yearly_reviews.plot(kind='bar', ax=ax)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_title('Number of Reviews by Year', fontsize=10, loc='center')
            plt.xticks(rotation=0)  # Ensure x-axis labels are horizontal
            st.pyplot(fig)
        
        if not monthly_reviews.empty:
            st.write("<br>", unsafe_allow_html=True)  # Adding spacing between charts
            fig, ax = plt.subplots(figsize=(10, 4))
            monthly_reviews.plot(kind='bar', ax=ax)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_title('Number of Reviews by Month', fontsize=10, loc='center')
            plt.xticks(rotation=0)  # Ensure x-axis labels are horizontal
            st.pyplot(fig)
        
        # Generate bar chart for review relating to food by year
        if 'count_food' in restaurant_data.columns:
            yearly_food_reviews = restaurant_data.groupby(['Year', 'label'])['count_food'].sum().unstack(fill_value=0)
            if not yearly_food_reviews.empty:
                st.write("<br>", unsafe_allow_html=True)  # Adding spacing between charts
                fig, ax = plt.subplots(figsize=(10, 4))
                yearly_food_reviews.plot(kind='bar', ax=ax)
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.set_title('Number of Words Related to FOOD in Reviews by Year', fontsize=10, loc='center')
                plt.xticks(rotation=0)  # Ensure x-axis labels are horizontal
                st.pyplot(fig)
        
        # Generate bar chart for review relating to price by year
        if 'count_price' in restaurant_data.columns:
            yearly_price_reviews = restaurant_data.groupby(['Year', 'label'])['count_price'].sum().unstack(fill_value=0)
            if not yearly_price_reviews.empty:
                st.write("<br>", unsafe_allow_html=True)  # Adding spacing between charts
                fig, ax = plt.subplots(figsize=(10, 4))
                yearly_price_reviews.plot(kind='bar', ax=ax)
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.set_title('Number of Words Related to PRICE in Reviews by Year', fontsize=10, loc='center')
                plt.xticks(rotation=0)  # Ensure x-axis labels are horizontal
                st.pyplot(fig)
        
        # Generate bar chart for review relating to service by year
        if 'count_service' in restaurant_data.columns:
            yearly_service_reviews = restaurant_data.groupby(['Year', 'label'])['count_service'].sum().unstack(fill_value=0)
            if not yearly_service_reviews.empty:
                st.write("<br>", unsafe_allow_html=True)  # Adding spacing between charts
                fig, ax = plt.subplots(figsize=(10, 4))
                yearly_service_reviews.plot(kind='bar', ax=ax)
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.set_title('Number of Words Related to SERVICE in Reviews by Year', fontsize=10, loc='center')
                plt.xticks(rotation=0)  # Ensure x-axis labels are horizontal
                st.pyplot(fig)



def load_file(file_path):
    with open(file_path, 'r', encoding='utf8') as file:
        return file.read().split('\n')

def process_key_value_list(lines):
    return {line.split('\t')[0]: line.split('\t')[1] for line in lines if line.strip()}

def load_files():
    files = {
        'emojicon': 'emojicon.txt',
        'teencode': 'teencode.txt',
        'english_vnmese': 'english-vnmese.txt',
        'wrong_words': 'wrong-word.txt',
        'stopwords': 'vietnamese-stopwords.txt'
    }

    data = {}
    data['emojicon'] = process_key_value_list(load_file(files['emojicon']))
    data['teencode'] = process_key_value_list(load_file(files['teencode']))
    data['english_vnmese'] = process_key_value_list(load_file(files['english_vnmese']))
    data['wrong_words'] = [word for word in load_file(files['wrong_words']) if word.strip()]
    data['stopwords'] = [word for word in load_file(files['stopwords']) if word.strip()]

    return data




def process_text(text, emoji_dict, teen_dict, wrong_lst):
    if not isinstance(text, str):
        text = str(text)
    document = text.lower()
    document = document.replace("’",'')
    document = regex.sub(r'\.+', ".", document)
    new_sentence =''
    for sentence in sent_tokenize(document):
        # if not(sentence.isascii()):
        ###### CONVERT EMOJICON
        sentence = ''.join(emoji_dict[word]+' ' if word in emoji_dict else word for word in list(sentence))
        ###### CONVERT TEENCODE
        sentence = ' '.join(teen_dict[word] if word in teen_dict else word for word in sentence.split())
        ###### DEL Punctuation & Numbers
        pattern = r'(?i)\b[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]+\b'
        sentence = ' '.join(regex.findall(pattern,sentence))
        # ...
        ###### DEL wrong words
        sentence = ' '.join('' if word in wrong_lst else word for word in sentence.split())
        new_sentence = new_sentence+ sentence + '. '
    document = new_sentence
    #print(document)
    ###### DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    #...
    return document



# Chuẩn hóa unicode tiếng việt
def loaddicchar():
    uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
    unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"

    dic = {}
    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
        '|')
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        '|')
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic

# Đưa toàn bộ dữ liệu qua hàm này để chuẩn hóa lại
def covert_unicode(txt):
    dicchar = loaddicchar()
    return regex.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dicchar[x.group()], txt)




def process_special_word(text):
    # có thể có nhiều từ đặc biệt cần ráp lại với nhau
    new_text = ''
    text_lst = text.split()
    i= 0
    # không, chẳng, chả...
    if 'không' in text_lst:
        while i <= len(text_lst) - 1:
            word = text_lst[i]
            #print(word)
            #print(i)
            if  word == 'không':
                next_idx = i+1
                if next_idx <= len(text_lst) -1:
                    word = word +'_'+ text_lst[next_idx]
                i= next_idx + 1
            else:
                i = i+1
            new_text = new_text + word + ' '
    else:
        new_text = text
    return new_text.strip()




# Hàm để chuẩn hóa các từ có ký tự lặp
def normalize_repeated_characters(text):
    # Thay thế mọi ký tự lặp liên tiếp bằng một ký tự đó
    # Ví dụ: "ngonnnn" thành "ngon", "thiệtttt" thành "thiệt"
    return re.sub(r'(.)\1+', r'\1', text)



def process_postag_thesea(text):
    new_document = ''
    for sentence in sent_tokenize(text):
        sentence = sentence.replace('.','')
        ###### POS tag
        lst_word_type = ['N','Np','A','AB','V','VB','VY','R']
        # lst_word_type = ['A','AB','V','VB','VY','R']
        sentence = ' '.join( word[0] if word[1].upper() in lst_word_type else '' for word in pos_tag(process_special_word(word_tokenize(sentence, format="text"))))
        new_document = new_document + sentence + ' '
    ###### DEL excess blank space
    new_document = regex.sub(r'\s+', ' ', new_document).strip()
    return new_document




               
