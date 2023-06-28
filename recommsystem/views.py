from django.shortcuts import render
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Create your views here.

def home(request):
    books_data = pd.read_csv('books.csv')
    books_data['index'] = range(1, len(books_data) + 1)
    features = ['title', 'subtitle', 'authors', 'categories', 'average_rating']
    for i in range(1, len(features) - 1):
        books_data[features[i]] = books_data[features[i]].fillna('')

    books_data['average_rating'] = books_data['average_rating'].fillna(
        books_data['average_rating'].median())

    combined_features = books_data['title'] + ' ' + books_data['subtitle'] + ' ' + \
        books_data['authors'] + ' ' + books_data['categories'] + \
        ' ' + str(books_data['average_rating'])

    vectorizer = TfidfVectorizer()
    feature_vector = vectorizer.fit_transform(combined_features)

    similarity = cosine_similarity(feature_vector)

    titles = []

    if request.method == "POST":
        title = request.POST.get('booktitle')
        book_name = title
        list_titles = books_data['title'].tolist()

        find_close_match = difflib.get_close_matches(book_name, list_titles)
        close_match = find_close_match[0]

        index_of_the_book = books_data[books_data.title ==
                                       close_match]['index'].values[0]
        similarity_score = list(enumerate(similarity[index_of_the_book]))

        sorted_similar_books = sorted(
            similarity_score, key=lambda x: x[1], reverse=True)

        # print the name of similar movies based on the index

        print('Books suggested for you : \n')

        i = 1
        for book in sorted_similar_books:
            index = book[0]
            title_from_index = books_data[books_data.index ==
                                          index]['title'].values[0]

            if (i < 30):
                titles.append(title_from_index)
                i += 1
        print(titles)
    return render(request, 'index.html', {'titles': titles})
