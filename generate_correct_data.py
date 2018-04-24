import csv
import numpy as np
from io import StringIO
from autocorrect import spell


def load_reviews():
    # Return all the reviews for all the hotels
    all_reviews = []
    a = 0
    with open('Hotel_Reviews.csv') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            a += 1
            if a == 1:
                continue

            # Check if there are words for negative and positive reviews
            if int(row[7]) > 0:
                all_reviews.append(row[6].lower())
            if int(row[10]) > 0:
                all_reviews.append(row[9].lower())
    return all_reviews


def print_words_to_file(all_reviews):
    out_file = open('all_words_spelled.txt', 'w')

    for review in all_reviews:
        for word in review.split(" "):
            if word != ' ' and word != '':
                spelled_word = spell(word)
                if spelled_word != word:
                    print("Old word", word, "New word", spelled_word)
                out_file.write('{}\n'.format(spelled_word))
    out_file.close()

def write_reviews_file(all_reviews):
    out_file = open('all_reviews.txt', 'w')

    for review in all_reviews:
        new_review = ""
        for word in review.split(" "):
            if word != ' ' and word != '':
                spelled_word = spell(word)
                print("Old word", word, "New word", spelled_word)
                new_review = new_review + spelled_word + " "
        out_file.write('{}\n'.format(new_review))
    out_file.close()


if __name__ == '__main__':

    all_reviews = load_reviews()
    print("Reviews loaded")

    print_words_to_file(all_reviews=all_reviews)
    print("Spelled words printed to file")
    write_reviews_file(all_reviews=all_reviews)
    print("Reviews printed to file")
