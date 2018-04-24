import csv
import numpy as np
from io import StringIO

def load_reviews():
    # Return all the reviews for all the hotels
    all_reviews = []
    a = 0
    with open('Hotel_Reviews.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            a += 1
            if a == 1:
                continue

            # Check if there are words for negative and positive reviews
            if row[7] > 0:
                all_reviews.append(row[6].lower())
            if row[10] > 0:
                all_reviews.append(row[9].lower())
    return all_reviews


def print_words_to_file(all_reviews):
    out_file = open('all_words.txt', 'w')

    for review in all_reviews:
        for word in review.split(" "):
            if word != ' ' and word != '':
                out_file.write('{}\n'.format(word))
    out_file.close()


if __name__ == '__main__':

    all_reviews = load_reviews()
    print("Reviews loaded")

    print_words_to_file(all_reviews=all_reviews)
    print("Words printed to file")
