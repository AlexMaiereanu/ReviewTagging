import numpy as np


def convert_string_to_list(input):
    output = input.split('[')[1].split(']')[0].split(' ')
    output = [float(x) for x in output if x != '']
    return np.asarray(output)

def read_reviews_data():
    embeddings = {}
    with open('word_embeddings_spelled.txt') as f:
        content = f.readlines()
        content = [word[0: len(word) - 1] for word in content]
        index = 0
        length = len(content)

        while (index < length):
            word = content[index]
            index += 1
            embeddings[word] = ""

            while ']' not in content[index]:
                embedding = content[index]
                embeddings[word] += embedding
                index += 1

            embedding = content[index]
            embeddings[word] += embedding
            embeddings[word] = convert_string_to_list(embeddings[word])
            index += 1
    return embeddings


def get_nearest_word(orig_word, embeddings, previous_words):
    minim = 1234567
    new_word = ""
    for word, word_vec in embeddings.items():
        if word == orig_word:
            continue

        dist = np.linalg.norm(embeddings[orig_word] - word_vec)

        if dist < minim and word not in previous_words:
            minim = dist
            new_word = word

    return new_word


def get_k_nearest_words(k, orig_word, embeddings):
    results = []
    first_word = get_nearest_word(orig_word, embeddings, [])
    results.append(first_word)

    for i in range(k - 1):
        next_word = get_nearest_word(orig_word, embeddings, results)
        results.append(next_word)

    return results


def get_all_neighbors(embeddings):
    neighbors = {}
    for index, word in enumerate(embeddings):
        print(index)
        k_neighbors = get_k_nearest_words(4, word, embeddings)
        neighbors[word] = k_neighbors

    return neighbors


def print_neighbors_to_file(neighbors):
    out_file = open('all_neighbors.txt', 'w')

    for word, neighbors in neighbors.items():
        out_file.write('{} -> {}\n'.format(word, neighbors))
    out_file.close()


def load_neighbors():
    neighbors = {}
    with open('all_neighbors.txt') as f:
        content = f.readlines()
        content = [word[0: len(word) - 1] for word in content]
        for line in content:
            line = line.replace("'", "").replace("[", "").replace("]", "")
            word = line.split(" -> ")[0]
            closer_words = line.split(" -> ")[1].split(', ')
            neighbors[word] = closer_words
    return neighbors


def load_reviews():
    with open('all_reviews.txt') as f:
        content = f.readlines()
        content = [word[0: len(word) - 1] for word in content]
        return content


def remove_noise_words_from_review(review, embeddings):
    word_list = []
    words = review.split()
    for word in words:
        if word in embeddings and word not in word_list:
            word_list.append(word)
    return word_list


def get_nearest_tag_to_vector(input_vector, embeddings, previous_tags):
    minim = 1234567
    tag = ""
    for word, embedding in embeddings.items():
        dist = np.linalg.norm(embedding - input_vector)
        if dist < minim and word not in previous_tags:
            minim = dist
            tag = word

    return tag


def tag_review(word_list, embeddings, amenities):
    average_vector = np.zeros(128)
    for word in word_list:
        average_vector += embeddings[word]

    tag1 = get_nearest_tag_to_vector(average_vector, amenities, [])
    tag2 = get_nearest_tag_to_vector(average_vector, amenities, [tag1])
    tag3 = get_nearest_tag_to_vector(average_vector, amenities, [tag1, tag2])
    return [tag1, tag2, tag3]


def load_amenities():
    with open('amenities.txt') as f:
        content = f.readlines()
        content = [word[0: len(word) - 1] for word in content]
        return content


def build_amenities(amenities_words, embeddings):
    amenities = {}
    for amenity in amenities_words:
        amenities[amenity] = embeddings[amenity]
    return amenities


if __name__ == '__main__':

    embeddings = read_reviews_data()
    print("Got the embeddings")

    neighbors = load_neighbors()
    print("Neighbors loaded")

    amenities_words = load_amenities()
    print("Amenities loaded")

    amenities = build_amenities(amenities_words, embeddings)

    reviews = load_reviews()

    clean_reviews = []
    for review in reviews:
        clean_review = remove_noise_words_from_review(review, embeddings)
        clean_reviews.append(clean_review)

    # Good examples: 56, 89, 1189, 1891
    review_number = 1234

    print(reviews[review_number])
    tags = tag_review(clean_reviews[review_number], embeddings, amenities)
    print(tags)
