from typing import List
from collections import Counter, OrderedDict


class CountVectorizer:
    """
    Класс для создания терм-документной матрицы по набору текстов.
    """

    def __init__(self):
        self.features = []

    def fit_transform(self, texts: List[str] = ['']) -> List[List[int]]:
        """
        Принимает текстовый корпус и создаёт по нему
        терм-документную матрицу вместе со словарем
        уникальных терминов.

        Возвращает терм-документную матрицу.
        """
        texts_transformed = []
        words_dict = OrderedDict()
        for text in texts:
            text = text.lower().split()
            words_dict.update(zip(text, [0] * len(text)))
            texts_transformed.append(Counter(text))
        self.features = list(words_dict.keys())
        fit_transformed = [
            [
                text[word] for word in self.features
            ] for text in texts_transformed
        ]
        return fit_transformed

    def get_feature_names(self) -> List[str]:
        """
        Ничего не принимает, при вызове
        возвращает список фичей (уникальных слов из корпуса).
        """
        return self.features


if __name__ == '__main__':
    corpus = [
        'Crock Pot Pasta Never boil pasta again',
        'Pasta Pomodoro Fresh ingredients Parmesan to taste'
    ]
    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(corpus)

    print(vectorizer.get_feature_names())
    print(count_matrix)

    vectorizer_2 = CountVectorizer()
    print(vectorizer_2.fit_transform())
