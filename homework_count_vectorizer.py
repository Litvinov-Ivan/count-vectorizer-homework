from typing import List


class CountVectorizer:
    """Создание терм-документной матрицы по набору текстов."""

    def __init__(self, texts: List[str] = None, features: List[str] = None):
        self.texts = texts
        self._features = features

    def fit_transform(self, texts: List[str]) -> List[List[int]]:
        """
        Принимает текстовый корпус.

        :return: Возвращает терм-документную матрицу.
        """

        self._features = []
        for word in ' '.join(texts).split():
            if word.lower() not in self._features:
                self._features.append(word.lower())

        texts_transformed = []
        for text in texts:
            text_transformed = []
            for word in self._features:
                text_transformed.append(text.lower().count(word))
            texts_transformed.append(text_transformed)
        return texts_transformed

    def get_feature_names(self) -> List[str]:
        """
        Ничего не принимает.

        :return: Возвращает список фичей (уникальных слов из корпуса).
        """

        return self._features


if __name__ == '__main__':
    corpus = [
        'Crock Pot Pasta Never boil pasta again',
        'Pasta Pomodoro Fresh ingredients Parmesan to taste'
    ]
    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(corpus)

    print(vectorizer.get_feature_names())
    print(count_matrix)
