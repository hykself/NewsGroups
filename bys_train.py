from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from textPreprocessor import TextPreprocessor
import joblib

def train_model(preprocessed_texts, labels):
    # 特征工程
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(preprocessed_texts)
    
    # 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42
    )
    
    # 模型训练
    classifier = MultinomialNB(alpha=0.1)
    classifier.fit(X_train, y_train)
    
    # 模型评估
    y_pred = classifier.predict(X_test)
    # 在训练函数中已有评估代码
    print(classification_report(y_test, y_pred))

    # 输出示例：
    #               precision    recall  f1-score   support
    #  comp.graphics       0.89      0.87      0.88       319
    #      sci.med       0.92      0.85      0.88       298
    # ...

    # 保存模型
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    joblib.dump(classifier, 'nb_classifier.pkl')
    
    return classifier, vectorizer

if __name__ == "__main__":
    # 初始化预处理类
    preprocessor = TextPreprocessor(
        dataset_path="F:/study/three/NLP/pythonProject3/20_newsgroups",
        use_stopwords=True,
        use_stemming=True
    )
    
    # 加载数据
    texts, labels = preprocessor.load_data()
    
    # 执行训练
    train_model(texts, labels)
