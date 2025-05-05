import os
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split

class TextPreprocessor:
    def __init__(self, dataset_path, use_stopwords=True, use_stemming=False):
        self.dataset_path = dataset_path
        try:
            self.stop_words = set(stopwords.words('english')) if use_stopwords else None
        except LookupError:
            print("NLTK停用词库未找到，请执行以下命令下载：\nimport nltk\nnltk.download('stopwords')")
            raise
        self.stemmer = PorterStemmer() if use_stemming else None

    def clean_text(self, text):
        # 移除特殊字符和数字
        text = re.sub(r'[^a-zA-Z\']', ' ', text)
        # 转换为小写
        text = text.lower()
        return text

    def tokenize(self, text):
        return word_tokenize(text)

    def remove_stopwords(self, tokens):
        if self.stop_words:
            return [word for word in tokens if word not in self.stop_words]
        return tokens

    def apply_stemming(self, tokens):
        if self.stemmer:
            return [self.stemmer.stem(word) for word in tokens]
        return tokens

    def load_data(self):
        texts = []
        labels = []

        # 遍历20个类别文件夹
        for label in os.listdir(self.dataset_path):
            category_path = os.path.join(self.dataset_path, label)
            if os.path.isdir(category_path):
                # 读取每个类别下的所有文件
                for file_name in os.listdir(category_path):
                    file_path = os.path.join(category_path, file_name)
                    if os.path.isfile(file_path):
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                                # 预处理流程
                                cleaned = self.clean_text(content)
                                tokens = self.tokenize(cleaned)
                                filtered = self.remove_stopwords(tokens)
                                stemmed = self.apply_stemming(filtered)
                                texts.append(' '.join(stemmed))
                                labels.append(label)
                        except Exception as e:
                            print(f"Error reading {file_path}: {str(e)}")
        return texts, labels

# 示例用法
if __name__ == "__main__":
    # 添加NLTK资源下载检查
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("正在下载NLTK必要资源...")
        nltk.download('punkt')
        nltk.download('stopwords')

    # 初始化预处理器（请替换为实际路径）
    preprocessor = TextPreprocessor(
        dataset_path="F:/study/three/NLP/pythonProject3/20_newsgroups",
        use_stopwords=True,
        use_stemming=True
    )

    # 加载并处理数据
    processed_texts, labels = preprocessor.load_data()

    # 新增数据集划分（训练集:测试集=8:2）
    X_train, X_test, y_train, y_test = train_test_split(
        processed_texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels  # 保持类别分布一致
    )

    # 保存划分后的数据集
    def save_dataset(data, labels, filename):
        with open(filename, "w", encoding="utf-8") as f:
            for text, label in zip(data, labels):
                f.write(f"{label}\t{text}\n")

    save_dataset(X_train, y_train, "train_data.txt")
    save_dataset(X_test, y_test, "test_data.txt")

    # 后续可保存处理后的数据或直接用于训练
    # 例如保存到文件：
    with open("processed_data.txt", "w", encoding="utf-8") as f:
        for text, label in zip(processed_texts, labels):
            f.write(f"{label}\t{text}\n")
