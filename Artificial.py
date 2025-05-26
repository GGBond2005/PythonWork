# -*- coding: utf-8 -*-
"""
IMDB情感分类完整实验代码（已修复特征维度问题）
环境要求：Python 3.8+, 需要安装以下包：
pip install numpy pandas matplotlib scikit-learn nltk tensorflow transformers
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_selection import SelectKBest, chi2
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from transformers import pipeline
import warnings
import joblib

warnings.filterwarnings('ignore')


# ========================
# 数据加载与预处理
# ========================
def load_data():
    (train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.imdb.load_data(num_words=10000)
    all_data = np.concatenate([train_data, test_data])
    all_labels = np.concatenate([train_labels, test_labels])

    X_temp, X_test, y_temp, y_test = train_test_split(
        all_data, all_labels, test_size=0.2, random_state=42)
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42)

    return (X_train, y_train), (X_dev, y_dev), (X_test, y_test)


class TextPreprocessor:
    def __init__(self):
        self.word_index = tf.keras.datasets.imdb.get_word_index()
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.reverse_word_index = {v: k for k, v in self.word_index.items()}

    def decode_review(self, indices):
        return ' '.join([self.reverse_word_index.get(i - 3, '?') for i in indices])

    def preprocess(self, text_indices):
        text = self.decode_review(text_indices)
        text = text.replace('<br />', ' ').lower()
        words = text.split()
        processed = [
            self.lemmatizer.lemmatize(w)
            for w in words
            if w.isalpha() and w not in self.stop_words
        ]
        return ' '.join(processed)


# ========================
# 特征工程（核心修复）
# ========================
class FeatureEngineer:
    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.selector_200 = None
        self.selector_2000 = None

    def tfidf_transform(self, texts):
        return self.vectorizer.fit_transform(texts)

    def feature_selection_train(self, X, y, k=200):
        """训练并保存特征选择器"""
        if k == 200:
            self.selector_200 = SelectKBest(chi2, k=k).fit(X, y)
            return self.selector_200.transform(X)
        elif k == 2000:
            self.selector_2000 = SelectKBest(chi2, k=k).fit(X, y)
            return self.selector_2000.transform(X)
        else:
            raise ValueError("不支持的k值")

    def feature_selection_transform(self, X, k=200):
        """应用训练好的特征选择器"""
        if k == 200 and self.selector_200:
            return self.selector_200.transform(X)
        elif k == 2000 and self.selector_2000:
            return self.selector_2000.transform(X)
        else:
            raise ValueError("特征选择器未初始化")


# ========================
# 模型构建（优化）
# ========================
class SentimentAnalyzer:
    def __init__(self):
        self.models = {
            'NaiveBayes': MultinomialNB(),
            'LogisticRegression': LogisticRegression(max_iter=1000),
            'LogReg_200': LogisticRegression(max_iter=1000),
            'LogReg_2000': LogisticRegression(max_iter=1000)
        }

    def train(self, X_train_full, X_train_200, X_train_2000, y_train):
        """明确分配训练数据"""
        self.models['NaiveBayes'].fit(X_train_full, y_train)
        self.models['LogisticRegression'].fit(X_train_full, y_train)
        self.models['LogReg_200'].fit(X_train_200, y_train)
        self.models['LogReg_2000'].fit(X_train_2000, y_train)

    def evaluate(self, model_name, X, y):
        model = self.models[model_name]
        y_pred = model.predict(X)
        print(f"\n{model_name} 分类报告:")
        print(classification_report(y, y_pred))
        return y_pred


# ========================
# 可视化工具（不变）
# ========================
class Visualizer:
    @staticmethod
    def plot_metrics(models, scores):
        plt.figure(figsize=(10, 6))
        plt.bar(models, scores)
        plt.title('模型性能对比')
        plt.ylabel('准确率')
        plt.ylim(0.7, 1.0)
        plt.savefig('model_comparison.png')
        plt.show()

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, model_name):
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(f'{model_name} 混淆矩阵')
        plt.savefig(f'cm_{model_name}.png')
        plt.show()


# ========================
# 主程序（关键修复）
# ========================
def main():
    # 数据加载
    (X_train, y_train), (X_dev, y_dev), (X_test, y_test) = load_data()

    # 数据预处理
    preprocessor = TextPreprocessor()
    print("\n预处理示例:")
    print("原始数据:", X_train[0][:10])
    print("解码文本:", preprocessor.decode_review(X_train[0])[:200] + "...")
    print("预处理后:", preprocessor.preprocess(X_train[0])[:200] + "...")

    # 预处理所有数据
    print("\n预处理数据中...")
    train_texts = [preprocessor.preprocess(d) for d in X_train]
    dev_texts = [preprocessor.preprocess(d) for d in X_dev]
    test_texts = [preprocessor.preprocess(d) for d in X_test]

    # 特征工程
    fe = FeatureEngineer(max_features=5000)

    # TF-IDF转换
    X_train_tf = fe.tfidf_transform(train_texts)
    X_dev_tf = fe.vectorizer.transform(dev_texts)
    X_test_tf = fe.vectorizer.transform(test_texts)

    # 特征选择（训练阶段）
    X_train_200 = fe.feature_selection_train(X_train_tf, y_train, k=200)
    X_train_2000 = fe.feature_selection_train(X_train_tf, y_train, k=2000)

    # 特征选择（转换阶段）
    X_dev_200 = fe.feature_selection_transform(X_dev_tf, k=200)
    X_dev_2000 = fe.feature_selection_transform(X_dev_tf, k=2000)
    X_test_200 = fe.feature_selection_transform(X_test_tf, k=200)
    X_test_2000 = fe.feature_selection_transform(X_test_tf, k=2000)

    # 维度验证（调试用）
    print("\n特征维度验证:")
    print(f"原始维度: {X_train_tf.shape[1]}")
    print(f"200维: {X_train_200.shape[1]}, {X_dev_200.shape[1]}")
    print(f"2000维: {X_train_2000.shape[1]}, {X_dev_2000.shape[1]}")

    # 模型训练
    analyzer = SentimentAnalyzer()
    analyzer.train(X_train_tf, X_train_200, X_train_2000, y_train)

    # 在开发集上评估
    model_names = ['NaiveBayes', 'LogisticRegression', 'LogReg_200', 'LogReg_2000']
    accuracies = []

    for name in model_names:
        # 精确匹配特征集
        if name == 'LogReg_200':
            X_eval = X_dev_200
        elif name == 'LogReg_2000':
            X_eval = X_dev_2000
        else:
            X_eval = X_dev_tf

        y_pred = analyzer.evaluate(name, X_eval, y_dev)
        acc = np.mean(y_pred == y_dev)
        accuracies.append(acc)
        Visualizer.plot_confusion_matrix(y_dev, y_pred, name)

    # 性能可视化
    Visualizer.plot_metrics(model_names, accuracies)

    # 保存特征工程器
    joblib.dump(fe, 'feature_engineer.joblib')
    print("\n✅ 特征工程器已保存")


if __name__ == "__main__":
    main()