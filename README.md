# SENTIMENT-ANALYSIS-WITH-NLP

COMPANY : CODETECH IT SOLUTIONS

NAME : ANAS AHMAD

INTERN ID : CT06DG87

DOMAIN : MACHINE LEARNING

DURATION : 6 WEEKS

MENTOR : NEELA SANTHOSH KUMAR

This project aimed to build a sentiment analysis system capable of classifying customer reviews as positive or negative using natural language processing (NLP) techniques and a machine learning algorithm. The entire development process was carried out using Python in a Jupyter Notebook environment, which provided an interactive coding interface, allowing for step-by-step analysis, debugging, and visualization of results. The initial step involved importing and exploring the dataset using the Pandas library. Pandas was instrumental in data manipulation, including loading the CSV file, checking for null values, understanding data distribution, and viewing samples of the review texts. Once the data was verified and cleaned, the core part of the task—text preprocessing—was handled using the Natural Language Toolkit (NLTK), one of the most widely used libraries in NLP. The preprocessing stage involved several steps: converting all text to lowercase for uniformity, tokenizing the text into individual words using nltk.word_tokenize(), removing stopwords using NLTK’s predefined stopword list to eliminate common but insignificant words (like "the", "is", "and"), and applying lemmatization to reduce words to their base or dictionary forms (for example, "running" becomes "run"). These steps helped convert raw text into a form suitable for machine learning algorithms.

After preprocessing, the next step was to convert the cleaned text into numerical features. For this, the TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer from the sklearn.feature_extraction.text module was used. TF-IDF assigns weights to words based on their frequency across the corpus and their importance in a specific document, thus helping to capture relevant textual features. The resulting TF-IDF matrix served as the input feature set for the classification model. The dataset was then split into training and testing sets using train_test_split() from sklearn.model_selection to ensure the model could be trained on one portion and evaluated on another, ensuring unbiased performance estimation. The classification model selected for this task was Logistic Regression, implemented using the sklearn.linear_model.LogisticRegression class. Logistic Regression is a simple yet powerful linear model ideal for binary classification problems like sentiment analysis. The model was trained using the TF-IDF features of the training set and then tested on the unseen testing set.

To evaluate the model’s performance, the accuracy score was calculated along with a confusion matrix using sklearn.metrics. The confusion matrix, which presents the number of correct and incorrect predictions categorized by class, was visualized using Matplotlib, providing a graphical representation of model performance. Each block of the confusion matrix (True Positive, False Positive, True Negative, and False Negative) was labeled for easier understanding. Matplotlib also helped in plotting other visual elements throughout the project for enhanced interpretability. Overall, the tools and libraries used in this project included Python, Pandas for data manipulation, NLTK for text preprocessing, Scikit-learn for machine learning and evaluation, Matplotlib for visualization, and Jupyter Notebook as the coding platform. These tools collectively supported the development of a robust sentiment analysis pipeline, showcasing the practical integration of NLP and machine learning for real-world text classification problems.

OUTPUT : 

![image](https://github.com/ANASAHMAD-CLOUD/SENTIMENT-ANALYSIS-WITH-NLP/blob/main/image.png)
