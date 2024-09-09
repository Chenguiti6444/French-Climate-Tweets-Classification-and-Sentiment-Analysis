### English Version

# French Global Warming Tweets Classification and Sentiment Analysis

## Overview
This project aimed to analyze and classify French tweets related to global warming and drought using Natural Language Processing (NLP) and Machine Learning techniques. The process involved scraping tweets using the `snscrape` Python library, preprocessing the text, and manually labeling the tweets for sentiment polarity (positive or negative). The dataset was visualized through word clouds and co-occurrence networks to identify common themes and word associations.

Several Machine Learning models, including Naive Bayes, Support Vector Machines (SVM), and Random Forest, were trained and evaluated for their performance in predicting tweet polarity. Techniques such as oversampling and grid search were used to enhance model performance. Additionally, pretrained Word2Vec embeddings were fine-tuned on the dataset to improve text representation and classification accuracy.

The ultimate goal was to build robust models capable of accurately classifying the sentiment of French tweets addressing climate change-related topics and to provide insights into public opinion within the French-speaking community on these critical issues.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Modeling Techniques](#modeling-techniques)
- [Results](#results)
- [Contributing](#contributing)
- [Contact](#contact)
- [License](#license)

## Project Structure
The project is organized as follows:
- `Twitter_Data_Scraping.ipynb`: Jupyter notebook for scraping tweets using the `snscrape` library.
- `Tweets_Preprocessing.ipynb`: Notebook for preprocessing the scraped tweets.
- `Tweets Polarity Predictions (TFIDF).ipynb`: Predicting tweet polarity using TF-IDF vectorization and machine learning models.
- `Tweets Polarity Predictions (Word2Vec).ipynb`: Predicting tweet polarity using fine-tuned Word2Vec embeddings and machine learning models.
- `Further_Analysis_of_Tweets.ipynb`: Additional analysis of tweets, including visualizations such as word clouds and co-occurrence networks.
- `negative_network.html` & `positive_network.html`: Visualizations of the co-occurrence networks for negative and positive tweets.
- CSV files (`train.csv`, `test.csv`, `tweets.csv`): Dataset files used for training and testing models.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/French-Climate-Tweets-Classification.git
   ```
2. Navigate to the project directory:
   ```bash
   cd French-Climate-Tweets-Classification
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Data Scraping**: Use the `Twitter_Data_Scraping.ipynb` notebook to scrape tweets using the `snscrape` library.
2. **Preprocessing**: Preprocess the tweets using the `Tweets_Preprocessing.ipynb` notebook. This step includes text cleaning, normalization, and labeling for sentiment polarity.
3. **Model Training and Evaluation**: Use the `Tweets Polarity Predictions (TFIDF).ipynb` and `Tweets Polarity Predictions (Word2Vec).ipynb` notebooks to train and evaluate machine learning models.
4. **Further Analysis**: Explore the dataset further using `Further_Analysis_of_Tweets.ipynb`, which includes creating word clouds and co-occurrence networks.

## Modeling Techniques
- **Text Preprocessing**: Involves cleaning and normalizing text to remove noise and standardize the input data.
- **TF-IDF Vectorization**: Converts text data into numerical features using Term Frequency-Inverse Document Frequency.
- **Word2Vec Embeddings**: Pretrained embeddings fine-tuned on the dataset to improve text representation.
- **Machine Learning Models**: Several models were trained, including:
  - **Logistic Regression**: [`LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
  - **MLP Classifier**: [`MLPClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
  - **SVC**: [`SVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
  - **AdaBoost Classifier**: [`AdaBoostClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
  - **Gradient Boosting**: [`GradientBoostingClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
  - **KNN**: [`KNeighborsClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
  - **Random Forest**: [`RandomForestClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
  - **Naive Bayes**: [`MultinomialNB`](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)
- **Oversampling & Grid Search**: Used to balance the dataset and optimize model hyperparameters.

## Results
The project successfully developed models that accurately classified the sentiment of French tweets on climate change and drought. The best performing model was **Random Forest** with a validation accuracy of **0.8977**. Visualizations such as word clouds and co-occurrence networks provided additional insights into the common themes and word associations within the tweets. The fine-tuned Word2Vec embeddings notably enhanced the classification accuracy, making the models more robust.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss changes.

## Contact
For questions or suggestions, please contact chenguiti.elmehdi@gmail.com.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


---
### Version Française
# Classification et Analyse des Sentiments des Tweets Français parlant du Réchauffement Climatique

## Vue d'ensemble
Ce projet a pour but d'analyser et de classifier les tweets Français concernant le réchauffement climatique et la sécheresse en utilisant des techniques de traitement du langage naturel (NLP) et d'apprentissage automatique. Le processus a impliqué l'extraction des tweets avec la bibliothèque Python `snscrape`, le prétraitement des tweets, et l'étiquetage manuel de ces derniers en deux classes (positif ou négatif). Les données ont été visualisées à travers des nuages de mots et des réseaux de cooccurrence pour identifier les termes récurrents et les associations de mots.

Divers modèles d'apprentissage automatique ont été entraînés et évalués pour leur capacité à prédire la polarité des tweets. Des techniques comme le suréchantillonnage et la recherche en grille ont été utilisées pour améliorer les performances des modèles. En outre, des embeddings Word2Vec pré-entraînés ont été ajustés sur le jeu de données pour affiner la représentation du texte et améliorer la précision de la classification.

L'objectif ultime était de développer des modèles robustes capables de classifier avec précision les sentiments des tweets Français abordant les sujets liés au changement climatique, et de fournir des informations sur l'opinion publique au sein de la communauté francophone sur ces enjeux critiques.

## Table des Matières
- [Vue d'ensemble](#vue-densemble)
- [Structure du Projet](#structure-du-projet)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Techniques de Modélisation](#techniques-de-modélisation)
- [Résultats](#résultats)
- [Contributions](#contributions)
- [Contact](#contact)
- [Licence](#licence)

## Structure du Projet
Le projet est organisé comme suit :
- `Twitter_Data_Scraping.ipynb` : Notebook Jupyter pour extraire des tweets en utilisant la bibliothèque `snscrape`.
- `Tweets_Preprocessing.ipynb` : Notebook pour le prétraitement des tweets extraits, comprenant le nettoyage, la normalisation et la standardisation des tweets.
- `Tweets Polarity Predictions (TFIDF).ipynb` : Prédiction de la polarité des tweets en utilisant la vectorisation TF-IDF et des modèles d'apprentissage automatique.
- `Tweets Polarity Predictions (Word2Vec).ipynb` : Prédiction de la polarité des tweets en utilisant des embeddings Word2Vec ajustés et des modèles d'apprentissage automatique.
- `Further_Analysis_of_Tweets.ipynb` : Analyse complémentaire des tweets, incluant des visualisations comme des nuages de mots et des réseaux de cooccurrence.
- `negative_network.html` & `positive_network.html` : Visualisations des réseaux de cooccurrence pour les tweets négatifs et positifs.
- Fichiers CSV (`train.csv`, `test.csv`, `tweets.csv`) : Fichiers de données utilisés pour entraîner et tester les modèles.

## Installation
1. Clonez le dépôt :
   ```bash
   git clone https://github.com/votreutilisateur/French-Climate-Tweets-Classification.git
   ```
2. Accédez au répertoire du projet :
   ```bash
   cd French-Climate-Tweets-Classification
   ```
3. Installez les dépendances requises :
   ```bash
   pip install -r requirements.txt
   ```

## Utilisation
1. **Extraction des Données** : Utilisez le notebook `Twitter_Data_Scraping.ipynb` pour extraire les tweets avec la bibliothèque `snscrape`.
2. **Prétraitement** : Prétraitez les tweets à l'aide du notebook `Tweets_Preprocessing.ipynb`. Cette étape comprend le nettoyage des tweets et leur normalisation.
3. **Entraînement et Évaluation des Modèles** : Utilisez les notebooks `Tweets Polarity Predictions (TFIDF).ipynb` et `Tweets Polarity Predictions (Word2Vec).ipynb` pour entraîner et évaluer les modèles d'apprentissage automatique.
4. **Analyse Supplémentaire** : Explorez davantage les données avec `Further_Analysis_of_Tweets.ipynb`, qui inclut la création de nuages de mots et de réseaux de cooccurrence.

## Techniques de Modélisation
- **Prétraitement des Textes** : Nettoyage et normalisation des textes pour éliminer le bruit et standardiser les données.
- **Vectorisation TF-IDF** : Conversion des données textuelles en caractéristiques numériques à l'aide de la vectorization TF-IDF.
- **Embeddings Word2Vec** : Embeddings pré-entraînés ajustés sur le jeu de données pour améliorer la représentation des tweets.
- **Modèles d'Apprentissage Automatique** : Plusieurs modèles ont été entraînés, notamment :
  - **Régression Logistique** : [`LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
  - **MLP Classifier** : [`MLPClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
  - **SVC** : [`SVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
  - **AdaBoost Classifier** : [`AdaBoostClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
  - **Gradient Boosting** : [`GradientBoostingClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
  - **KNN** : [`KNeighborsClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
  - **Random Forest** : [`RandomForestClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
  - **Naive Bayes** : [`MultinomialNB`](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)
- **Suréchantillonnage & Recherche en Grille** : Utilisés pour équilibrer le jeu de données et optimiser les hyperparamètres des modèles.

## Résultats
Le projet a permis de développer des modèles capables de classifier avec précision les sentiments des tweets en français sur le changement climatique et la sécheresse. Le modèle le plus performant a été **Random Forest** avec une précision de validation de **0.8977**. Les visualisations, telles que les nuages de mots et les réseaux de cooccurrence, ont apporté des informations supplémentaires sur les thèmes récurrents et les associations de mots dans les tweets. Les embeddings Word2Vec ajustés ont significativement amélioré la précision de la classification, rendant les modèles plus robustes.

## Contributions
Les contributions sont les bienvenues ! N'hésitez pas à soumettre une demande de tirage (pull request) ou à ouvrir un problème pour discuter des modifications.

## Contact
Pour toute question ou suggestion, veuillez contacter chenguiti.elmehdi@gmail.com.

## Licence
Ce projet est sous licence MIT. Voir le fichier [LICENCE](LICENSE) pour plus de détails.
