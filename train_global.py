import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from utility import generate_question
from utility import choose_question
from utility import retrain_on_subset
from utility import update_data 
from utility import get_best_match

df = pd.read_csv('onpiece.csv')
df_copy = df
X = df.drop('Character', axis = 1)
names = df['Character']

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
model = RandomForestClassifier(n_estimators=170, random_state=42)
model.fit(X,names)
importance = pd.Series(model.feature_importances_, index=X.columns)
importance.sort_values(ascending=False).to_csv('feature_importance.csv', header=True)

asked_questions = []
user_responses = pd.Series(-1, index=X.columns)
while df.shape[0] > 1 or (len(asked_questions) < 20): 
    new_importance = retrain_on_subset(df, asked_questions)
    if new_importance is not None:
        importance = new_importance

    current_quest = choose_question(importance, asked_questions,user_responses)
    
    if current_quest is None:
        break

    print(generate_question(current_quest))
    ans = input("(y/n)? ").lower()
    numeric_answer = 1 if ans == 'y' else 0
    user_responses[current_quest] = numeric_answer

    df = update_data(df, current_quest, numeric_answer)
    asked_questions.append(current_quest)

    X = df.drop('Character', axis=1)
    names = df['Character']
    
print(get_best_match(df_copy,user_responses))


