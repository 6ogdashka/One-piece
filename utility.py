import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity


def generate_question(column_name):
    clean_name = column_name.replace('"', '').strip()

    if clean_name == 'is_alive':
        return "Is your character currently alive?"
    
    if 'FruitType' in clean_name:
        fruit_type = clean_name.replace('FruitType_', '')
        return f"Is your character a {fruit_type} user?"
    
    if 'Arc' in clean_name:
        arc_name = clean_name.replace('Arc_', '')
        return f"Did your character first appear in the {arc_name} arc?"

    return f"Is your character a {clean_name}?"

def choose_question(importance_series, asked_questions):
    sorted_features = importance_series.sort_values(ascending=False).index
    for feat in sorted_features:
        if feat not in asked_questions:
            return feat
    return None

def retrain_on_subset(subset_df, current_asked):
    if len(subset_df) < 5: 
        return None
    
    X_sub = subset_df.drop('Character', axis=1)
    y_sub = subset_df['Character']
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_sub, y_sub)
    
    importance = pd.Series(model.feature_importances_, index=X_sub.columns)
    return importance

def update_data(current_df, column, answer):
    new_df = current_df[current_df[column] == answer].copy()
    return new_df

def get_best_match(working_df, user_answers_profile):
    answered_cols = user_answers_profile[user_answers_profile != -1].index
    
    if len(answered_cols) == 0 or working_df.empty:
        return None

    user_vec = user_answers_profile[answered_cols].values.reshape(1, -1)
    cand_vecs = working_df[answered_cols].values
    
    scores = cosine_similarity(user_vec, cand_vecs)[0]
    best_idx = scores.argmax()
    
    return working_df.iloc[best_idx]['Character']