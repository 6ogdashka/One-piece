import random
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity

EXCLUSIVE_GROUPS = {
    'blood': ['Blood_F', 'Blood_S', 'Blood_S (RH-)', 'Blood_X', 'Blood_XF'],
    'age': ['age_under_25', 'age_25_60', 'age_over_60'],
    'height': ['height_short', 'height_tall', 'height_giant'],
    'fruit_type': [
        'FruitType_Ancient Zoan', 'FruitType_Artificial Zoan', 'FruitType_Logia', 
        'FruitType_Mythical Zoan', 'FruitType_Paramecia', 'FruitType_Zoan'
    ],
    'origin': 'Origin_'
}

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

def choose_question(importance_series, asked_questions, user_responses, randomize=False):
    exclude_list = set(asked_questions)
    for key, value in EXCLUSIVE_GROUPS.items():
        if isinstance(value, list):
            if any(user_responses.get(col) == 1 for col in value if col in user_responses.index):
                exclude_list.update(value)
        elif isinstance(value, str):
            group_cols = [c for c in user_responses.index if c.startswith(value)]
            if any(user_responses.get(col) == 1 for col in group_cols):
                exclude_list.update(group_cols)

    available_features = [
        f for f in importance_series.sort_values(ascending=False).index 
        if f not in exclude_list
    ]
    if not available_features:
        return None
    if not randomize:
        return available_features[0]
    top_pool = available_features[:3]
    return random.choice(top_pool)

def retrain_on_subset(subset_df, columns_to_exclude):
    if len(subset_df) < 2: 
        return None
    X_sub = subset_df.drop(['Character'] + list(columns_to_exclude), axis=1, errors='ignore')
    if X_sub.empty:
        return None
    y_sub = subset_df['Character']
    model = RandomForestClassifier(n_estimators=60, random_state=42)
    model.fit(X_sub, y_sub)
    
    importance = pd.Series(model.feature_importances_, index=X_sub.columns) 
    total_chars = len(subset_df)
    categories = {
        'is_alive': 1.0,
        'has_devil_fruit': 1.5,
        'FruitType_': 1.4,
        'height_': 1.2,
        'age_': 0.9,
        'Arc_': 1.0,
        'Origin_': 0.6,
        'Blood_': 1.4
    }
    for feature in importance.index:
        pos_count = subset_df[feature].sum()
        ratio = pos_count / total_chars
        balance_coeff = 1.0 - abs(0.5 - ratio)
        category_weight = 1.0
        for weight_key, weight_value in categories.items():
            if weight_key in feature:
                category_weight = weight_value
                break
        importance[feature] *= (balance_coeff * category_weight)
    return importance

def update_data(current_df, column, answer):
    return current_df[current_df[column] == answer].copy()

def get_best_match(working_df, user_answers_profile):
    answered_cols = user_answers_profile[user_answers_profile != -1].index
    if len(answered_cols) == 0 or working_df.empty:
        return None
    user_vec = user_answers_profile[answered_cols].values.reshape(1, -1)
    cand_vecs = working_df[answered_cols].values
    scores = cosine_similarity(user_vec, cand_vecs)[0]
    best_idx = scores.argmax()
    return working_df.iloc[best_idx]['Character']