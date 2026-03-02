import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from utility import (choose_question, generate_question, 
                     update_data, retrain_on_subset, get_best_match)

st.set_page_config(page_title="One Piece Akinator", page_icon="🏴‍☠️")
st.title("🏴‍☠️ One Piece Akinator")

if 'initialized' not in st.session_state:
    df = pd.read_csv('onpiece.csv')
    st.session_state.df = df
    st.session_state.df_copy = df.copy()
    st.session_state.asked_questions = []
    
    X = df.drop('Character', axis=1)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, df['Character'])
    st.session_state.importance = pd.Series(model.feature_importances_, index=X.columns)
    
    st.session_state.user_responses = pd.Series(-1, index=X.columns)
    st.session_state.game_over = False
    st.session_state.initialized = True

def process_answer(ans_value):
    curr_q = st.session_state.current_quest
    st.session_state.user_responses[curr_q] = ans_value
    st.session_state.asked_questions.append(curr_q)
    
    st.session_state.df = update_data(st.session_state.df, curr_q, ans_value)
    
    new_imp = retrain_on_subset(st.session_state.df, st.session_state.asked_questions)
    if new_imp is not None:
        st.session_state.importance = new_imp
            
    if st.session_state.df.shape[0] <= 1 and len(st.session_state.asked_questions) >= 15:
        st.session_state.game_over = True

if not st.session_state.game_over:
    q = choose_question(st.session_state.importance, st.session_state.asked_questions)
    st.session_state.current_quest = q
    
    if q:
        st.subheader(generate_question(q))
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("y", use_container_width=True):
                process_answer(1)
                st.rerun()
        with col2:
            if st.button("n", use_container_width=True):
                process_answer(0)
                st.rerun()
                
        st.divider()
        st.write(f"Вопросов задано: {len(st.session_state.asked_questions)}")
        st.write(f"Осталось кандидатов: {st.session_state.df.shape[0]}")
    else:
        st.session_state.game_over = True

else:
    winner = get_best_match(st.session_state.df_copy, st.session_state.user_responses)
    st.balloons()
    st.header(f"Я думаю, это... {winner}!")
    
    if st.button("Сыграть еще раз"):
        st.session_state.clear()
        st.rerun()