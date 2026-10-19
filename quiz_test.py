import streamlit as st

quiz = [
    {
        "question": "What does a Random Forest combine?",
        "options": [
            "One very deep tree",
            "Many independent trees",
            "Only linear models",
            "A single neuron",
        ],
        "answer": 1,
        "explanation": "It builds many trees on bootstrap samples and combines their predictions.",
    },
    {
        "question": "What does XGBoost add to each new tree?",
        "options": [
            "It corrects the errors of the previous trees",
            "It ignores the earlier trees",
            "It removes features",
            "It shuffles the labels",
        ],
        "answer": 0,
        "explanation": "Boosting builds trees one after another, each fixing earlier mistakes.",
    },
]

st.title("Quiz test")

picked = []
for i, q in enumerate(quiz):
    choice = st.radio(f"Q{i + 1}. {q['question']}", q["options"], index=None, key=f"q{i}")
    picked.append(choice)

if st.button("Check answers"):
    score = 0
    for i, q in enumerate(quiz):
        correct = q["options"][q["answer"]]
        if picked[i] == correct:
            score += 1
            st.success(f"Q{i + 1}: correct")
        else:
            st.error(f"Q{i + 1}: the answer is '{correct}'. {q['explanation']}")
    st.write(f"Score: {score}/{len(quiz)}")