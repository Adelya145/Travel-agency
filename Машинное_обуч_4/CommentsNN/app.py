import streamlit as st
import pandas as pd
import requests

st.set_page_config(
    page_title="Toxicity Classifier",
    page_icon="🔍",
    layout="centered"
)

API_URL = "http://localhost:8000"

st.title("🔍 Классификатор токсичности комментариев")
st.markdown("---")


def check_api_connection():
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


api_connected = check_api_connection()

if not api_connected:
    st.error("❌ API не доступно. Запустите FastAPI сервер")
    st.code("uvicorn api:app --reload")
    st.stop()

tab1, tab2 = st.tabs(["📊 Анализ базы данных", "✍️ Ручной ввод комментария"])

with tab1:
    st.header("Анализ комментариев из базы данных")

    if st.button("📊 Загрузить комментарии и предсказать токсичность", type="primary", use_container_width=True):
        with st.spinner("Анализ комментариев..."):
            try:
                response = requests.get(f"{API_URL}/predict-all")

                if response.status_code == 200:
                    results = response.json()

                    if results:
                        df = pd.DataFrame(results)

                        df['original_label'] = df['comment_toxic'].map({0: 'Хороший', 1: 'Плохой'})
                        df['predicted_label'] = df['predicted_class'].map({0: 'Хороший', 1: 'Плохой'})
                        df['probability_value'] = df['predicted_probability'] * 100
                        df['correct'] = df['is_correct'].map({True: '✅ Да', False: '❌ Нет'})

                        total = len(df)
                        correct = df['is_correct'].sum()
                        accuracy = (correct / total * 100) if total > 0 else 0

                        # Метрики
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Всего комментариев", total)
                        with col2:
                            st.metric("Совпадений", f"{correct}/{total}")
                        with col3:
                            st.metric("Точность", f"{accuracy:.1f}%")

                        st.markdown("---")

                        for idx, row in df.iterrows():
                            if row['predicted_class'] == 1:
                                box_color = "#ffebee"
                                badge_color = "#f44336"
                                badge_text = "⚠️ ТОКСИЧНЫЙ"
                                prob_display = row['probability_value']
                                prob_text = f"{prob_display:.1f}% токсичности"
                            else:
                                box_color = "#e8f5e8"
                                badge_color = "#4caf50"
                                badge_text = "✅ ХОРОШИЙ"
                                prob_display = 100 - row['probability_value']
                                prob_text = f"{prob_display:.1f}% нетоксичности"

                            st.markdown(f"""
                            <div style="
                                background-color: {box_color};
                                padding: 15px;
                                border-radius: 10px;
                                margin-bottom: 15px;
                                border-left: 5px solid {badge_color};
                            ">
                                <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                                    <span style="font-weight: bold;">ID: {row['comment_id']}</span>
                                    <span style="
                                        background-color: {badge_color};
                                        color: white;
                                        padding: 3px 10px;
                                        border-radius: 15px;
                                        font-size: 12px;
                                    ">{badge_text}</span>
                                </div>
                                <p style="font-size: 16px; margin-bottom: 10px;">"{row['comment_text']}"</p>
                                <div style="
                                    background-color: white;
                                    padding: 10px;
                                    border-radius: 5px;
                                    font-size: 14px;
                                ">
                                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                                        <div><span style="color: #666;">Оригинал:</span> <strong>{row['original_label']}</strong></div>
                                        <div><span style="color: #666;">Предсказание:</span> <strong>{row['predicted_label']}</strong></div>
                                        <div><span style="color: #666;">Вероятность:</span> <strong>{prob_text}</strong></div>
                                        <div><span style="color: #666;">Совпадение:</span> <strong>{row['correct']}</strong></div>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                        st.success("✅ Анализ завершен!")
                    else:
                        st.warning("В базе данных нет комментариев")
                else:
                    st.error("Ошибка при получении данных от API")

            except Exception as e:
                st.error(f"Ошибка: {e}")

with tab2:
    st.header("✍️ Добавить новый комментарий")

    st.markdown("""
    Напишите комментарий, и он будет автоматически проанализирован и сохранен в базе данных.
    """)

    user_comment = st.text_area(
        "Текст комментария:",
        height=120,
        placeholder="Введите текст комментария...",
        key="manual_input"
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button("💾 Проанализировать и сохранить", type="primary", use_container_width=True)

    if analyze_button and user_comment.strip():
        with st.spinner("Анализ и сохранение..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict-single",
                    json={"text": user_comment}
                )

                if response.status_code == 200:
                    result = response.json()

                    prob_value = result['probability']
                    toxic_class = result['predicted_class']

                    save_response = requests.post(
                        f"{API_URL}/save-comment",
                        json={
                            "text": user_comment,
                            "toxicity": toxic_class
                        }
                    )

                    if save_response.status_code == 200:
                        save_result = save_response.json()

                        if toxic_class == 1:
                            st.error(f"⚠️ ТОКСИЧНЫЙ КОММЕНТАРИЙ")
                            st.markdown(f"**Вероятность токсичности:** {prob_value:.1%}")
                            st.progress(prob_value)
                        else:
                            st.success(f"✅ НЕТОКСИЧНЫЙ КОММЕНТАРИЙ")
                            nontoxic_prob = 1 - prob_value
                            st.markdown(f"**Вероятность нетоксичности:** {nontoxic_prob:.1%}")
                            st.progress(nontoxic_prob)

                        st.success(f"✅ Комментарий сохранен в БД с ID: {save_result['comment_id']}")
                    else:
                        st.error("❌ Ошибка при сохранении в БД")
                else:
                    st.error("❌ Ошибка при анализе комментария")

            except Exception as e:
                st.error(f"Ошибка: {e}")

    elif analyze_button:
        st.warning("Введите текст комментария")