from attention_pool import AttentionPool
from utils import clean_text
import tensorflow as tf
import streamlit as st
import plotly.graph_objects as go

custom_objects = {
    "AttentionPool": AttentionPool
}
model = tf.keras.models.load_model("best.keras", custom_objects=custom_objects)

def main():
    st.title("AI text detector")

    user_input = st.text_area("Enter your text:", "", height=250)

    if user_input:
        if len(user_input.split()) < 3:
            st.info("Enter at least 3 words.")
            return
         
        with st.spinner("Loading..."):
            try:
                text_input = clean_text(user_input)
                text_input = text_input.lower()
                
                input_ds = tf.data.Dataset.from_tensor_slices([text_input])
                input_ds = input_ds.batch(1, drop_remainder=False)
                pred_prob = model.predict(input_ds)[0][0]
                pred_prob = float(pred_prob)
                pred_label = int(pred_prob >= 0.5)

                if pred_label == 0:
                    st.success("The text is likely written by a human.")
                else:
                    st.warning("The text is likely written by AI.")
                
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=pred_prob * 100,
                    title={'text': "AI Probability (%)"},
                    gauge={'axis': {'range': [0, 100]}}
                ))
                st.plotly_chart(fig)
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
            

if __name__ == "__main__":
    main()