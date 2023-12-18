import gradio as gr
from transformers import pipeline

# Load the summarization pipeline
pipe = pipeline("summarization", model="VictorNGomes/pttmario5")

def summarize_text(text):
    # Use the summarization pipeline to generate a summary
    summary = pipe(text, max_length=150, min_length=50, length_penalty=2.0, num_beams=4)[0]['summary_text']
    return summary

iface = gr.Interface(
    fn=summarize_text,
    inputs=gr.Textbox(),
    outputs=gr.Textbox(),
    live=True,
    analytics_enabled=False
)

iface.launch()