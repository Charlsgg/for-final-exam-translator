import gradio as gr
from transformers import pipeline

# Define available languages and their corresponding translation models
available_languages = {
    "German": "translation_en_to_de",
    "French": "translation_en_to_fr",
}

# Cache translation pipelines
translation_pipelines = {}

def loadPipelines():
    """Pre-load all translation pipelines for available languages."""
    for language, model_name in available_languages.items():
        try:
            translation_pipelines[language] = pipeline(model_name)
        except Exception as e:
            print(f"Error loading pipeline for {language}: {e}")

# Pre-load pipelines when the script is run
loadPipelines()

def getPipeline(targetLanguage):
    """Retrieve the translation pipeline for the target language."""
    return translation_pipelines.get(targetLanguage)

# Define the translation function
def translateTransformers(fromText, targetLanguage):
    if not fromText.strip():
        return "Please enter some text to translate."
    
    pipeline = getPipeline(targetLanguage)
    if not pipeline:
        return f"Translation pipeline not available for '{targetLanguage}'."
    
    try:
        results = pipeline(fromText)
        return results[0]['translation_text']
    except Exception as e:
        return f"Error during translation: {e}"

# Define the Gradio interface with the "Ocean" theme
interface = gr.Interface(
    fn=translateTransformers,
    inputs=[
        gr.Textbox(lines=2, placeholder='Text to translate', label='Input Text'),
        gr.Dropdown(
            choices=list(available_languages.keys()), 
            label="Target Language",
            value=list(available_languages.keys())[0]  # Set default language
        ),
    ],
    outputs=gr.Textbox(label="Translated Text"),
    examples=[
        ["Hello, how are you?", "German"],
        ["I love programming!", "French"],
    ],
    title="Multilingual Translator",
    description="Translate English text into different languages using Transformer models.",
    live=True,  # Enable live translation as the user types
)

# Launch the interface
if __name__ == "__main__":
    interface.launch(
        share=True  # Optional: Generate a shareable link
    )
