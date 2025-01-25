import gradio as gr
from transformers import pipeline

# Define available languages and their corresponding translation models
available_languages = {
    "German": "Helsinki-NLP/opus-mt-en-de",
    "French": "Helsinki-NLP/opus-mt-en-fr",
    "Tagalog": "Helsinki-NLP/opus-mt-en-tl",
}

# Define reverse translation models
reverse_languages = {
    "German": "Helsinki-NLP/opus-mt-de-en",
    "French": "Helsinki-NLP/opus-mt-fr-en",
    "Tagalog": "Helsinki-NLP/opus-mt-tl-en",
}

# Cache translation pipelines
translation_pipelines = {}
reverse_pipelines = {}

def loadPipelines():
    """Pre-load all translation pipelines for available languages."""
    for language, model_name in available_languages.items():
        try:
            translation_pipelines[language] = pipeline("translation", model=model_name)
        except Exception as e:
            print(f"Error loading pipeline for {language}: {e}")

    for language, model_name in reverse_languages.items():
        try:
            reverse_pipelines[language] = pipeline("translation", model=model_name)
        except Exception as e:
            print(f"Error loading reverse pipeline for {language}: {e}")

# Pre-load pipelines when the script is run
loadPipelines()

def getPipeline(targetLanguage, reverse=False):
    """Retrieve the translation pipeline for the target language."""
    return reverse_pipelines.get(targetLanguage) if reverse else translation_pipelines.get(targetLanguage)

# Define the translation function
def translateTransformers(fromText, targetLanguage, reverse=False):
    if not fromText.strip():
        return "Please enter some text to translate."
    
    pipeline = getPipeline(targetLanguage, reverse)
    if not pipeline:
        direction = "reverse" if reverse else ""
        return f"Translation pipeline not available for {direction} '{targetLanguage}'."
    
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
        gr.Checkbox(label="Reverse Translation (Target to English)", value=False),
    ],
    outputs=gr.Textbox(label="Translated Text"),
    examples=[
        ["Hello, how are you?", "German", False],
        ["I love programming!", "French", False],
        ["Good morning!", "Tagalog", False],
        ["Guten Tag!", "German", True],
        ["Bonjour!", "French", True],
        ["Magandang umaga!", "Tagalog", True],
    ],
    title="Multilingual Translator",
    description="Translate English text into different languages or reversely using Transformer models.",
    live=True,  # Enable live translation as the user types
)

# Launch the interface
if __name__ == "__main__":
    interface.launch(
        share=True  # Optional: Generate a shareable link
    )
