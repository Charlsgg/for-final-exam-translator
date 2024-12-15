  import gradio as gr
  from transformers import pipeline

  # Define the translation pipeline
  translation_pipeline = pipeline('translation_en_to_de')

  # Translation function
  def translate_transformers(from_text):
      try:
          results = translation_pipeline(from_text)
          return results[0]['translation_text']
      except Exception as e:
          return f"Error: {str(e)}"

  # Gradio interface
  interface = gr.Interface(
      fn=translate_transformers,
      inputs=gr.Textbox(lines=2, placeholder="Enter English text to translate"),
      outputs=gr.Textbox(label="Translated Text"),
      title="English to German Translator",
      description="Translate English text into German using a pre-trained Hugging Face Transformer."
  )

  # Launch the interface
  interface.launch()
