from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import gradio as gr
import mlflow
import mlflow.pytorch

# Load BLIP processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Function to generate captions for an input image
def generate_caption(img):
    with mlflow.start_run():  # Start an MLflow run
        try:
            # Convert input image to PIL Image
            img_input = Image.fromarray(img)

            # Log the input image as an artifact
            img_input.save("input_image.png")
            mlflow.log_artifact("input_image.png")

            # Process the image and generate a caption
            inputs = processor(img_input, return_tensors="pt")
            out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)

            # Log model parameters and results
            mlflow.log_param("model_name", "Salesforce/blip-image-captioning-base")
            mlflow.log_metric("output_length", len(caption))
            with open("output_caption.txt", "w") as f:
                f.write(caption)
            mlflow.log_artifact("output_caption.txt")

            return caption
        except Exception as e:
            mlflow.log_param("error", str(e))
            return f"Error occurred: {str(e)}"

# Set up Gradio interface
demo = gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(label="Upload Image"),
    outputs=gr.Text(label="Generated Caption"),
)

# Launch Gradio app and log the model with MLflow
if __name__ == "__main__":
    # Set the MLflow experiment
    mlflow.set_experiment("Image Captioning Experiment")

    # Log the model for reuse
    with mlflow.start_run():
        mlflow.pytorch.log_model(model, "blip_model")

    # Launch the Gradio interface
    demo.launch()