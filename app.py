from flask import Flask, request, render_template, jsonify, redirect, url_for, send_from_directory, abort, session
from werkzeug.utils import secure_filename, safe_join
import os
import pytesseract
from PIL import Image
import google.generativeai as genai
import pdf2image
import requests
import aiohttp
import asyncio
from functools import lru_cache
import logging
import torch
import numpy as np
import cv2
from diffusers import StableDiffusionPipeline

# Configuration
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'pdf'}
app.secret_key = 'f4977e0200ba57eb369e0cd8580dbbe93677aa197fbecaf0'  # Replace with your actual key
app.config['GEMINI_API_KEY'] = 'AIzaSyDFRqwNShMlcXZG9TDdhg8GUXxnGRuF0PQ'  # Replace with your actual Gemini AI API key

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Set up Google Gemini AI and Generative AI configurations
genai.configure(api_key=app.config['GEMINI_API_KEY'])
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

# Google Custom Search configuration
api_key = 'AIzaSyAdiEPhnQhWW6pdaMuMe6z3x6xkbOiGtWA'
search_engine_id = '855dfb1be4fc843cd'

# Logging configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


# Allowed file check
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# Clear uploaded/downloaded files
def clear_downloaded_images(folder='downloads'):
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    logging.info(f"Deleted file: {file_path}")
            except Exception as e:
                logging.error(f"Error deleting file {file_path}: {e}")


# Image search function (using cache)
@lru_cache(maxsize=10)
def search_high_res_images(prompt, num_images=5, start=1):
    search_url = (f"https://www.googleapis.com/customsearch/v1?q={prompt}&key={api_key}"
                  f"&cx={search_engine_id}&searchType=image&num={num_images}&start={start}&imgSize=large")

    try:
        response = requests.get(search_url)
        response.raise_for_status()
        search_results = response.json()
        return search_results.get("items", [])
    except requests.exceptions.RequestException as e:
        logging.error(f"Error during API request: {e}")
        return []


# Asynchronous image download function with retries and error handling
async def download_image(session, image_url, folder='downloads', retries=3):
    if not os.path.exists(folder):
        os.makedirs(folder)

    for attempt in range(retries):
        try:
            async with session.get(image_url, timeout=20) as response:
                response.raise_for_status()

                # Create the image path safely
                image_name = os.path.join(folder, image_url.split('/')[-1].split('?')[0]).replace("\\", "/")
                logging.info(f"Saving image to: {image_name}")
                
                # Write the image to the specified folder
                with open(image_name, 'wb') as f:
                    f.write(await response.read())

                logging.info(f"Image downloaded: {image_name}")
                return image_name
        except asyncio.TimeoutError:
            logging.warning(f"Timeout on attempt {attempt + 1} for {image_url}")
        except aiohttp.ClientError as e:
            logging.error(f"Error downloading image {image_url}: {e}")
        except PermissionError as e:
            logging.error(f"Permission error for {image_name}: {e}")
            return None  # or handle as needed
        except Exception as e:
            logging.error(f"An error occurred: {e}")
        await asyncio.sleep(1)  # Short wait before retrying


# Download multiple images asynchronously
async def download_images(image_urls):
    async with aiohttp.ClientSession() as session:
        tasks = [download_image(session, url) for url in image_urls]
        return await asyncio.gather(*tasks)


# Image Generation Configuration
class CFG:
    device = "cuda"  # Use "cpu" if CUDA is not available or if you face memory issues
    seed = 42
    generator = torch.Generator().manual_seed(seed)
    image_gen_steps = 20
    image_gen_model_id = "stabilityai/stable-diffusion-2-1"
    image_gen_size = (512, 512)
    image_gen_guidance_scale = 8

# Initialize the model
image_gen_model = StableDiffusionPipeline.from_pretrained(
    CFG.image_gen_model_id,
    torch_dtype=torch.float16,
    revision="fp16",
).to(CFG.device)

# Function to generate images
def generate_image(prompt):
    # Clear CUDA cache to free up memory
    torch.cuda.empty_cache()

    image = image_gen_model(
        prompt,
        num_inference_steps=CFG.image_gen_steps,
        generator=CFG.generator,
        guidance_scale=CFG.image_gen_guidance_scale
    ).images[0]
    
    # Resize image to desired output size
    image = image.resize(CFG.image_gen_size)
    
    # Convert to NumPy array for OpenCV processing
    image_np = np.array(image)
    
    # Sharpen and enhance image
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_image = cv2.filter2D(image_np, -1, sharpen_kernel)
    enhanced_image = cv2.convertScaleAbs(sharpened_image, alpha=1.1, beta=10)
    
    return enhanced_image


# Flask routes
@app.route('/')
def index():
    return render_template('index.html', message='', result='')


@app.route('/ai_notes')
def ai_notes():
    return render_template('ai_notes.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify(message="No file part", result='')

    file = request.files['file']
    if file.filename == '':
        return jsonify(message="No selected file", result='')

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        for f in os.listdir(app.config['UPLOAD_FOLDER']):
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], f))

        file.save(file_path)
        return jsonify(message="File uploaded successfully", result='')
    else:
        return jsonify(message="Unsupported file format", result='')


@app.route('/analyze', methods=['POST'])
def analyze_file():
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    if not files:
        return jsonify(message='', result="No files to analyze")

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], files[0])
    filename = files[0]

    try:
        extracted_text = ""
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = Image.open(file_path)
            extracted_text = pytesseract.image_to_string(image)
        elif file_path.lower().endswith('.pdf'):
            images = pdf2image.convert_from_path(file_path)
            for img in images:
                extracted_text += pytesseract.image_to_string(img)
        else:
            return jsonify(message='', result="Unsupported file format")

        return redirect(url_for('results', text=extracted_text, filename=filename))
    except Exception as e:
        app.logger.error(f'Error during analysis: {e}')
        return jsonify(message='Error during analysis', result=str(e))


@app.route('/results')
def results():
    text = request.args.get('text', '')
    filename = request.args.get('filename', '')
    return render_template('results.html', text=text, filename=filename)


@app.route('/gemini_analysis', methods=['POST'])
def gemini_analysis():
    data = request.get_json()
    text = data.get('text')

    try:
        chat_session = model.start_chat(history=[])
        summary_response = chat_session.send_message(f"Provide a brief summary,what the extracted content is about in one or two lines: {text}")
        summary = summary_response.text if summary_response else "Summary not available."

        highlights_response = chat_session.send_message(f"Highlight key points: {text}")
        highlights = highlights_response.text if highlights_response else "Highlights not available."

        return jsonify({'summary': summary, 'highlights': highlights})
    except Exception as e:
        app.logger.error(f'Failed to get summary and highlights: {e}')
        return jsonify(error='Failed to get summary and highlights', details=str(e)), 500


@app.route('/chat')
def new_page():
    return render_template('EDITH-AI.html')


@app.route('/gemini_programming_analysis', methods=['POST'])
def gemini_programming_analysis():
    data = request.get_json()
    prompt = data.get('prompt')

    try:
        chat_session = model.start_chat(history=[])
        programming_response = chat_session.send_message(f"Answer any question : {prompt}")
        response_text = programming_response.text if programming_response else "Response not available."
        
        return jsonify({'response': response_text})
    except Exception as e:
        app.logger.error(f'Failed to get programming response: {e}')
        return jsonify(error='Failed to get programming response', details=str(e)), 500


# New route for image generation
@app.route('/img_gen', methods=['GET', 'POST'])
def img_gen():
    generated_image_path = None  # Initialize variable for image path
    if request.method == 'POST':
        prompt = request.form['prompt']
        generated_image = generate_image(prompt)

        # Save the generated image temporarily to display it later
        img_pil = Image.fromarray(generated_image)
        img_pil.save('static/generated_image.png')

        generated_image_path = 'static/generated_image.png'  # Update with path to the image

        return render_template('imgen.html', prompt=prompt, generated_image_path=generated_image_path)

    return render_template('imgen.html', prompt=None, generated_image_path=None)

@app.route('/img_search', methods=['GET', 'POST'])
async def img_search():
    if request.method == 'POST':
        prompt = request.form['prompt']
        num_images = 10
        start = int(request.form.get('start', 1))

        if start == 1:
            clear_downloaded_images()
            session['previous_images'] = []

        results = search_high_res_images(prompt, num_images, start)
        image_urls = [result['link'] for result in results]
        previously_displayed = set(session.get('previous_images', []))
        new_image_urls = [url for url in image_urls if url not in previously_displayed]

        if not new_image_urls:
            start += num_images
            results = search_high_res_images(prompt, num_images, start)
            new_image_urls = [result['link'] for result in results]

        image_paths = await download_images(new_image_urls)
        session['previous_images'].extend(new_image_urls)

        return render_template('img_search.html', image_paths=image_paths, prompt=prompt, start=start + num_images)

    return render_template('img_search.html')


@app.route('/downloads/<path:filename>')
def download_file(filename):
    file_path = safe_join('downloads', filename)
    if not os.path.isfile(file_path):
        abort(404)
    return send_from_directory('downloads', filename)



@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500


if __name__ == "__main__":
    app.run(debug=True)
