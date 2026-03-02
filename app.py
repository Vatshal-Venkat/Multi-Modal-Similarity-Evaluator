

import gradio as gr
import numpy as np
import torch
import cv2
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean, cityblock

from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torchaudio
import open_clip
import matplotlib.pyplot as plt
import io

# ---------------- Model Setup ----------------
text_model_cache = {}
cross_encoder_cache = {}

# Enhanced Sentence Transformer Embeddings + Optional Cross-Encoder

def get_text_embedding(text, model_name):
    if model_name not in text_model_cache:
        text_model_cache[model_name] = SentenceTransformer(model_name)
    model = text_model_cache[model_name]

    encoded_input = model.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        model_output = model._first_module().auto_model(**encoded_input)

    attention_mask = encoded_input['attention_mask']
    token_embeddings = model_output.last_hidden_state

    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = input_mask_expanded.sum(dim=1)
    mean_pooled = sum_embeddings / sum_mask

    embedding = mean_pooled.squeeze().numpy()
    return embedding / np.linalg.norm(embedding + 1e-10)

def get_cross_encoder_score(text1, text2, model_name="cross-encoder/stsb-roberta-base"):
    if model_name not in cross_encoder_cache:
        cross_encoder_cache[model_name] = CrossEncoder(model_name)
    model = cross_encoder_cache[model_name]
    score = model.predict([(text1, text2)])
    return float(score[0])

# OpenCLIP for image and video
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
clip_model.eval()

def get_image_embedding(image_path, model_name):
    image = Image.open(image_path).convert("RGB")
    image_tensor = clip_preprocess(image).unsqueeze(0)
    with torch.no_grad():
        emb = clip_model.encode_image(image_tensor).squeeze().numpy()
    return emb / np.linalg.norm(emb + 1e-10)

# Audio with Wav2Vec2
audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
audio_model.eval()

def get_audio_embedding(audio_path, model_name):
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)
    inputs = audio_processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = audio_model(**inputs)
    emb = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return emb / np.linalg.norm(emb + 1e-10)

# Video embeddings using sampled frames + OpenCLIP
def get_video_embedding(video_path, model_name, num_frames=5):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    selected_frames = np.linspace(0, frame_count - 1, num_frames).astype(int)

    embeddings = []
    for frame_num in selected_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            continue
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image_tensor = clip_preprocess(image).unsqueeze(0)
        with torch.no_grad():
            emb = clip_model.encode_image(image_tensor).squeeze().numpy()
        embeddings.append(emb)

    cap.release()
    if embeddings:
        emb = np.mean(embeddings, axis=0)
        return emb / np.linalg.norm(emb + 1e-10)
    else:
        return np.zeros(512)

# ---------------- Similarity Evaluation Logic ----------------
def evaluate_similarities(text1, text2, img1, img2, audio1, audio2, video1, video2,
                          text_models, img_models, audio_models, video_models):

    results = {}

    # TEXT
    if text1 and text2 and text_models:
        text_results = {}
        for model in text_models:
            emb1 = get_text_embedding(text1, model)
            emb2 = get_text_embedding(text2, model)
            sim = cosine_similarity([emb1], [emb2])[0][0]
            text_results[model] = round(float(sim), 3)

        # Add cross-encoder score
        cross_score = get_cross_encoder_score(text1, text2)
        text_results["cross-encoder/stsb-roberta-base"] = round(cross_score, 3)
        results["Text"] = text_results

    # IMAGE
    if img1 and img2 and img_models:
        img_results = {}
        for model in img_models:
            emb1 = get_image_embedding(img1, model)
            emb2 = get_image_embedding(img2, model)
            cos_sim = cosine_similarity([emb1], [emb2])[0][0]
            euc_dist = euclidean(emb1, emb2)
            man_dist = cityblock(emb1, emb2)
            img_results[model] = {
                "Cosine": round(float(cos_sim), 3),
                "Euclidean": round(float(euc_dist), 3),
                "Manhattan": round(float(man_dist), 3)
            }
        results["Image"] = img_results

    # AUDIO
    if audio1 and audio2 and audio_models:
        audio_results = {}
        for model in audio_models:
            emb1 = get_audio_embedding(audio1, model)
            emb2 = get_audio_embedding(audio2, model)
            sim = cosine_similarity([emb1], [emb2])[0][0]
            audio_results[model] = round(float(sim), 3)
        results["Audio"] = audio_results

    # VIDEO
    if video1 and video2 and video_models:
        video_results = {}
        for model in video_models:
            emb1 = get_video_embedding(video1, model)
            emb2 = get_video_embedding(video2, model)
            sim = cosine_similarity([emb1], [emb2])[0][0]
            video_results[model] = round(float(sim), 3)
        results["Video"] = video_results

    return results

def render_similarity_graphs(similarity_data):
    figs = []

    for modality, model_data in similarity_data.items():
        labels = []
        cosine_vals = []
        euclidean_vals = []
        manhattan_vals = []

        for model, scores in model_data.items():
            labels.append(model)

            if isinstance(scores, float):
                cosine_vals.append(scores)
            else:
                cosine_vals.append(scores.get("Cosine", 0))
                euclidean_vals.append(scores.get("Euclidean", 0))
                manhattan_vals.append(scores.get("Manhattan", 0))

        fig, ax = plt.subplots(figsize=(6, 3.5))
        x = np.arange(len(labels))

        ax.bar(x - 0.2, cosine_vals, width=0.2, label="Cosine", color="Red")

        if euclidean_vals:
            ax.bar(x, euclidean_vals, width=0.2, label="Euclidean", color="lightgreen")
        if manhattan_vals:
            ax.bar(x + 0.2, manhattan_vals, width=0.2, label="Manhattan", color="salmon")

        ax.set_title(f"{modality} Similarities")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15)
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png")
        buf.seek(0)
        figs.append(buf)
        plt.close(fig)

    return figs

# ---------------- Gradio UI ----------------
with gr.Blocks() as demo:
    gr.Markdown("## 🔍 Cross-Modal Similarity Evaluator")

    with gr.Tab("Text"):
        text1 = gr.Textbox(label="Text A")
        text2 = gr.Textbox(label="Text B")
        text_models = gr.CheckboxGroup(
            choices=["all-MiniLM-L6-v2", "paraphrase-MiniLM-L3-v2", "distiluse-base-multilingual-cased-v1"],
            label="Select Text Models"
        )

    with gr.Tab("Image"):
        img1 = gr.Image(type="filepath", label="Image A")
        img2 = gr.Image(type="filepath", label="Image B")
        img_models = gr.CheckboxGroup(choices=["CLIP"], label="Select Image Models")

    with gr.Tab("Audio"):
        audio1 = gr.Audio(type="filepath", label="Audio A")
        audio2 = gr.Audio(type="filepath", label="Audio B")
        audio_models = gr.CheckboxGroup(choices=["WAV2VEC"], label="Select Audio Models")

    with gr.Tab("Video"):
        video1 = gr.Video(label="Video A")
        video2 = gr.Video(label="Video B")
        video_models = gr.CheckboxGroup(choices=["CLIP"], label="Select Video Models")

    with gr.Row():
        submit = gr.Button("🔍 Compute Similarity")
        clear = gr.Button("🔄 Clear All")

    output_json = gr.JSON(label="Similarity Results")
    output_images = gr.Gallery(label="Similarity Graphs", columns=2, rows=2)

    def evaluate_and_plot(*args):
        sim_data = evaluate_similarities(*args)
        buffers = render_similarity_graphs(sim_data)
        images = [Image.open(buf) for buf in buffers]
        return sim_data, images

    submit.click(fn=evaluate_and_plot,
             inputs=[text1, text2, img1, img2, audio1, audio2, video1, video2,
                     text_models, img_models, audio_models, video_models],
             outputs=[output_json, output_images])

    clear.click(fn=lambda: ("", "", None, None, None, None, None, None, [], [], [], [], None, None),
    inputs=[],
    outputs=[text1, text2, img1, img2, audio1, audio2, video1, video2,
             text_models, img_models, audio_models, video_models, output_json, output_images])


if __name__ == "__main__":
    demo.queue()
    demo.launch()
