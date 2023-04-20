
import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import whisper
import numpy as np
from ffmpeg import FFmpeg
import torch

app = Flask(__name__)
CORS(app)
app.config['JSON_AS_ASCII'] = False
app.config['JSONIFY_MIMETYPE'] = "application/json;charset=utf-8"

models = {}

# https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
# load_audio
def load_audio(buf):
    ffmpeg = (
        FFmpeg()
        .option("y")
        .input("pipe:0")
        .output(
            "pipe:1",
            {"codec:a": "pcm_s16le"},
            ac=1,
            ar=16000,
            f="s16le",
        )
    )
    out = ffmpeg.execute(buf)

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

@app.route("/asr", methods=['GET', 'POST'])
def ocr_route():
    if request.method == 'POST':
        buf = request.files['file'].read()
        audio = load_audio(buf)
        model_name = request.form.get('model') or 'base'
        model = models.get(model_name)
        if not model:
            model = whisper.load_model(model_name)
            models[model_name] = model

        data = model.transcribe(audio)
        result = {
            'model': model_name,
            'data': data,
        }

        return jsonify({'result': result})
    else:
        return render_template('upload.html')

@app.route("/device", methods=['GET'])
def device_route():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return jsonify({'result': device})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
