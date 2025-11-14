# import os
# import numpy as np
# from flask import Flask, render_template, request, jsonify
# import pickle
# from PIL import Image
# import warnings
# import re
# warnings.filterwarnings('ignore')

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'static/uploads'
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# model = None
# tokenizer = None
# feature_model = None
# max_length = None
# vocab_size = None

# def load_models():
#     global model, tokenizer, feature_model, max_length, vocab_size
    
#     try:
#         print("Loading tokenizer...")
        
#         with open('tokenizer.pkl', 'rb') as f:
#             tokenizer = pickle.load(f)
        
#         vocab_size = len(tokenizer.word_index) + 1
#         print(f"Vocabulary size: {vocab_size}")
        
#         print("Loading caption model...")
        
#         # Sử dụng keras thay vì tensorflow.keras nếu cần
#         try:
#             from tensorflow.keras.models import load_model
#             model = load_model('caption_model_final.keras', compile=False)
#         except:
#             from keras.models import load_model
#             model = load_model('caption_model_final.keras', compile=False)
        
#         try:
#             from tensorflow.keras.optimizers import Adam
#             model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001))
#         except:
#             from keras.optimizers import Adam
#             model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001))
        
#         max_length = model.inputs[1].shape[1]
#         print(f"Detected max_length: {max_length}")
        
#         print("Loading feature extraction model...")
        
#         try:
#             from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
#             from tensorflow.keras.models import Model
#         except:
#             from keras.applications.inception_v3 import InceptionV3, preprocess_input
#             from keras.models import Model
        
#         base = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
#         feature_model = Model(base.input, base.output)
        
#         print("All models loaded successfully!")
        
#     except Exception as e:
#         print(f"Error loading models: {e}")
#         import traceback
#         traceback.print_exc()

# def extract_features(image_path):
#     try:
#         try:
#             from tensorflow.keras.preprocessing.image import load_img, img_to_array
#             from tensorflow.keras.applications.inception_v3 import preprocess_input
#         except:
#             from keras.preprocessing.image import load_img, img_to_array
#             from keras.applications.inception_v3 import preprocess_input
        
#         target_size = (299, 299)
        
#         img = load_img(image_path, target_size=target_size)
#         x = img_to_array(img)
#         x = np.expand_dims(x, axis=0)
#         x = preprocess_input(x)
        
#         feat = feature_model.predict(x, verbose=0)
        
#         print(f"Extracted features shape: {feat.shape}")
#         return feat[0]
        
#     except Exception as e:
#         print(f"Error extracting features: {e}")
#         return None

# def generate_caption_clean(model, tokenizer, photo_feature, max_length):
#     try:
#         try:
#             from tensorflow.keras.preprocessing.sequence import pad_sequences
#         except:
#             from keras.preprocessing.sequence import pad_sequences
        
#         in_text = 'startseq'
#         for _ in range(max_length):
#             seq = tokenizer.texts_to_sequences([in_text])[0]
#             seq = pad_sequences([seq], maxlen=max_length)
            
#             yhat = model.predict([photo_feature.reshape((1, 2048)), seq], verbose=0)
#             yhat = np.argmax(yhat)
#             word = tokenizer.index_word.get(yhat)
            
#             if word is None:
#                 break
#             in_text += ' ' + word
#             if word == 'endseq':
#                 break
        
#         out = in_text.replace('startseq', '').replace('endseq', '').strip()
#         out = re.sub(r'\s+', ' ', out).strip()
#         return out
        
#     except Exception as e:
#         print(f"Error generating caption: {e}")
#         return "Cannot generate caption"

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         if 'image' not in request.files:
#             return jsonify({'error': 'No file selected'})
        
#         file = request.files['image']
#         if file.filename == '':
#             return jsonify({'error': 'No file selected'})
        
#         if model is None or tokenizer is None:
#             return jsonify({'error': 'Model not loaded'})
        
#         allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
#         if '.' not in file.filename or file.filename.split('.')[-1].lower() not in allowed_extensions:
#             return jsonify({'error': 'File format not supported'})
        
#         filename = f"temp_{os.urandom(8).hex()}.jpg"
#         file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(file_path)
        
#         image_features = extract_features(file_path)
#         if image_features is None:
#             return jsonify({'error': 'Cannot process image'})
        
#         caption = generate_caption_clean(model, tokenizer, image_features, max_length)
        
#         return jsonify({
#             'success': True,
#             'caption': caption,
#             'image_url': file_path
#         })
        
#     except Exception as e:
#         return jsonify({'error': f'Server error: {str(e)}'})

# if __name__ == '__main__':
#     if not os.path.exists(app.config['UPLOAD_FOLDER']):
#         os.makedirs(app.config['UPLOAD_FOLDER'])
    
#     print("Loading models...")
#     load_models()
    
#     if model and tokenizer and feature_model:
#         print("Flask app running on http://localhost:5000")
#         app.run(debug=True, host='0.0.0.0', port=5000)
#     else:
#         print("Cannot start app due to model loading errors")



import os
import numpy as np
from flask import Flask, render_template, request, jsonify
import pickle
from PIL import Image
import warnings
import re
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

model = None
tokenizer = None
feature_model = None
max_length = None
vocab_size = None

def load_models():
    global model, tokenizer, feature_model, max_length, vocab_size
    
    try:
        print("Loading tokenizer...")
        
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        
        vocab_size = len(tokenizer.word_index) + 1
        print(f"Vocabulary size: {vocab_size}")
        
        print("Loading caption model...")
        from tensorflow.keras.models import load_model
        
        model = load_model('caption_model_final.keras', compile=False)
        
        from tensorflow.keras.optimizers import Adam
        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001))
        
        max_length = model.inputs[1].shape[1]
        print(f"Detected max_length: {max_length}")
        
        print("Loading feature extraction model...")
        
        from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
        from tensorflow.keras.models import Model
        
        base = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
        feature_model = Model(base.input, base.output)
        
        print("All models loaded successfully!")
        
    except Exception as e:
        print(f"Error loading models: {e}")
        import traceback
        traceback.print_exc()

def extract_features(image_path):
    try:
        from tensorflow.keras.preprocessing.image import load_img, img_to_array
        from tensorflow.keras.applications.inception_v3 import preprocess_input
        
        target_size = (299, 299)
        
        img = load_img(image_path, target_size=target_size)
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        feat = feature_model.predict(x, verbose=0)
        
        print(f"Extracted features shape: {feat.shape}")
        return feat[0]
        
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def generate_caption_clean(model, tokenizer, photo_feature, max_length):
    try:
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        
        in_text = 'startseq'
        for _ in range(max_length):
            seq = tokenizer.texts_to_sequences([in_text])[0]
            seq = pad_sequences([seq], maxlen=max_length)
            
            yhat = model.predict([photo_feature.reshape((1, 2048)), seq], verbose=0)
            yhat = np.argmax(yhat)
            word = tokenizer.index_word.get(yhat)
            
            if word is None:
                break
            in_text += ' ' + word
            if word == 'endseq':
                break
        
        out = in_text.replace('startseq', '').replace('endseq', '').strip()
        out = re.sub(r'\s+', ' ', out).strip()
        return out
        
    except Exception as e:
        print(f"Error generating caption: {e}")
        return "Cannot generate caption"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No file selected'})
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        if model is None or tokenizer is None:
            return jsonify({'error': 'Model not loaded'})
        
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
        if '.' not in file.filename or file.filename.split('.')[-1].lower() not in allowed_extensions:
            return jsonify({'error': 'File format not supported'})
        
        filename = f"temp_{os.urandom(8).hex()}.jpg"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        image_features = extract_features(file_path)
        if image_features is None:
            return jsonify({'error': 'Cannot process image'})
        
        caption = generate_caption_clean(model, tokenizer, image_features, max_length)
        
        return jsonify({
            'success': True,
            'caption': caption,
            'image_url': file_path
        })
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'})

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    print("Loading models...")
    load_models()
    
    if model and tokenizer and feature_model:
        print("Flask app running on http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Cannot start app due to model loading errors")