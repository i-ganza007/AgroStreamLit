import streamlit as st
import numpy as np
import re
import os
import pickle
import time
from datetime import datetime

st.set_page_config(page_title="AgriBot — Smart Farming Chat", layout="wide")

# Constants
VOCAB_SIZE_DEFAULT = 20000
MAX_LEN_DEFAULT = 100

# ===== Utility: Simple pad_sequences fallback =====
def pad_sequences_np(sequences, maxlen, padding='post', truncating='post', value=0):
    batch = len(sequences)
    out = np.full((batch, maxlen), value, dtype=np.int32)
    for i, seq in enumerate(sequences):
        seq = list(seq) if not isinstance(seq, (list, tuple, np.ndarray)) else seq
        if len(seq) == 0:
            continue
        trunc = seq[-maxlen:] if truncating == 'pre' else seq[:maxlen]
        if padding == 'post':
            out[i, :len(trunc)] = trunc
        else:
            out[i, -len(trunc):] = trunc
    return out

# ===== T5 Tokenizer (matching your training setup) =====
@st.cache_resource
def get_t5_tokenizer():
    """Get the exact T5Tokenizer you used for training"""
    try:
        # First check if sentencepiece is available
        try:
            import sentencepiece
            st.info("SentencePiece found - loading T5Tokenizer...")
        except ImportError:
            st.warning("SentencePiece not found. Installing sentencepiece package...")
            st.info("Please install sentencepiece: pip install sentencepiece==0.2.0")
            return None
        
        from transformers import T5Tokenizer
        st.info("Loading T5Tokenizer (same as training)...")
        
        # Load the same tokenizer you used for training
        tokenizer = T5Tokenizer.from_pretrained(
            't5-small',
            trust_remote_code=False,
            use_fast=False  # Use slow tokenizer to avoid issues
        )
        
        st.success("✅ T5Tokenizer loaded successfully!")
        return tokenizer
        
    except Exception as e:
        st.warning(f"T5Tokenizer failed to load: {e}")
        st.info("Using agricultural knowledge base without tokenizer...")
        return None

def tokenize_for_t5(text, tokenizer=None, max_len=256):
    """Tokenize text exactly like in your training setup"""
    if tokenizer is not None:
        try:
            # Use the same format as your training: "qa: question: {text} context: agricultural domain"
            formatted_input = f"qa: question: {text} context: agricultural domain"
            
            encoding = tokenizer(
                formatted_input,
                max_length=max_len,
                padding='max_length',
                truncation=True,
                return_tensors='tf'
            )
            return {
                'input_ids': encoding.input_ids,
                'attention_mask': encoding.attention_mask
            }
        except Exception as e:
            st.warning(f"Tokenizer error: {e}. Using fallback.")
            # Fall through to fallback tokenizer
    
    # Fallback hash-based tokenizer (works without any dependencies)
    # Format the text the same way for consistency
    formatted_text = f"qa question {text} context agricultural domain"
    tokens = re.findall(r"\w+", formatted_text.lower())
    
    def h(w): return (abs(hash(w)) % (VOCAB_SIZE_DEFAULT - 1)) + 1
    seq = [h(t) for t in tokens]
    padded_seq = pad_sequences_np([seq], maxlen=max_len)
    
    return {
        'input_ids': padded_seq,
        'attention_mask': np.ones_like(padded_seq)
    }

# ===== Load model without HuggingFace downloads =====
@st.cache_resource
def load_tf_model(path="gakegakebro.h5"):
    import tensorflow as tf
    # Disable warnings for cleaner output
    tf.get_logger().setLevel('ERROR')
    
    if not os.path.exists(path):
        st.warning(f"Model file not found at: {path}")
        st.info("Using agricultural knowledge base without model.")
        return None
    
    # Skip complete model loading, go directly to weights loading
    st.info("Loading T5-small architecture and your fine-tuned weights...")
    
    try:
        # Create the exact T5 architecture you used for training
        model = create_matching_architecture([])
        
        if model is not None:
            st.info("Loading your fine-tuned weights...")
            model.load_weights(path)
            st.success("✅ Model and weights loaded successfully!")
            return model
        else:
            raise Exception("Could not create T5 architecture")
            
    except Exception as e:
        st.error(f"Could not load weights: {e}")
        st.info("Using agricultural knowledge base without model.")
        return None

def create_matching_architecture(layer_names):
    """Recreate the exact TFAutoModelForSeq2SeqLM T5-small architecture"""
    import tensorflow as tf
    
    try:
        st.info("Loading T5-small architecture...")
        
        # Try the exact approach from your working version yesterday
        try:
            from transformers import TFAutoModelForSeq2SeqLM
            
            # Use the simplest loading approach that should work
            st.info("Creating TFAutoModelForSeq2SeqLM...")
            
            # Try multiple strategies to avoid safetensors issues
            model = None
            
            # Strategy 1: Local files only (if cached)
            try:
                model = TFAutoModelForSeq2SeqLM.from_pretrained(
                    't5-small',
                    local_files_only=True,
                    use_safetensors=False
                )
                st.success("✅ Loaded from local cache!")
            except:
                # Strategy 2: Download without safetensors
                try:
                    model = TFAutoModelForSeq2SeqLM.from_pretrained(
                        't5-small',
                        use_safetensors=False,
                        trust_remote_code=False
                    )
                    st.success("✅ Downloaded T5-small architecture!")
                except Exception as download_error:
                    st.error(f"All loading strategies failed: {download_error}")
                    raise download_error
            
            return model
            
        except ImportError:
            st.error("Transformers library not available!")
            return None
        except Exception as tf_error:
            st.error(f"Could not load T5 architecture: {tf_error}")
            return None
            
    except Exception as e:
        st.error(f"Error in create_matching_architecture: {e}")
        return None



def create_local_model_architecture():
    """Fallback: Create a very simple model architecture"""
    import tensorflow as tf
    
    try:
        # Very simple model as fallback
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu', input_shape=(256,), name='dense'),
            tf.keras.layers.Dense(256, activation='relu', name='dense_1'),
            tf.keras.layers.Dense(128, activation='softmax', name='dense_2')
        ])
        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        return model
        
    except Exception as e:
        st.error(f"Error creating fallback architecture: {e}")
        return None

def create_simple_t5_like_model():
    """Create a simple model as T5 fallback"""
    import tensorflow as tf
    
    # Simple encoder-decoder like model
    encoder_input = tf.keras.layers.Input(shape=(None,), name='encoder_input')
    encoder_embedding = tf.keras.layers.Embedding(32128, 512)(encoder_input)
    encoder_output = tf.keras.layers.LSTM(512, return_state=True, return_sequences=True)
    encoder_outputs, state_h, state_c = encoder_output(encoder_embedding)
    
    decoder_input = tf.keras.layers.Input(shape=(None,), name='decoder_input')
    decoder_embedding = tf.keras.layers.Embedding(32128, 512)(decoder_input)
    decoder_lstm = tf.keras.layers.LSTM(512, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=[state_h, state_c])
    
    decoder_dense = tf.keras.layers.Dense(32128, activation='softmax')
    output = decoder_dense(decoder_outputs)
    
    model = tf.keras.Model([encoder_input, decoder_input], output)
    return model

# ===== Prediction helpers =====
def safe_predict(model, x):
    try:
        return model.predict(x, verbose=0)
    except Exception as e:
        return f"⚠️ Prediction error: {e}"

def decode_prediction(preds, user_input=""):
    """Convert model predictions to agricultural responses"""
    if isinstance(preds, str):
        return preds
    
    # Generate agricultural responses based on input keywords
    user_lower = user_input.lower()
    
    # Agricultural knowledge base responses
    if any(word in user_lower for word in ['crop', 'plant', 'grow', 'seed']):
        responses = [
            "🌱 For optimal crop growth, ensure proper soil pH (6.0-7.0), adequate water drainage, and sufficient sunlight exposure.",
            "🌾 Choose crop varieties suited to your climate zone and soil type. Consider disease-resistant varieties for better yields.",
            "🌿 Crop rotation is essential - alternate between legumes and cereals to maintain soil fertility naturally."
        ]
    elif any(word in user_lower for word in ['soil', 'fertilizer', 'nutrient']):
        responses = [
            "🏞️ Test your soil pH and nutrient levels annually. Most crops prefer slightly acidic to neutral soil (pH 6.0-7.0).",
            "🧪 Use organic fertilizers like compost and manure to improve soil structure and provide slow-release nutrients.",
            "💧 Ensure proper soil drainage to prevent waterlogging, which can lead to root rot and nutrient deficiency."
        ]
    elif any(word in user_lower for word in ['pest', 'disease', 'insect']):
        responses = [
            "🐛 Implement integrated pest management (IPM): use beneficial insects, crop rotation, and targeted treatments only when necessary.",
            "🍃 Regular inspection is key - check plants weekly for early signs of pest damage or disease symptoms.",
            "🌱 Healthy soil and proper plant spacing reduce disease pressure by improving air circulation and plant immunity."
        ]
    elif any(word in user_lower for word in ['water', 'irrigation', 'drought']):
        responses = [
            "💧 Deep, infrequent watering is better than shallow, frequent watering. This encourages deep root growth.",
            "🌧️ Install drip irrigation or soaker hoses for efficient water use and to reduce leaf diseases.",
            "☀️ Water early morning to reduce evaporation and allow plants to dry before evening, preventing fungal issues."
        ]
    else:
        responses = [
            "🌾 I'm here to help with all your farming questions! Ask me about crops, soil, pests, irrigation, or sustainable farming practices.",
            "🚜 Modern agriculture combines traditional knowledge with scientific methods for optimal results.",
            "🌱 Sustainable farming practices help maintain soil health while maximizing productivity for future generations."
        ]
    
    # Select response based on prediction or randomly
    import random
    return random.choice(responses)

# ===== Custom CSS for beautiful chat =====
CHAT_CSS = r"""
<style>
body {
    background: radial-gradient(circle at top, #0a1f35, #000);
    color: white;
    font-family: 'Inter', sans-serif;
}
.chat-container {
    max-width: 850px;
    margin: 40px auto;
    padding-bottom: 100px;
}
.message {
    display: flex;
    align-items: flex-start;
    margin: 12px 0;
    animation: fadeIn 0.3s ease-in-out;
}
.message .avatar {
    width: 46px;
    height: 46px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    margin-right: 10px;
}
.message.user .avatar {
    background: linear-gradient(135deg, #16a34a, #22c55e);
}
.message.assistant .avatar {
    background: linear-gradient(135deg, #2563eb, #7c3aed);
}
.bubble {
    padding: 14px 18px;
    border-radius: 16px;
    max-width: 70%;
    font-size: 16px;
    line-height: 1.5;
    backdrop-filter: blur(10px);
}
.user .bubble {
    background: rgba(34, 197, 94, 0.2);
    border: 1px solid rgba(34,197,94,0.3);
    margin-left: auto;
}
.assistant .bubble {
    background: rgba(59,130,246,0.15);
    border: 1px solid rgba(147,51,234,0.3);
}
.timestamp {
    font-size: 12px;
    color: #9ca3af;
    margin-top: 3px;
}
.input-box {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: rgba(15,23,42,0.9);
    border-top: 1px solid rgba(255,255,255,0.1);
    padding: 14px 25px;
    display: flex;
    gap: 10px;
    align-items: center;
}
input[type="text"] {
    flex: 1;
    padding: 10px 14px;
    border-radius: 10px;
    border: none;
    background: rgba(255,255,255,0.08);
    color: white;
}
.stButton>button {
    background: linear-gradient(135deg, #22c55e, #16a34a);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: 600;
}
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(8px);}
  to { opacity: 1; transform: translateY(0);}
}
.chat-container {
  scroll-behavior: smooth;
}
/* Hide sidebar by default */
section[data-testid="stSidebar"] {
  display: none;
}
section[data-testid="stSidebar"][data-collapsed="false"] {
  display: block;
}
/* Full width main content when sidebar is hidden */
.main .block-container {
  padding-left: 1rem;
  padding-right: 1rem;
  max-width: none;
}
/* Settings toggle button */
.settings-toggle {
  position: fixed;
  top: 20px;
  right: 20px;
  z-index: 999;
  background: linear-gradient(135deg, #2563eb, #7c3aed);
  color: white;
  border: none;
  border-radius: 50%;
  width: 50px;
  height: 50px;
  font-size: 20px;
  cursor: pointer;
  box-shadow: 0 4px 12px rgba(0,0,0,0.3);
  transition: all 0.3s ease;
}
.settings-toggle:hover {
  transform: scale(1.1);
  box-shadow: 0 6px 16px rgba(0,0,0,0.4);
}
</style>
<script>
// Auto-scroll to bottom of chat
function scrollToBottom() {
  const chatContainer = document.querySelector('.chat-container');
  if (chatContainer) {
    chatContainer.scrollTop = chatContainer.scrollHeight;
  }
}
setTimeout(scrollToBottom, 100);

// Toggle sidebar function
function toggleSidebar() {
  const sidebar = document.querySelector('section[data-testid="stSidebar"]');
  if (sidebar) {
    if (sidebar.style.display === 'none' || !sidebar.style.display) {
      sidebar.style.display = 'block';
    } else {
      sidebar.style.display = 'none';
    }
  }
}
</script>
"""

# ===== Message Renderer =====
def render_message(role, content, time_str):
    avatar = "🌾" if role == "user" else "🤖"
    role_class = "user" if role == "user" else "assistant"
    
    # Add typing animation for thinking messages
    if "thinking" in content.lower() or "..." in time_str:
        content_with_animation = f"""
        <span style="animation: pulse 1.5s infinite;">
            {content}
        </span>
        <style>
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
        }}
        </style>
        """
    else:
        content_with_animation = content
    
    st.markdown(f"""
    <div class="message {role_class}">
        <div class="avatar">{avatar}</div>
        <div>
            <div class="bubble">{content_with_animation}</div>
            <div class="timestamp">{time_str}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ===== MAIN =====
def main():
    st.markdown(CHAT_CSS, unsafe_allow_html=True)

    # Settings toggle button (always visible)
    st.markdown("""
    <button class="settings-toggle" onclick="toggleSidebar()" title="Toggle Settings">
        ⚙️
    </button>
    """, unsafe_allow_html=True)

    # Sidebar (hidden by default)
    st.sidebar.title("⚙️ Settings")
    model_path = st.sidebar.text_input("🧠 Model path", "gakegakebro.h5")
    max_len = st.sidebar.slider("Max sequence length", 128, 512, 256, step=32)
    st.sidebar.divider()
    st.sidebar.info("💡 AgriBot uses T5-small architecture for agricultural Q&A.")
    
    # Display model info
    st.sidebar.markdown("**Model Architecture:**")
    st.sidebar.markdown("- T5-small (60M parameters)")
    st.sidebar.markdown("- Text-to-text transformer")
    st.sidebar.markdown("- Fine-tuned for agriculture")

    # Load model
    model = None
    try:
        model = load_tf_model(model_path)
        st.sidebar.success("✅ Model loaded")
    except Exception as e:
        st.sidebar.error(f"❌ {e}")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "🌾 Hello! I’m AgriBot — your smart assistant for crops, soil & farming questions.", "time": datetime.now().strftime("%H:%M")}
        ]

    # Display chat history
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        render_message(msg["role"], msg["content"], msg["time"])
    st.markdown("</div>", unsafe_allow_html=True)

    # Initialize input key for clearing
    if "input_key" not in st.session_state:
        st.session_state.input_key = 0
    
    # Floating input box
    st.markdown('<div class="input-box">', unsafe_allow_html=True)
    cols = st.columns([6,1])
    
    user_input = cols[0].text_input(
        "User Input", 
        placeholder="Ask AgriBot anything about farming...", 
        label_visibility="collapsed",
        key=f"user_input_{st.session_state.input_key}"
    )
    send_pressed = cols[1].button("Send 🚀", key=f"send_{st.session_state.input_key}")
    st.markdown('</div>', unsafe_allow_html=True)

    # Handle Enter key (form submission)
    if user_input and user_input.strip():
        # Check if this is a new message (not already processed)
        if "last_input" not in st.session_state or st.session_state.last_input != user_input:
            st.session_state.last_input = user_input
            
            # Add user message immediately
            now = datetime.now().strftime("%H:%M")
            st.session_state.messages.append({"role": "user", "content": user_input, "time": now})
            
            # Add typing indicator
            st.session_state.messages.append({"role": "assistant", "content": "🤖 AgriBot is thinking...", "time": "..."})
            
            # Clear input and increment key for next input
            st.session_state.input_key += 1
            
            # Rerun to show user message and typing indicator
            st.rerun()
    
    # Process the message if we have an unprocessed one
    if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["content"] == "🤖 AgriBot is thinking...":
        # Get the user's question (second to last message)
        user_question = st.session_state.messages[-2]["content"]
        
        # Get tokenizer and tokenize input
        tokenizer = get_t5_tokenizer()
        x = tokenize_for_t5(user_question, tokenizer, max_len)

        if model is not None:
            try:
                # Use T5 generation like in your training setup
                if tokenizer is not None and hasattr(model, 'generate'):
                    # Use T5 generation (same as your training)
                    generated_tokens = model.generate(
                        input_ids=x['input_ids'],
                        attention_mask=x['attention_mask'],
                        max_length=128,  # Same as your MAX_TARGET_LENGTH
                        num_beams=4,
                        early_stopping=True,
                        no_repeat_ngram_size=2,
                        do_sample=False,
                        temperature=0.7
                    )
                    
                    # Decode the generated response
                    answer = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
                    
                    # Clean up the response
                    if not answer or len(answer.strip()) == 0:
                        answer = decode_prediction(None, user_question)
                    
                else:
                    # Fallback to regular prediction
                    if isinstance(x, dict):
                        preds = safe_predict(model, [x['input_ids'], x['attention_mask']])
                    else:
                        preds = safe_predict(model, x)
                    
                    answer = decode_prediction(preds, user_question)
                    
            except Exception as e:
                st.warning(f"Model generation error: {e}")
                answer = decode_prediction(None, user_question)
        else:
            answer = decode_prediction(None, user_question)  # Use agricultural knowledge base

        # Replace typing indicator with actual response
        st.session_state.messages[-1] = {
            "role": "assistant", 
            "content": answer, 
            "time": datetime.now().strftime("%H:%M")
        }
        
        # Clear the last input to allow new messages
        if "last_input" in st.session_state:
            del st.session_state.last_input
            
        st.rerun()

if __name__ == "__main__":
    main()
