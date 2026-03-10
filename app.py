import os
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# -----------------------------
# Settings
# -----------------------------
LEVEL_OPTIONS = ["easy", "medium", "hard"]
SETTING_OPTIONS = ["office", "school", "family dinner", "online", "bar"]
TONE_OPTIONS = ["sarcastic", "dry", "wholesome", "absurd"]

CKPT_DIR = "./checkpoints/try3/distilgpt2_lora_finetuned3"
BASE_MODEL_NAME = "distilgpt2"
_device = "cuda" if torch.cuda.is_available() else "cpu"

_tok = None
_model = None

# -----------------------------
# Load model
# -----------------------------
def load_model():
    global _tok, _model
    if _model is not None and _tok is not None:
        return

    if not os.path.isdir(CKPT_DIR):
        st.error(f"Checkpoint directory not found: {CKPT_DIR}")
        return

    _tok = AutoTokenizer.from_pretrained(CKPT_DIR)
    _tok.pad_token = _tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME).to(_device)
    _model = PeftModel.from_pretrained(base, CKPT_DIR).to(_device)
    _model.eval()

# -----------------------------
# Instruction builder
# -----------------------------
def make_instruction(level, setting, tone):
    return f"Write a joke.\nLEVEL: {level}\nSETTING: {setting}\nTONE: {tone}"

def build_prompt(level, setting, tone, extra):
    inst = make_instruction(level, setting, tone)
    extra = extra.strip()
    if extra:
        inst += "\n\nExtra details:\n" + extra
    return inst + "\nJoke:\n"

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Reddit Joke Generator", layout="wide")

st.markdown("""
# 🎭 Reddit Jokes Generator
Team: BOUCHENTOUF Omar, MOHAMED JOURFI, ABOUBEKRIN MOHAMED SALEM, Stephie JEAN JOSEPH
""")

# Sidebar controls
st.sidebar.header("Controls")
level = st.sidebar.radio("Level", LEVEL_OPTIONS)
setting = st.sidebar.selectbox("Setting", SETTING_OPTIONS)
tone = st.sidebar.radio("Tone", TONE_OPTIONS)
extra = st.sidebar.text_area("Extra guidance", placeholder="Optional: topic, characters, constraints...")

st.sidebar.header("Sampling")
max_tokens = st.sidebar.slider("Max tokens", 20, 200, 80, step=10)
temperature = st.sidebar.slider("Temperature", 0.1, 1.5, 0.8, step=0.05)
top_p = st.sidebar.slider("Top-p", 0.5, 1.0, 0.95, step=0.01)

# Generate button
if st.button("Generate joke"):
    load_model()
    if _model is not None:
        prompt = build_prompt(level, setting, tone, extra)
        inputs = _tok(prompt, return_tensors="pt").to(_device)
        gen_ids = _model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            pad_token_id=_tok.eos_token_id,
        )
        full_text = _tok.decode(gen_ids[0], skip_special_tokens=True)
        joke = full_text[len(prompt):].strip()

        st.markdown("### Instruction")
        st.text(prompt)
        st.markdown("### Generated Joke")
        st.success(joke)