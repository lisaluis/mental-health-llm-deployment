import os
import gc
import time
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = os.getenv("BASE_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
SFT_PATH = os.getenv("SFT_ADAPTER_PATH", "./models/qwen_qlora_sft_100")
ORPO_PATH = os.getenv("ORPO_ADAPTER_PATH", "./models/qwen_orpo_adapter")

st.set_page_config(page_title="Mental Health LLM Lab", page_icon="🫶", layout="wide")
st.title("Mental Health LLM Lab")
st.caption("Research prototype | Qwen2.5-1.5B-Instruct | QLoRA SFT → ORPO")

with st.sidebar:
    st.header("Experiment parameters")
    model_choice = st.selectbox("Model configuration", ["Base", "SFT (QLoRA)", "SFT → ORPO"])
    quantization = st.selectbox("Inference quantization", ["Auto (recommended)", "4-bit NF4", "8-bit", "None"])
    max_new_tokens = st.slider("Maximum new tokens", 32, 512, 180, 16)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.05)
    top_k = st.slider("Top-k", 1, 100, 50, 1)
    do_sample = st.checkbox("Sampling enabled", value=True)
    st.divider()
    st.subheader("Training configuration (reference)")
    st.caption("These settings describe the saved experiment. Changing them here does not retrain the model.")
    st.code("SFT: NF4, r=8, alpha=16, dropout=0.05, epochs=2, LR=2e-4, max_seq_len=512\nORPO: 20 preference pairs, 1 epoch, LR=5e-6, beta=0.1")
    st.warning("Not a diagnostic or treatment tool. If someone may be in immediate danger, contact local emergency services or a trusted person who can help now.")

def get_quant_config(choice):
    if not torch.cuda.is_available():
        if choice in ("4-bit NF4", "8-bit"):
            raise RuntimeError("4-bit/8-bit bitsandbytes inference generally requires a supported CUDA GPU in this setup. Choose 'None' for CPU inference, though it may be slow and memory intensive.")
        return None
    if choice in ("Auto (recommended)", "4-bit NF4"):
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )
    if choice == "8-bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None

@st.cache_resource(show_spinner="Loading model. First load can take several minutes...")
def load_model(model_name, quant_choice):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    kwargs = {"device_map": "auto"}
    qconfig = get_quant_config(quant_choice)
    if qconfig is not None:
        kwargs["quantization_config"] = qconfig
    if torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.float16
    else:
        kwargs["torch_dtype"] = torch.float32
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **kwargs)
    if model_name == "SFT (QLoRA)":
        if not os.path.isdir(SFT_PATH):
            raise FileNotFoundError(f"SFT adapter directory not found: {SFT_PATH}")
        model = PeftModel.from_pretrained(model, SFT_PATH)
    elif model_name == "SFT → ORPO":
        if not os.path.isdir(ORPO_PATH):
            raise FileNotFoundError(f"ORPO adapter directory not found: {ORPO_PATH}")
        # This checkpoint was saved after ORPO training initialized from SFT.
        # Load the resulting adapter directly onto the original base model.
        model = PeftModel.from_pretrained(model, ORPO_PATH)
    model.eval()
    return tokenizer, model

def generate(model_name, quant_choice, prompt, max_tokens, sampling, temp, p, k, system_text):
    tokenizer, model = load_model(model_name, quant_choice)
    messages = []
    if system_text.strip():
        messages.append({"role": "system", "content": system_text.strip()})
    messages.append({"role": "user", "content": prompt.strip()})
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    kwargs = {
        "max_new_tokens": max_tokens,
        "do_sample": sampling,
        "pad_token_id": tokenizer.eos_token_id
    }
    if sampling:
        kwargs.update({"temperature": max(0.01, temp), "top_p": p, "top_k": k})
    start = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, **kwargs)
    elapsed = time.time() - start
    answer = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return answer, elapsed

tab_chat, tab_compare, tab_details = st.tabs(["Chat / Playground", "Compare models", "Experiment details"])

with tab_chat:
    col1, col2 = st.columns([2, 1])
    with col1:
        prompt = st.text_area("User prompt", height=140, placeholder="I've been anxious about exams and can't concentrate.")
        system_prompt = st.text_area("Optional system instruction", value="You are a supportive assistant. Be empathetic, practical, non-judgmental, and do not diagnose or replace professional care.", height=100)
    with col2:
        st.markdown("#### Active settings")
        st.write(f"**Configuration:** {model_choice}")
        st.write(f"**Quantization:** {quantization}")
        st.write(f"**Decoding:** {'Sampling' if do_sample else 'Greedy'}")
        st.write(f"**Max tokens:** {max_new_tokens}")
        st.write(f"**Temperature:** {temperature if do_sample else 'Ignored (greedy)'}")
        st.write(f"**Top-p / Top-k:** {f'{top_p} / {top_k}' if do_sample else 'Ignored (greedy)'}")
    if st.button("Generate response", type="primary"):
        if not prompt.strip():
            st.info("Enter a prompt first.")
        else:
            try:
                with st.spinner("Generating response..."):
                    answer, elapsed = generate(model_choice, quantization, prompt, max_new_tokens, do_sample, temperature, top_p, top_k, system_prompt)
                st.markdown("### Model response")
                st.write(answer)
                st.caption(f"Generation time: {elapsed:.1f} seconds | Configuration: {model_choice}")
                st.caption("Outputs may be inaccurate or incomplete. This prototype is not clinically validated.")
            except Exception as exc:
                st.error(f"Generation failed: {exc}")
                st.info("Check adapter paths, host memory, and whether the selected quantization is supported.")

with tab_compare:
    st.subheader("Same-prompt comparison")
    compare_prompt = st.text_area("Prompt for all configurations", key="compare_prompt", height=100)
    compare_quant = st.selectbox("Comparison quantization", ["Auto (recommended)", "4-bit NF4", "8-bit", "None"], index=0)
    compare_tokens = st.slider("Comparison max new tokens", 32, 256, 128, 16)
    if st.button("Run comparison"):
        if not compare_prompt.strip():
            st.info("Enter a prompt.")
        else:
            for name in ["Base", "SFT (QLoRA)", "SFT → ORPO"]:
                st.markdown(f"### {name}")
                try:
                    answer, elapsed = generate(name, compare_quant, compare_prompt, compare_tokens, False, 0.7, 0.9, 50, "You are supportive, practical, and non-diagnostic.")
                    st.write(answer)
                    st.caption(f"{elapsed:.1f} seconds")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as exc:
                    st.error(f"{name}: {exc}")

with tab_details:
    st.header("Model and experiment details")
    st.markdown("""
**Base model:** `Qwen/Qwen2.5-1.5B-Instruct`

**Configurations**
- **Base:** unmodified instruction model.
- **SFT (QLoRA):** base plus the saved QLoRA SFT adapter.
- **SFT → ORPO:** saved final checkpoint produced by ORPO training initialized from SFT. For inference, the saved checkpoint is loaded directly onto the base model.

**Recorded SFT settings**
- 100 provisional examples; 4-bit NF4 with double quantization; FP16 compute
- LoRA rank 8, alpha 16, dropout 0.05
- 2 epochs, batch size 1, gradient accumulation 4
- Learning rate 2e-4, max sequence length 512

**Recorded ORPO settings**
- 20 preference pairs, 1 epoch, learning rate 5e-6, beta 0.1
- Preference data is small and uses a repeated rejected response; results are preliminary.

**Important:** inference controls in the sidebar change generation settings only. They do not change the trained LoRA parameters or retrain the model. Quantization choices affect inference memory/precision.
""")
    st.info("For self-harm or immediate danger, encourage immediate human assistance and appropriate local emergency resources. Do not rely on this prototype for crisis response.")
