Qwen Astrologer Fine-tune

Fine-tuning Qwen2.5-7B-Instruct on astrologer↔user chat conversations to build a model that gives kundli-based astrological guidance — with empathy, proper kundli analysis flow, and grounded (non-fatalistic) predictions.

Overview

The base Qwen2.5-7B-Instruct model is a general-purpose chat model — it can talk about anything reasonably well, but has no specific grounding in Vedic astrology conventions, kundli-reading style, or the specific tone (empathetic, asks for birth details, gives a "let me analyse your kundli" pause before answering) that this use case needs.

This project fine-tunes it on a set of manually curated astrologer-user conversations so the model picks up:


Kundli/Vedic astrology terminology and reasoning (bhav, dasha, grahon ki position, etc.)
The "please wait while I analyse your kundli" interaction pattern
Empathetic, non-fatalistic responses (no guaranteed predictions, no death/illness predictions)
Predicting reasonable future timeframes instead of exact dates


Approach: QLoRA (Quantized LoRA)

Full fine-tuning of a 7B parameter model needs 80GB+ GPU memory, which isn't practical on a single consumer/free-tier GPU. Instead this uses QLoRA:


4-bit quantization — the base model is loaded in 4-bit (NF4) precision, cutting memory usage roughly 4x compared to fp16
LoRA (Low-Rank Adaptation) — instead of updating all 7B parameters, small trainable "adapter" matrices are added to the attention and MLP layers. Only these adapters (roughly 0.2-0.5% of total parameters) are trained; the base model stays frozen
After training, the LoRA adapter is merged back into the base model weights, producing a single standalone model that can be served normally (e.g. via vLLM)


This makes it possible to fine-tune on a single T4 GPU (16GB VRAM, available free on Google Colab).

Data


55 manually written and curated astrologer-user conversations
Mix of Hindi, Hinglish, and English
Each conversation includes a system prompt defining astrologer persona/guardrails (e.g. never predicting death/illness, not analysing a third party's chart without consent, handling emotional distress with care before jumping to astrology)
Data format: JSONL, one conversation per line:


json  {"messages": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."}
  ]}

See data/astrology_chats_fixed.jsonl.

Training process


Load tokenizer + apply chat template — each conversation is converted into Qwen's native ChatML format using tokenizer.apply_chat_template(), so the training format matches exactly what the model expects at inference time.
Load base model in 4-bit using BitsAndBytesConfig (NF4 quantization, bfloat16 compute dtype).
Attach LoRA adapters (peft.LoraConfig) to the attention (q/k/v/o_proj) and MLP (gate/up/down_proj) layers. Rank r=16, alpha 32.
Train with SFTTrainer (from trl) for a few epochs over the dataset, with gradient accumulation to keep the effective batch size reasonable while fitting in limited VRAM.
Merge LoRA weights into the base model (merge_and_unload()) and save the result as a standalone HF model directory — this merged model is what gets deployed/served.


Full script: finetune_qwen.py

Setup

bashpip install -r requirements.txt

Recommended: run on Google Colab (free T4 GPU) or any single-GPU machine with 16GB+ VRAM.

Running

bashpython finetune_qwen.py

Adjust these constants at the top of the script as needed:


BASE_MODEL — base model to fine-tune (Qwen2.5-7B-Instruct by default)
DATA_PATH — path to the training JSONL file
NUM_EPOCHS, LR, BATCH_SIZE, GRAD_ACCUM — training hyperparameters
On memory-constrained GPUs (e.g. free-tier T4), keep BATCH_SIZE=1 and enable gradient checkpointing.


Output: a merged, deployable model saved to qwen-astrologer-lora-merged/ (not included in this repo due to size — see below).

Model weights

The final merged model (~15GB) is not stored in this repo due to GitHub file size limits.
