# LLM From Scratch

A compact GPT-style language model implemented in PyTorch, based on the code and concepts from [Build a Large Language Model (From Scratch)](https://github.com/rasbt/LLMs-from-scratch). The model is trained from random initialization on the bundled text, `The_Verdict.txt`; it does not use pretrained GPT-2 weights. GPT-2's tokenizer is used to encode and decode text.

## Requirements

Python and the packages listed in `requirements.txt` are needed. Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Train

From the repository directory, run:

```bash
python gpt_train.py
```

Training uses 90% of `The_Verdict.txt` for training and 10% for validation. The script selects CUDA when available and otherwise runs on CPU. It trains a 12-layer, 768-dimensional GPT-style model for 10 epochs with a context length of 256 tokens. Training can be slow, especially on CPU.

At the end, the script saves the weights to `model.pth` and the loss plot to `loss.pdf`. Running training again replaces those output files.

## Generate Text

To generate from the bundled checkpoint:

```bash
python gpt_generate.py
```

The generation script loads `model.pth`, uses the prompt `Every effort moves you`, and prints 15 generated tokens. It runs inference on CPU. The checkpoint must match the model configuration in `gpt_generate.py`; train with `gpt_train.py` to create a compatible checkpoint.

## Project Files

- `gpt.py` defines the GPT model, text-generation function, and training data loader.
- `gpt_train.py` trains the model on `The_Verdict.txt` and saves the checkpoint and loss plot.
- `gpt_generate.py` loads the checkpoint and generates text from a prompt.
- `The_Verdict.txt` is the training and validation text.
- `model.pth` is a saved model checkpoint.
- `requirements.txt` lists the Python dependencies.