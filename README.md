# Agribot — T5 Small Streamlit demo

This small demo shows how to run a T5 small model (TensorFlow) inside a Streamlit app. It includes guidance to avoid the common `keras`/`tensorflow` import conflicts.

## Setup (Windows, bash)

1. Create and activate a clean virtual environment (recommended):

```bash
python -m venv envs
source envs/Scripts/activate
```

2. Upgrade pip and install requirements:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Important: Do NOT install the standalone `keras` package. If you previously installed it, run `pip uninstall keras` before installing `tensorflow`.

If you have an NVIDIA GPU and want GPU acceleration, ensure your CUDA/cuDNN versions are compatible with the chosen TensorFlow version (see TensorFlow release notes). For Windows, the easiest path is to use CPU-only TF unless you already have a working GPU driver stack.

## Run

```bash
streamlit run streamlit_app.py
```

The app will display GPU detection, load the T5 tokenizer and TF seq2seq model (with a fallback to snapshot download if remote loading fails), and provide a simple QA interface.

## Troubleshooting

- If you see errors like `ModuleNotFoundError: No module named 'keras'` or import conflicts, make sure you didn't pip-install `keras` separately. TensorFlow 2.x includes Keras as `tf.keras`.
- If you experience TF/CUDA errors, either install a TF version matching your CUDA toolkit or use CPU-only TF.
- On slow machines, model loading can take a while. Consider using a smaller model or running the model on a machine with sufficient RAM.

## Next steps

- Add caching for model generation to speed up repeated questions.
- Add batching and rate-limiting for production deployments.
- Provide a Dockerfile for reproducible environments.
