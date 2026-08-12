from transformers import AutoModel, AutoVideoProcessor, AutoModelForCausalLM, AutoTokenizer

AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256").save_pretrained("./jepa_bridge_qwen/jepa")
AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitl-fpc64-256").save_pretrained("./jepa_bridge_qwen/jepa")
AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct").save_pretrained("./jepa_bridge_qwen/qwen")
AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct").save_pretrained("./jepa_bridge_qwen/qwen")