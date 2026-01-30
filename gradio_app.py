
import gradio as gr
import json
import os
import requests
import uuid
import time
import websocket
import random
from PIL import Image
import io
import urllib.parse

# ComfyUI Host
COMFY_HOST = "127.0.0.1:8188"
WORKFLOW_DIR = os.path.join(os.path.dirname(__file__), "workflows")
os.makedirs(WORKFLOW_DIR, exist_ok=True)

# Function to get available models (checkpoints)
def get_available_models():
    try:
        response = requests.get(f"http://{COMFY_HOST}/object_info/CheckpointLoaderSimple")
        if response.status_code == 200:
            data = response.json()
            return data.get("CheckpointLoaderSimple", {}).get("input", {}).get("required", {}).get("ckpt_name", [[]])[0]
    except Exception as e:
        print(f"Error fetching models: {e}")
        return ["flux1-dev-fp8.safetensors"] # Fallback

# Function to get available workflows
def get_workflows():
    files = [f for f in os.listdir(WORKFLOW_DIR) if f.endswith(".json")]
    return files if files else ["No workflows found"]

def queue_prompt(prompt_workflow):
    p = {"prompt": prompt_workflow, "client_id": str(uuid.uuid4())}
    data = json.dumps(p).encode('utf-8')
    resp = requests.post(f"http://{COMFY_HOST}/prompt", data=data)
    return resp.json()

def get_image(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with requests.get(f"http://{COMFY_HOST}/view?{url_values}", stream=True) as response:
        response.raise_for_status()
        return response.content

def get_history(prompt_id):
    with requests.get(f"http://{COMFY_HOST}/history/{prompt_id}") as response:
        return response.json()

def generate_image(workflow_name, model_name, positive_prompt, negative_prompt, seed, randomize_seed):
    if workflow_name == "No workflows found":
        raise gr.Error("No workflow selected.")
    
    workflow_path = os.path.join(WORKFLOW_DIR, workflow_name)
    try:
        with open(workflow_path, "r", encoding="utf-8") as f:
            workflow = json.load(f)
    except Exception as e:
        raise gr.Error(f"Failed to load workflow: {e}")

    # Update workflow with inputs
    # This logic assumes standard ComfyUI node types. 
    # It attempts to find nodes by class_type or title.
    
    # 1. Update Checkpoint
    checkpoint_loader = None
    for node_id, node in workflow.items():
        if node.get("class_type") == "CheckpointLoaderSimple":
            checkpoint_loader = node
            break
    
    if checkpoint_loader:
        checkpoint_loader["inputs"]["ckpt_name"] = model_name
    
    # 2. Update Prompts
    # We look for CLIPTextEncode. Usually one is connected to 'positive' and one to 'negative' on KSampler.
    # But determining which is which without walking the graph is tricky.
    # Heuristic: Check _meta title if available.
    
    # Reset prompts
    pos_node = None
    neg_node = None

    for node_id, node in workflow.items():
        if node.get("class_type") == "CLIPTextEncode":
            title = node.get("_meta", {}).get("title", "").lower()
            if "positive" in title:
                pos_node = node
            elif "negative" in title:
                neg_node = node
            # Fallback: if we haven't found named ones, we might need graph traversal, 
            # but for now let's rely on standard conventions or the first/second found.

    # If titles aren't helpful, we might resort to simple overwrites if there are exactly 2 text nodes.
    if not pos_node or not neg_node:
        text_nodes = [n for n in workflow.values() if n.get("class_type") == "CLIPTextEncode"]
        if len(text_nodes) >= 1:
            pos_node = text_nodes[0]
        if len(text_nodes) >= 2:
            neg_node = text_nodes[1]
            
    if pos_node:
        pos_node["inputs"]["text"] = positive_prompt
    if neg_node:
        neg_node["inputs"]["text"] = negative_prompt

    # 3. Update Seed (KSampler)
    if randomize_seed:
        seed = random.randint(1, 100000000000000)
    
    for node_id, node in workflow.items():
        if node.get("class_type") == "KSampler":
            if "seed" in node["inputs"]:
                node["inputs"]["seed"] = seed
    
    # Queue Prompt
    try:
        client_id = str(uuid.uuid4())
        ws = websocket.WebSocket()
        ws.connect(f"ws://{COMFY_HOST}/ws?clientId={client_id}")
        
        p = {"prompt": workflow, "client_id": client_id}
        data = json.dumps(p).encode('utf-8')
        req = requests.post(f"http://{COMFY_HOST}/prompt", data=data)
        prompt_id = req.json()["prompt_id"]
        
        # Wait for completion
        while True:
            out = ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message['type'] == 'executing':
                    data = message['data']
                    if data['node'] is None and data['prompt_id'] == prompt_id:
                        break # Execution finished
        
        # Get History
        history = get_history(prompt_id)[prompt_id]
        outputs = history['outputs']
        
        images = []
        for node_id in outputs:
            node_output = outputs[node_id]
            if 'images' in node_output:
                for image in node_output['images']:
                    img_data = get_image(image['filename'], image['subfolder'], image['type'])
                    images.append(Image.open(io.BytesIO(img_data)))
        
        return images, seed
        
    except Exception as e:
        raise gr.Error(f"Generation failed: {e}")

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# ComfyUI Worker Interface")
    
    with gr.Row():
        with gr.Column():
            with gr.Row():
                workflow_dropdown = gr.Dropdown(
                    label="Select Workflow", 
                    choices=get_workflows(), 
                    value=get_workflows()[0] if get_workflows() else None,
                    interactive=True
                )
                refresh_btn = gr.Button("Refresh Workflows", size="sm")

            model_dropdown = gr.Dropdown(
                label="Select Model",
                choices=get_available_models(),
                value="flux1-dev-fp8.safetensors",
                allow_custom_value=True
            )
            
            prompt_input = gr.Textbox(label="Positive Prompt", lines=3, value="A beautiful landscape")
            neg_prompt_input = gr.Textbox(label="Negative Prompt", lines=2, value="bad quality, blurry")
            
            with gr.Row():
                seed_input = gr.Number(label="Seed", value=12345, precision=0)
                randomize_check = gr.Checkbox(label="Randomize Seed", value=True)
            
            generate_btn = gr.Button("Generate", variant="primary")
            
        with gr.Column():
            gallery = gr.Gallery(label="Generated Images")
            output_seed = gr.Number(label="Used Seed")

    def refresh_lists():
        return gr.update(choices=get_workflows()), gr.update(choices=get_available_models())

    refresh_btn.click(refresh_lists, outputs=[workflow_dropdown, model_dropdown])
    
    generate_btn.click(
        fn=generate_image,
        inputs=[workflow_dropdown, model_dropdown, prompt_input, neg_prompt_input, seed_input, randomize_check],
        outputs=[gallery, output_seed]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
