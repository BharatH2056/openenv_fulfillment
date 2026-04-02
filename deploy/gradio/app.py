from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import gradio as gr
import threading
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from env.environment import FulfillmentEnv
from deploy.gradio.gradio_ui import demo

app = FastAPI(title="OpenEnv E-commerce Fulfillment API", version="1.0.0")

# Global variables for thread-safe state management
env_lock = threading.Lock()
global_env = FulfillmentEnv(seed=42)
global_env.reset()

class ResetRequest(BaseModel):
    task_id: str = "task_easy"
    seed: int | None = None

class StepRequest(BaseModel):
    action_type: str
    target_id: str | None = None
    parameters: dict = {}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/reset")
@app.get("/reset")
def reset_env(req: ResetRequest = ResetRequest()):
    with env_lock:
        if req.seed is not None:
            global_env.seed = req.seed
        try:
            obs = global_env.reset(task_id=req.task_id)
            return {"status": "success", "observation": obs.to_dict()}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

@app.post("/step")
def step_env(req: StepRequest):
    with env_lock:
        try:
            action_dict = {
                "action_type": req.action_type,
                "target_id": req.target_id,
                "parameters": req.parameters
            }
            result = global_env.step(action_dict)
            return {
                "observation": global_env.get_observation().to_dict(),
                "reward": result.reward,
                "done": result.done,
                "info": result.info
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

@app.get("/state")
def get_state():
    with env_lock:
        try:
            obs = global_env.get_observation()
            return obs.to_dict()
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

# Mount the gradio app on the root
app = gr.mount_gradio_app(app, demo, path="/")
