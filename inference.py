import os
import sys
import json
from typing import List, Optional
from openai import OpenAI
from env.environment import FulfillmentEnv
from env.models import Action, ActionType

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error.replace("\n", " ") if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def main():
    API_BASE_URL = os.getenv("API_BASE_URL") or "https://api.openai.com/v1"
    MODEL_NAME = os.getenv("MODEL_NAME") or "gpt-4o-mini"
    API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
    BENCHMARK = "openenv_fulfillment"
    TASK_NAME = "task_easy"
    
    if not API_KEY:
        API_KEY = "dummy_mock_key"
        
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    env = FulfillmentEnv(seed=42)
    obs = env.reset(task_id=TASK_NAME)
    
    rewards: List[float] = []
    steps_taken = 0
    done = False
    
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
    
    system_prompt = "You are an AI order fulfillment agent. Analyze the text state provided and return a JSON with keys 'action' and 'target_id'."
    
    try:
        while not done:
            steps_taken += 1
            obs_dict = obs.to_dict()
            
            prompt = f"Observation: orders={obs_dict['orders_pending']} low_stock={len(obs_dict['low_stock_items'])} returns={obs_dict['returns_pending']} queue={obs_dict['customer_queue']}"
            action_chosen = None
            error_msg = None
            
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=60,
                    temperature=0.1
                )
                raw = response.choices[0].message.content
                if '{' in raw and '}' in raw:
                    action_chosen = json.loads(raw[raw.find('{'):raw.rfind('}')+1])
            except Exception as e:
                error_msg = str(e)
                
            if not action_chosen or "action" not in action_chosen:
                if obs_dict.get("orders_pending", 0) > 0:
                    action_chosen = {"action": ActionType.PROCESS_ORDER.value, "target_id": None}
                elif obs_dict.get("urgent_inquiries", 0) > 0:
                    action_chosen = {"action": ActionType.HANDLE_CUSTOMER.value, "target_id": None}
                elif obs_dict.get("returns_pending", 0) > 0:
                    action_chosen = {"action": ActionType.PROCESS_RETURN.value, "target_id": None}
                elif len(obs_dict.get("low_stock_items", [])) > 0:
                    action_chosen = {"action": ActionType.RESTOCK_ITEM.value, "target_id": obs_dict["low_stock_items"][0]}
                elif obs_dict.get("customer_queue", 0) > 0:
                    action_chosen = {"action": ActionType.HANDLE_CUSTOMER.value, "target_id": None}
                else:
                    action_chosen = {"action": ActionType.IDLE.value, "target_id": None}
                
            try:
                action = Action(
                    action_type=ActionType(action_chosen.get("action", "idle")),
                    target_id=action_chosen.get("target_id")
                )
            except ValueError as e:
                action = Action(ActionType.IDLE)
                error_msg = str(e)
            
            action_str = f"{action.action_type.value}"
            if action.target_id:
                action_str += f"('{action.target_id}')"
                
            try:
                result = env.step(action)
                obs = env.get_observation()
                reward = result.reward or 0.0
                done = result.done
                
                if hasattr(result, 'info') and isinstance(result.info, dict) and "error" in result.info:
                    error_msg = result.info["error"]
                    
            except Exception as e:
                reward = 0.0
                done = True
                error_msg = str(e)
            
            rewards.append(reward)
            log_step(step=steps_taken, action=action_str, reward=reward, done=done, error=error_msg)
            
    finally:
        score = env.calculate_score()
        success = score >= 0.7 
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    main()
