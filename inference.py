import os
import sys
import json
import datetime
from openai import OpenAI
from env.environment import FulfillmentEnv
from env.models import Action, ActionType

def main():
    api_base_url = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    hf_token = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
    
    if not hf_token:
        print("Warning: Neither HF_TOKEN nor OPENAI_API_KEY is set. Authentication may fail.", file=sys.stderr)
        hf_token = "dummy_mock_key_unauthenticated"
        
    client = OpenAI(base_url=api_base_url, api_key=hf_token)
    
    task_id = "task_easy"
    
    # Structured log format compliance
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    print(f"[START] Timestamp: {timestamp} | Task: {task_id} | Model: {model_name}")
    
    env = FulfillmentEnv(seed=42)
    obs = env.reset(task_id=task_id)
    
    total_reward = 0.0
    step = 0
    done = False
    
    def heuristic_fallback(obs_dict):
        if obs_dict.get("orders_pending", 0) > 0:
            return {"action": ActionType.PROCESS_ORDER.value, "target_id": None}
        if obs_dict.get("urgent_inquiries", 0) > 0:
            return {"action": ActionType.HANDLE_CUSTOMER.value, "target_id": None}
        if obs_dict.get("returns_pending", 0) > 0:
            return {"action": ActionType.PROCESS_RETURN.value, "target_id": None}
        if len(obs_dict.get("low_stock_items", [])) > 0:
            return {"action": ActionType.RESTOCK_ITEM.value, "target_id": obs_dict["low_stock_items"][0]}
        if obs_dict.get("customer_queue", 0) > 0:
            return {"action": ActionType.HANDLE_CUSTOMER.value, "target_id": None}
        return {"action": ActionType.IDLE.value, "target_id": None}
    
    system_prompt = "You are an AI order fulfillment agent. Analyze the text state provided and return a JSON with keys 'action' and 'target_id'."
    
    while not done:
        step += 1
        obs_dict = obs.to_dict()
        
        prompt = f"Observation: orders={obs_dict['orders_pending']} low_stock={len(obs_dict['low_stock_items'])} returns={obs_dict['returns_pending']} queue={obs_dict['customer_queue']}"
        action_chosen = None
        
        try:
            response = client.chat.completions.create(
                model=model_name,
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
            # Mask sensitive data in logs explicitly if logging exceptions 
            pass
            
        if not action_chosen or "action" not in action_chosen:
            action_chosen = heuristic_fallback(obs_dict)
            
        try:
            action = Action(
                action_type=ActionType(action_chosen.get("action", "idle")),
                target_id=action_chosen.get("target_id")
            )
        except ValueError:
            action = Action(ActionType.IDLE)
        
        result = env.step(action)
        total_reward += result.reward
        done = result.done
        obs = env.get_observation()
        
        # Structured Logging Compliance
        print(f"[STEP] {step} | Action: {action.action_type.value} | Target: {action.target_id} | Reward: {result.reward:.3f} | TotalReward: {total_reward:.3f} | OrderCompleted: {obs_dict['orders_completed']}")
        
    final_score = env.calculate_score()
    print(f"[END] Timestamp: {datetime.datetime.now(datetime.timezone.utc).isoformat()} | Task: {task_id} | Final Score: {final_score:.3f}")

if __name__ == "__main__":
    main()
