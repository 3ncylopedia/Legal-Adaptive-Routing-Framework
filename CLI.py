import os
import sys

# Ensure CWD is always the project root so paths and logs are correct even if spawned from IDEs or Applescript
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)

import platform
import subprocess
import time
import logging
from dotenv import load_dotenv, set_key

# --- LOGGING SETUP ---
# Capture full pipeline trace (INFO+) to file, keeping the terminal clean
logging.basicConfig(
    filename=os.path.join(PROJECT_ROOT, 'cli_errors.log'),
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    force=True
)
# Silence noisy third-party library debug logs — keep only our code
for noisy in ('urllib3', 'requests', 'charset_normalizer', 'faiss'):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# --- RETRY SETTINGS ---
MAX_RETRIES = 5
BASE_DELAY = 2

def _is_rate_limited_(error_msg):
    err_str = str(error_msg).lower()
    return "429" in err_str or "rate" in err_str or "rate-limited" in err_str or "too many requests" in err_str

# --- 0. TERMINAL POPUP LOGIC ---
def summon_terminal():
    if "--spawned" in sys.argv:
        # We are already in the spawned terminal
        sys.argv.remove("--spawned")
        return
    
    # Check if we should spawn based on user prompt requirements.
    # We will spawn a new terminal instead of running inside the IDE's terminal.
    system = platform.system()
    try:
        if system == "Windows":
            # Spawn a new cmd window and run this script
            subprocess.Popen(['start', 'cmd', '/k', sys.executable] + sys.argv + ['--spawned'], shell=True)
            sys.exit(0)
        elif system == "Darwin":
            # Spawn a new Terminal window on macOS
            script_path = os.path.abspath(sys.argv[0])
            args = " ".join([f'"{arg}"' for arg in sys.argv[1:]])
            script = f'''
            tell application "Terminal"
                activate
                do script "{sys.executable} \\"{script_path}\\" {args} --spawned"
            end tell
            '''
            subprocess.Popen(['osascript', '-e', script])
            sys.exit(0)
        # Linux isn't explicitly required, will proceed in current terminal
    except Exception as e:
        print(f"Could not spawn terminal: {e}")
        # Proceed in current terminal if spawning fails

summon_terminal()

# --- IMPORTS ---
# Delayed imports to keep the popup fast and avoid loading heavy ML models prematurely
from src.adaptive_routing import (
    FrameworkConfig, 
    TriageModule, 
    SemanticRouterModule, 
    LegalRetrievalModule
)

# --- 1. UI HELPERS ---
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_banner():
    # ANSI Color Codes
    CYAN = '\033[96m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    
    banner = f"""{BLUE}
    ========================================================{CYAN}
     _       __  __  ____  _____ 
    | |     / _ \\|  _ \\|  ___|
    | |    | |_| | |_) | |_   
    | |___ |  _  |  _ <|  _|  
    |_____||_| |_|_| \\_\\_|    
                              
      Legal Adaptive Routing Framework
      {YELLOW}Philippine & Hong Kong Legal Queries{BLUE}
    ========================================================
    {RESET}  Team 404FoundUs | Saint Louis University
    {BLUE}========================================================{RESET}
    """
    print(banner)

# --- 2. CONFIGURATION MENU ---
def interactive_config():
    load_dotenv()
    
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    triage_model = os.getenv("TRIAGE_MODEL", "z-ai/glm-4.5-air:free")
    router_model = os.getenv("ROUTER_MODEL", "z-ai/glm-4.5-air:free")
    general_model = os.getenv("GENERAL_MODEL", "z-ai/glm-4.5-air:free")
    reasoning_model = os.getenv("REASONING_MODEL", "z-ai/glm-4.5-air:free")
    casual_model = os.getenv("CASUAL_MODEL", "z-ai/glm-4.5-air:free")

    while True:
        clear_screen()
        print_banner()
        print("--- Framework Configuration ---\n")
        print(f"1. OpenRouter API Key: {'[SET]' if api_key else '[NOT SET]'}")
        print(f"2. Triage Model:       {triage_model}")
        print(f"3. Router Model:       {router_model}")
        print(f"4. General Model:      {general_model}")
        print(f"5. Reasoning Model:    {reasoning_model}")
        print(f"6. Casual Model:       {casual_model}")
        print("\nS. Save and Start Chat")
        print("Q. Quit")
        
        choice = input("\nSelect an option to edit (1-6, S, Q): ").strip().upper()
        if choice == '1':
            api_key = input("Enter OpenRouter API Key: ").strip() or api_key
        elif choice == '2':
            triage_model = input("Enter Triage Model: ").strip() or triage_model
        elif choice == '3':
            router_model = input("Enter Router Model: ").strip() or router_model
        elif choice == '4':
            general_model = input("Enter General Model: ").strip() or general_model
        elif choice == '5':
            reasoning_model = input("Enter Reasoning Model: ").strip() or reasoning_model
        elif choice == '6':
            casual_model = input("Enter Casual Model: ").strip() or casual_model
        elif choice == 'S':
            break
        elif choice == 'Q':
            print("Exiting...")
            sys.exit(0)

    # Apply configuration
    FrameworkConfig._update_settings_(
        api_key=api_key,
        triage_model=triage_model,
        router_model=router_model,
        general_model=general_model,
        reasoning_model=reasoning_model,
        casual_model=casual_model
    )
    
    save_ans = input("\nSave configuration to .env? (y/n): ").strip().lower()
    if save_ans == 'y':
        env_file = ".env"
        try:
            set_key(env_file, "OPENROUTER_API_KEY", api_key)
            set_key(env_file, "TRIAGE_MODEL", triage_model)
            set_key(env_file, "ROUTER_MODEL", router_model)
            set_key(env_file, "GENERAL_MODEL", general_model)
            set_key(env_file, "REASONING_MODEL", reasoning_model)
            set_key(env_file, "CASUAL_MODEL", casual_model)
            print("Configuration saved to .env.")
        except Exception as e:
            print("Could not save to .env (using temporary config).")

# --- 3. MAIN APP ---
def main():
    clear_screen()
    print_banner()
    if "--fast" not in sys.argv:
        # Prompt for interactive config
        interactive_config()
    else:
        load_dotenv()
        FrameworkConfig._update_settings_(
            api_key=os.getenv("OPENROUTER_API_KEY", ""),
            triage_model=os.getenv("TRIAGE_MODEL", "z-ai/glm-4.5-air:free"),
            router_model=os.getenv("ROUTER_MODEL", "z-ai/glm-4.5-air:free"),
            general_model=os.getenv("GENERAL_MODEL", "z-ai/glm-4.5-air:free"),
            reasoning_model=os.getenv("REASONING_MODEL", "z-ai/glm-4.5-air:free"),
            casual_model=os.getenv("CASUAL_MODEL", "z-ai/glm-4.5-air:free")
        )

    clear_screen()
    print_banner()
    print("\n[System] Initializing Adaptive Routing Framework...")
    
    try:
        triage = TriageModule()
        router = SemanticRouterModule()
        retrieval = LegalRetrievalModule(
            index_path="localfiles/legal-basis/combined_index.faiss",
            chunks_path="localfiles/legal-basis/combined_index.json"
        )
    except Exception as e:
        print(f"\n[Error] Initialization Failed: {e}")
        input("\nPress Enter to exit...")
        sys.exit(1)

    history = []

    print("\n" + "="*56)
    print("                LEGAL ASSISTANT READY")
    print("="*56)
    print(" Type 'exit' or 'quit' to end the session.")
    print(" Type 'clear' to clear conversation history.")
    print("="*56 + "\n")

    while True:
        try:
            user_input = input("\n👤 User: ").strip()
            if not user_input:
                continue
                
            if user_input.lower() in ['exit', 'quit']:
                print("\nGoodbye!")
                break
            if user_input.lower() == 'clear':
                history = []
                clear_screen()
                print_banner()
                print("\n[System] History cleared.\n")
                continue

            # Stage 1: Triage (Normalization)
            triage_result = None
            detected_language = "Unknown"
            normalized_text = user_input
            
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    triage_result = triage._process_request_(user_input)
                    normalized_text = triage_result.get("normalized_text", user_input)
                    detected_language = triage_result.get("detected_language", "Unknown")
                    if triage_result.get("error"):
                        raise Exception(triage_result["error"])
                    break
                except Exception as e:
                    logging.error(f"Triage error on attempt {attempt}: {e}")
                    if _is_rate_limited_(e) and attempt < MAX_RETRIES:
                        delay = BASE_DELAY * attempt
                        print(f"   ⏳ [Triage] Rate-limited. Retrying in {delay}s...")
                        time.sleep(delay)
                    else:
                        print(f"   ⚠️ [Triage] Failed to normalize text. Using raw input.")
                        break

            logging.info(f"[Triage] language={detected_language!r} normalized={normalized_text!r}")
            print(f"\n⚙️  [Triage] Language: {detected_language}")
            print(f"   [Normalized] {normalized_text}")

            # Stage 2: Classification Only
            classification = {"route": "General-LLM", "confidence": 0.0, "trigger_signals": []}
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    result = router._process_routing_(normalized_text, threshold=0.1)
                    if result.get("error"):
                        if result.get("error") == "LLMEngine failed to acknowledge the input.":
                            classification = {
                                "route": "Casual-LLM",
                                "confidence": 1.0,
                                "trigger_signals": ["Fallback due to threshold failure"]
                            }
                            break
                        else:
                            raise Exception(result["error"])
                    classification = result
                    break
                except Exception as e:
                    logging.error(f"Routing error on attempt {attempt}: {e}")
                    if _is_rate_limited_(e) and attempt < MAX_RETRIES:
                        delay = BASE_DELAY * attempt
                        print(f"   ⏳ [Router] Rate-limited. Retrying in {delay}s...")
                        time.sleep(delay)
                    else:
                        print(f"   ⚠️ [Router] Failed to classify route. Defaulting to General-LLM.")
                        break

            route = classification.get("route", "General-LLM")
            logging.info(f"[Router] route={route!r} confidence={classification.get('confidence', 0.0):.2f} signals={classification.get('trigger_signals', [])}")
            print(f"🔀 [Router] Selected Pathway: {route} (Confidence: {classification.get('confidence', 0.0):.2f})")

            # Stage 3: RAG Retrieval (skip for Casual)
            context_str = None
            if route != "Casual-LLM":
                try:
                    retrieval_output = retrieval._process_retrieval_(normalized_text)
                    chunks = retrieval_output.get("retrieved_chunks", [])
                    if chunks:
                        context_str = "\n\n".join([c.get("chunk", "") for c in chunks[:3]])
                        print(f"📚 [RAG] Retrieved {len(chunks[:3])} relevant legal sources.")
                except Exception as e:
                    logging.error(f"Retrieval error: {e}")
                    print(f"   ⚠️ [RAG] Retrieval encountered an error. Proceeding without context.")

            # Stage 4: Generation (Multi-Turn)
            # Add the user's message to history BEFORE generation so the API always has messages
            history.append({"role": "user", "content": normalized_text})

            print("🤖 [Assistant] Thinking...")
            response = ""
            accepted = False
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    gen_result = router._generate_conversation_(
                        classification=classification,
                        messages=history,
                        context=context_str
                    )
                    if gen_result.get("error"):
                        raise Exception(gen_result["error"])
                    raw_response = gen_result.get("response_text")
                    logging.info(f"[Generation] attempt={attempt} raw_response_type={type(raw_response).__name__!r} raw_response_value={raw_response!r}")
                    response = raw_response or "No response generated."
                    accepted = gen_result.get("accepted", True)
                    break
                except Exception as e:
                    logging.error(f"Generation error on attempt {attempt}: {e}")
                    if _is_rate_limited_(e) and attempt < MAX_RETRIES:
                        delay = BASE_DELAY * attempt
                        print(f"   ⏳ [Generator] Rate-limited. Retrying in {delay}s...")
                        time.sleep(delay)
                    else:
                        response = "I am currently unable to process your query due to a technical error. Please check cli_errors.log."
                        accepted = False
                        break

            # Update history with the assistant's response
            history.append({"role": "assistant", "content": response})

            # Output
            accepted_str = "✓ Accepted" if accepted else "✗ Requires Review / Error"
            print(f"✅ [Status] {accepted_str}")
            print("\n==================== RESPONSE ====================")
            print(f"{response}")
            print("==================================================\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logging.error(f"Unexpected main loop error: {e}")
            print(f"\n❌ [Error] An unexpected error occurred. See cli_errors.log for details.")

if __name__ == "__main__":
    main()
