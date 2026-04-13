import cv2, time, os, threading, re
from PIL import Image
from src.vlm_client import VLMClient
from examples_of_usage.homework_assistant_prompts import HomeworkPrompts

class HomeworkAssistant:
    """
    A homework assistant that monitors a task using a webcam and evaluates it using a VLM.
    """
    def __init__(self, task_description: str, check_interval_seconds: int = 10, model_name: str = "groq/llama-3.2-11b-vision-preview", evaluation_mode: str = "vanilla"):
        self.task_description = task_description
        self.check_interval = check_interval_seconds
        self.evaluation_mode = evaluation_mode
        
        # Initialize the VLM Client using the specified model. 
        # By default, it expects the provider prefix like "groq/"
        self.vlm = VLMClient(model_name=model_name)
        
        self.is_running = False
        self.last_check_time = 0.0
        self.latest_feedback = "Waiting for first check..."
        self.past_thoughts: list[str] = []
    
    def process_frame(self, frame):
        """Processes a single frame by sending it to the VLM for feedback."""
        try:
            # Convert cv2 BGR frame to RGB for PIL
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            
            if self.evaluation_mode == "vanilla":
                prompt = HomeworkPrompts.get_vanilla_prompt(self.task_description)
                # Call the VLM client
                feedback = self.vlm(text_prompt=prompt, image=pil_img)
                self.latest_feedback = feedback
                print(f"VLM Vanilla Feedback: {feedback}")
                
            elif self.evaluation_mode == "react":
                system_prompt = HomeworkPrompts.get_react_system_prompt()
                user_prompt = HomeworkPrompts.get_react_prompt(self.task_description, self.past_thoughts)
                
                # Combine system prompt if the VLM interface doesn't expose it directly for multimodal
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
                
                response = self.vlm(text_prompt=full_prompt, image=pil_img)
                print(f"Raw ReAct Response:\n{response}\n")
                
                # Parse Thought and Action
                thought_match = re.search(r"Thought:\s*(.*?)(?=Action:|$)", response, re.IGNORECASE | re.DOTALL)
                action_match = re.search(r"Action:\s*(.*)", response, re.IGNORECASE | re.DOTALL)
                
                thought = thought_match.group(1).strip() if thought_match else "No thought parsed."
                action = action_match.group(1).strip() if action_match else "Could not determine action."
                
                # Store thought for history (keep last 3 to avoid context window blowing up)
                self.past_thoughts.append(thought)
                if len(self.past_thoughts) > 3:
                     self.past_thoughts.pop(0)
                     
                self.latest_feedback = action
                print(f"Parsed Action: {action}")
            else:
                self.latest_feedback = f"Unknown evaluation mode: {self.evaluation_mode}"
            
        except Exception as e:
            self.latest_feedback = f"Error calling VLM: {e}"
            print(self.latest_feedback)

    def start(self):
        """Starts the webcam capture loop and periodic evaluation."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open the webcam.")
            return

        self.is_running = True
        print(f"Started Homework Assistant for task: '{self.task_description}'")
        print("Press 'q' in the video window to quit.")

        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab a frame from the webcam.")
                break
                
            current_time = time.time()
            if current_time - self.last_check_time >= self.check_interval:
                self.last_check_time = current_time
                print("\n--- Capturing frame for VLM evaluation ---")
                
                # Run VLM check in a separate daemon thread to avoid freezing the video feed
                check_thread = threading.Thread(target=self.process_frame, args=(frame.copy(),))
                check_thread.daemon = True
                check_thread.start()

            # Display the video feed with the latest feedback
            display_frame = frame.copy()
            
            # Simple text wrapping for the OpenCV display
            y = 30
            for line in self._wrap_text(self.latest_feedback):
                # Using black text with a green outline for readability on various backgrounds
                cv2.putText(display_frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                cv2.putText(display_frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                y += 25
                
            cv2.imshow('Homework Assistant', display_frame)

            # Wait for 1 ms and check if 'q' is pressed to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.is_running = False

        # Clean up resources
        cap.release()
        cv2.destroyAllWindows()

    def _wrap_text(self, text: str, max_chars: int = 60) -> list[str]:
        """Helper to wrap text for OpenCV putText."""
        words: list[str] = text.split()
        lines = []
        current_line = []
        current_length = 0
        for word in words:
            if current_length + len(word) + 1 <= max_chars:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                lines.append(" ".join(current_line))
                current_line = [word]
                current_length = len(word) + 1
        if current_line:
            lines.append(" ".join(current_line))
        return lines

if __name__ == "__main__":
    # Ensure there's a default API key or prompt the user
    if not os.getenv("GROQ_API_KEY"):
         print("WARNING: GROQ_API_KEY environment variable is not set.")
         print("The VLM request will fail unless GROQ_API_KEY is configured.")
         # os.environ["GROQ_API_KEY"] = "your_api_key_here" 

    # Example task
    task_to_monitor = "Changing a light bulb"
    
    assistant = HomeworkAssistant(
        task_description=task_to_monitor,
        check_interval_seconds=10,
        # Using a Groq vision model as specified in the example
        model_name="groq/llama-3.2-11b-vision-preview",
        evaluation_mode="react" # You can toggle this to "vanilla" for ablation studies
    )
    
    # Start the assistant
    assistant.start()
