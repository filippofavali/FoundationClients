class HomeworkPrompts:
    """
    Centralized collection of prompts used for the Homework Assistant.
    Supports both vanilla (zero-shot) and ReAct (Reasoning and Acting) paradigms.
    """
    
    @staticmethod
    def get_vanilla_prompt(task_description: str) -> str:
        """
        Returns a simple, zero-shot evaluation prompt for a single image.
        """
        return (
            f"The user is currently performing the following task: '{task_description}'. "
            f"Based on this image, evaluate if they are doing it correctly. Provide brief, "
            f"constructive feedback. If you cannot see them or the task clearly, state that."
        )

    @staticmethod
    def get_react_system_prompt() -> str:
        """
        Returns the system-level instruction for the ReAct loop.
        Note: If a VLM endpoint does not natively support system prompts or is text-image combined,
        this can be prepended to the user prompt.
        """
        return (
            "You are a helpful and observable Homework Assistant. "
            "You evaluate images of users performing tasks using a Thought and Action loop.\n"
            "You MUST structure every response exactly as follows:\n\n"
            "Thought: [Explain your reasoning about what you see in the image and whether the task is being done correctly. "
            "Consider context from any previous thoughts if provided.]\n"
            "Action: [A brief, constructive feedback string to be displayed visually to the user.]\n\n"
            "Do not include any other formatting. Always include exactly one Thought and exactly one Action."
        )

    @staticmethod
    def get_react_prompt(task_description: str, past_thoughts: list[str]) -> str:
        """
        Returns the execution prompt for the ReAct loop taking into account past inferences.
        """
        prompt = (
            f"The user is currently performing the task: '{task_description}'.\n"
            "Analyze the image and provide your analysis using the Thought/Action format.\n"
        )
        
        if past_thoughts:
            prompt += "For context, here are your Thoughts from previous consecutive frames:\n"
            for i, past_thought in enumerate(past_thoughts):
                prompt += f"Previous Thought {i+1}: {past_thought}\n"
                
        return prompt
