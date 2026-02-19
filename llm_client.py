from base_client import BaseFoundationClient
try:
    import google.generativeai as genai
except ImportError:
    genai = None

class LLMClient(BaseFoundationClient):
    """
    Client for Text-to-Text interaction.
    """
    
    def __call__(self, user_message: str, system_message: str = "You are a helpful assistant.", **kwargs) -> str:
        # Merge call-specific overrides
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        top_p = kwargs.get("top_p", self.top_p)
        stream = kwargs.get("stream", self.stream)
        
        if self.provider in ["groq", "openai"]:
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
            
            # OpenAI/Groq parameters
            params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "stream": stream,
            }
            
            if self.provider == "groq":
                 params["max_completion_tokens"] = max_tokens
            else:
                 params["max_tokens"] = max_tokens

            response = self.client.chat.completions.create(**params)
            
            if stream:
                full_response = ""
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        print(content, end="", flush=True)
                print() # Newline
                return full_response
            else:
                content = response.choices[0].message.content
                if hasattr(response, 'usage'):
                    self._update_metrics(response.usage.prompt_tokens, response.usage.completion_tokens)
                return content

        elif self.provider == "anthropic":
            # Anthropic does not support system message in the messages list in the same way (it's a top level param)
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_message,
                messages=[
                    {"role": "user", "content": user_message}
                ]
            )
            content = response.content[0].text
            self._update_metrics(response.usage.input_tokens, response.usage.output_tokens)
            return content

        elif self.provider == "gemini":
            config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                top_p=top_p,
            )
            
            complete_prompt = f"System: {system_message}\nUser: {user_message}"
            response = self.client.generate_content(complete_prompt, generation_config=config)
            
            usage = response.usage_metadata
            self._update_metrics(usage.prompt_token_count, usage.candidates_token_count)
            
            return response.text

        else:
            raise NotImplementedError(f"Provider {self.provider} not implemented.")
