from ollama import chat


class OllamaLLM:
    def __init__(self, model: str):
        self.model = model

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        response = chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            think=False,
            options={
                "temperature": 0,
                "num_predict": 256,
                "num_ctx": 2048,
            },
        )

        print("\n--- LLM PERFORMANCE ---")

        if response.load_duration:
            print(
                f"Model load: "
                f"{response.load_duration / 1e9:.2f}s"
            )

        if response.prompt_eval_duration:
            print(
                f"Prompt processing: "
                f"{response.prompt_eval_duration / 1e9:.2f}s"
            )

        if response.eval_duration:
            generation_seconds = (
                response.eval_duration / 1e9
            )

            print(
                f"Generation: "
                f"{generation_seconds:.2f}s"
            )

            print(
                f"Generated tokens: "
                f"{response.eval_count}"
            )

            if generation_seconds > 0:
                print(
                    f"Generation speed: "
                    f"{response.eval_count / generation_seconds:.2f} tok/s"
                )

        print("-----------------------\n")

        return response.message.content