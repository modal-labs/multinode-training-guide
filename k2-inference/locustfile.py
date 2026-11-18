import logging
import random

import locust


messages = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant. Keep your responses concise and accurate.",
    },
    {
        "role": "user",
        "content": "What are the key differences between machine learning and deep learning?",
    },
]

alternative_messages = [
    [
        {
            "role": "system", 
            "content": "You are an expert software engineer. Provide clear technical explanations."
        },
        {
            "role": "user",
            "content": "Explain the benefits of using async/await in Python programming.",
        },
    ],
    [
        {
            "role": "system",
            "content": "You are a data science expert. Help with analytics questions.",
        },
        {
            "role": "user", 
            "content": "How do you handle missing data in a machine learning pipeline?",
        },
    ],
    [
        {
            "role": "user",
            "content": "Write a Python function to calculate the Fibonacci sequence up to n terms.",
        },
    ],
    [
        {
            "role": "user",
            "content": "Explain quantum computing in simple terms for a beginner.",
        },
    ],
]


class K2InferenceUser(locust.HttpUser):
    wait_time = locust.between(1, 3)
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    @locust.task(weight=10)
    def chat_completion(self):
        """Standard chat completion request"""
        payload = {
            "model": "kimi-k2",
            "messages": random.choice([messages] + alternative_messages),
            "max_tokens": 512,
            "temperature": 0.7,
        }
        response = self.client.request(
            "POST", "/v1/chat/completions", json=payload, headers=self.headers
        )
        response.raise_for_status()
        
        # Log 1% of responses for debugging
        if random.random() < 0.01:
            response_data = response.json()
            if "choices" in response_data and response_data["choices"]:
                content = response_data["choices"][0]["message"]["content"]
                logging.info(f"Sample response: {content[:100]}...")

    @locust.task(weight=3)
    def chat_completion_streaming(self):
        """Streaming chat completion request"""
        payload = {
            "model": "kimi-k2",
            "messages": random.choice([messages] + alternative_messages),
            "max_tokens": 256,
            "temperature": 0.7,
            "stream": True,
        }
        with self.client.request(
            "POST", "/v1/chat/completions", json=payload, headers=self.headers, stream=True
        ) as response:
            response.raise_for_status()
            # Consume the stream to simulate real usage
            for line in response.iter_lines():
                if line and line.startswith(b"data: "):
                    # Process SSE data
                    pass

    @locust.task(weight=1)
    def models_endpoint(self):
        """Test models endpoint"""
        response = self.client.get("/v1/models", headers=self.headers)
        response.raise_for_status()

    @locust.task(weight=1) 
    def health_check(self):
        """Test health endpoint"""
        response = self.client.get("/health", headers=self.headers)
        response.raise_for_status()
