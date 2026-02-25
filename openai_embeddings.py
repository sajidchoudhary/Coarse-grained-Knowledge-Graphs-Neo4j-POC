class OpenAIEmbeddingClient:

    def __init__(self, model="text-embedding-3-small", api_key=None, http_client=None):
        import os, httpx
        from openai import OpenAI

        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set")

        if http_client is None:
            http_client = httpx.Client(verify=False)

        self.client = OpenAI(
            api_key=self.api_key,
            http_client=http_client,
        )

    def embed_documents(self, texts):
        resp = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )
        return [d.embedding for d in resp.data]

    def embed_query(self, text):
        resp = self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        return resp.data[0].embedding

    # 🔥 THIS FIXES FAISS
    def __call__(self, text):
        return self.embed_query(text)
