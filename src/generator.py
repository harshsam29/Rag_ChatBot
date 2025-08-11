from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

class Generator:
    def __init__(self):
        try:
            self.llm = ChatOllama(model="mistral", temperature=0)
            self.prompt_template = ChatPromptTemplate.from_template(
                """
                You are a helpful assistant answering questions based on the provided document.
                Use only the following context to answer the query factually and concisely. 
                If the information is not available or unclear, state that and suggest contacting support or escalating via arbitration if applicable.
                Context: {context}
                Query: {query}
                Answer:
                """
            )
            logger.info("Generator initialized with Mistral model")
        except Exception as e:
            logger.error(f"Error initializing Generator: {e}")
            raise

    def generate(self, query, context_chunks_with_meta):
        try:
            if not context_chunks_with_meta:
                return "No relevant context found. Please contact support for assistance.", []
            context = "\n".join([f"Page {m.get('page', 'N/A')}: {c}" for c, m in context_chunks_with_meta])
            prompt = self.prompt_template.format(context=context, query=query)
            response = ""
            for chunk in self.llm.stream(prompt):
                if hasattr(chunk, 'content'):
                    response += chunk.content
                else:
                    logger.warning("Unexpected chunk format in stream")
            return response.strip(), [c for c, _ in context_chunks_with_meta]
        except Exception as e:
            logger.error(f"Error generating response for query '{query}': {e}")
            return f"Error generating response: {e}. Please try again.", []