import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.vectorstores import VectorStore
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from pydantic import SecretStr
from task._constants import DIAL_URL, API_KEY


SYSTEM_PROMPT = """You are a RAG-powered assistant that assists users with their questions about microwave usage.
            
## Structure of User message:
`RAG CONTEXT` - Retrieved documents relevant to the query.
`USER QUESTION` - The user's actual question.

## Instructions:
- Use information from `RAG CONTEXT` as context when answering the `USER QUESTION`.
- Cite specific sources when using information from the context.
- Answer ONLY based on conversation history and RAG context.
- If no relevant information exists in `RAG CONTEXT` or conversation history, state that you cannot answer the question.
"""

USER_PROMPT = """##RAG CONTEXT:
{context}


##USER QUESTION: 
{query}"""


class MicrowaveRAG:

    def __init__(self, embeddings: AzureOpenAIEmbeddings, llm_client: AzureChatOpenAI):
        self.llm_client = llm_client
        self.embeddings = embeddings
        self.vectorstore = self._setup_vectorstore()

    def _setup_vectorstore(self) -> VectorStore:
        """Initialize the RAG system"""
        print("🔄 Initializing Microwave Manual RAG System...")

        index_dir = os.path.join(os.path.dirname(__file__), "microwave_faiss_index")
        # If index exists locally - load it
        if os.path.isdir(index_dir):
            print("📂 Found existing FAISS index. Loading from 'microwave_faiss_index'...")
            try:
                vectorstore = FAISS.load_local(index_dir, self.embeddings, allow_dangerous_serialization=True)
                print("✅ Loaded existing FAISS index.")
                return vectorstore
            except Exception as e:
                print(f"⚠️ Failed to load existing index (will recreate): {e}")

        # Otherwise create a new index from source documents
        print("⚠️ No existing FAISS index found. Creating a new one...")
        return self._create_new_index()

    def _create_new_index(self) -> VectorStore:
        print("📖 Loading text document...")
        base_dir = os.path.dirname(__file__)
        file_path = os.path.join(base_dir, "microwave_manual.txt")

        # 1. Create loader and load documents
        loader = TextLoader(file_path=file_path, encoding="utf-8")
        documents = loader.load()

        # 2. Split documents into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            separators=["\n\n", "\n", "."],
        )
        print("🔧 Splitting documents into chunks (chunk_size=300, overlap=50)...")
        chunks = splitter.split_documents(documents)
        print(f"📦 Creating FAISS vectorstore from {len(chunks)} chunks...")

        # 3. Create vector store and save locally
        vectorstore = FAISS.from_documents(chunks, self.embeddings)
        index_dir = os.path.join(base_dir, "microwave_faiss_index")
        vectorstore.save_local(index_dir)
        print(f"✅ Saved FAISS index to '{index_dir}'")

        return vectorstore

    def retrieve_context(self, query: str, k: int = 4, score=0.3) -> str:
        """
        Retrieve the context for a given query.
        Args:
              query (str): The query to retrieve the context for.
              k (int): The number of relevant documents(chunks) to retrieve.
              score (float): The similarity score between documents and query. Range 0.0 to 1.0.
        """
        print(f"{'=' * 100}\n🔍 STEP 1: RETRIEVAL\n{'-' * 100}")
        print(f"Query: '{query}'")
        print(f"Searching for top {k} most relevant chunks with similarity score {score}:")

        # Perform similarity search (get documents with scores)
        try:
            results = self.vectorstore.similarity_search_with_score(query=query, k=k)
        except TypeError:
            # Some implementations accept named params (score_threshold)
            results = self.vectorstore.similarity_search_with_score(query=query, k=k, score_threshold=score)

        context_parts = []
        # Iterate through results and collect relevant chunks based on score threshold
        for doc, sc in results:
            # If score is provided as similarity in range [0,1], filter by >= score
            try:
                numeric_score = float(sc)
            except Exception:
                numeric_score = None

            if numeric_score is None:
                # If score isn't numeric for some reason, include the chunk
                print("Result score: None (including by default)")
                print(f"Page content:\n{doc.page_content}\n{'-'*40}")
                context_parts.append(doc.page_content)
                continue

            if numeric_score >= score:
                print(f"Result score: {numeric_score}")
                print(f"Page content:\n{doc.page_content}\n{'-'*40}")
                context_parts.append(doc.page_content)
            else:
                print(f"Skipped chunk with score {numeric_score} below threshold {score}")

        print("=" * 100)
        return "\n\n".join(context_parts) # will join all chunks ion one string with `\n\n` separator between chunks

    def augment_prompt(self, query: str, context: str) -> str:
        print(f"\n🔗 STEP 2: AUGMENTATION\n{'-' * 100}")

        augmented_prompt = USER_PROMPT.format(context=context, query=query)

        print(f"{augmented_prompt}\n{'=' * 100}")
        return augmented_prompt

    def generate_answer(self, augmented_prompt: str) -> str:
        print(f"\n🤖 STEP 3: GENERATION\n{'-' * 100}")

        # 1. Create messages
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=augmented_prompt),
        ]

        # 2. Invoke llm client
        response = None
        # Prefer using the ChatModel `.generate()` API which expects a batch (list of message lists)
        if hasattr(self.llm_client, "generate"):
            try:
                response = self.llm_client.generate(messages=[messages])
            except Exception as e:
                print(f"⚠️ Error while calling .generate(): {e}")
                raise
        elif callable(self.llm_client):
            # Fallback for older/other clients that implement __call__
            response = self.llm_client(messages=messages)
        else:
            raise TypeError(
                "LLM client is not callable and does not implement 'generate'. "
                "Use a ChatModel-compatible client or update the call in generate_answer()."
            )

        # 3. Try to extract content from probable response structures
        content = None
        try:
            # LangChain ChatModel.generate returns batched `generations` structure
            content = response.generations[0][0].text
        except Exception:
            try:
                content = response.content
            except Exception:
                try:
                    content = response.choices[0].message["content"]
                except Exception:
                    content = str(response)

        # 4. Print and return
        print(f"Response:\n{content}")
        return content


def main(rag: MicrowaveRAG):
    print("🎯 Microwave RAG Assistant")

    while True:
        user_question = input("\n> ").strip()
        if not user_question:
            continue
        if user_question.lower() in ("exit", "quit"):
            print("👋 Goodbye!")
            break

        # Step 1: Retrieval of context
        context = rag.retrieve_context(user_question)
        # Step 2: Augmentation
        augmented = rag.augment_prompt(user_question, context)
        # Step 3: Generation
        answer = rag.generate_answer(augmented)
        print(f"\n\n{answer}")



main(
    MicrowaveRAG(
        # Embeddings client
        embeddings=AzureOpenAIEmbeddings(
            deployment="text-embedding-3-small-1",
            azure_endpoint=DIAL_URL,
            api_key=SecretStr(API_KEY),
        ),
        # Chat LLM client
        llm_client=AzureChatOpenAI(
            temperature=0.0,
            azure_deployment="gpt-4o",
            azure_endpoint=DIAL_URL,
            api_key=SecretStr(API_KEY),
            api_version="",
        ),
    )
)