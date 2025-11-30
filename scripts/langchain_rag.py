import glob
import os
import base64
import shutil
import textwrap

from dotenv import load_dotenv # type: ignore

from langchain_community.document_loaders import TextLoader # type: ignore
from langchain_community.vectorstores import FAISS # type: ignore
from langchain_text_splitters import MarkdownTextSplitter # type: ignore

from langchain_openai import OpenAIEmbeddings, ChatOpenAI # type: ignore

from langchain_core.prompts import ChatPromptTemplate # type: ignore
from langchain_core.output_parsers import StrOutputParser # type: ignore
from langchain_core.runnables import RunnableParallel, RunnablePassthrough # type: ignore
from langchain_core.documents import Document # type: ignore
from langchain_core.messages import HumanMessage # type: ignore

# Load environment variables
load_dotenv()


# Printout the extracted text
def print_wrapped(text: str) -> None:
    """Print text wrapped to terminal width, preserving existing line breaks."""
    width = shutil.get_terminal_size((80, 20)).columns
    for line in text.split("\n"):
        if line.strip() == "":
            # Preserve blank lines
            print()
        else:
            print(textwrap.fill(line, width=width))


def format_docs(docs):
    """Turn a list of Documents into a single context string."""
    return "\n\n---\n\n".join(d.page_content for d in docs)


class RegulatoryRAG:
    """RAG helper that embeds regulatory documents and answers 
    user questions with GPT-4o and FAISS."""
    def __init__(self):
        # LLM for answering questions
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)

        # Embeddings (FAISS persists to disk)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

        # Prompt for question answering
        self.prompt = ChatPromptTemplate.from_template(
            """
You are an expert in regulatory compliance. Answer the following question using the provided regulatory context.

- Be precise and concise.
- Cite specific requirements or clauses when possible.
- If the context is insufficient, say so clearly and state what else would be needed.

Context:
{context}

Question:
{input}

Answer:
"""
        )

        self.vectorstore = None
        self.retrieval_chain = None

    def load_multiple_documents(self, file_paths):
        """Load and process multiple markdown documents (text-only RAG)."""
        all_documents = []

        for file_path in file_paths:
            print(f"Loading document: {file_path}")
            loader = TextLoader(file_path, encoding="utf-8")
            documents = loader.load()

            # Add source metadata for cross-referencing
            for doc in documents:
                doc.metadata["source_file"] = os.path.basename(file_path)
                doc.metadata["source_path"] = file_path
                doc.metadata["content_type"] = "text"

            all_documents.extend(documents)

        # Split all documents together
        text_splitter = MarkdownTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_documents(all_documents)
        print(f"Created {len(chunks)} chunks from {len(file_paths)} documents")

        # Create unified vector store
        print("Creating embeddings for all documents (this may take a moment)...")
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)

        # Save vector store locally
        os.makedirs("embeddings_cache/vectorstore", exist_ok=True)
        self.vectorstore.save_local("embeddings_cache/vectorstore")
        print("✅ All documents processed and embeddings saved locally")

        # Create retrieval chain
        self._create_retrieval_chain()

    def load_images_with_vision(self, markdown_files, image_files):
        """
        Load markdown documents and process images using an OpenAI Vision-capable model.
        Combines text + AI-generated image descriptions into one vector store.
        """
        all_documents = []

        # 1) Load markdown documents
        for file_path in markdown_files:
            print(f"Loading document: {file_path}")
            loader = TextLoader(file_path, encoding="utf-8")
            documents = loader.load()

            for doc in documents:
                doc.metadata["source_file"] = os.path.basename(file_path)
                doc.metadata["source_path"] = file_path
                doc.metadata["content_type"] = "text"

            all_documents.extend(documents)

        # 2) Process images with GPT-4o (vision)
        if image_files:
            print("Processing images with GPT-4o (vision)...")

        vision_llm = ChatOpenAI(model="gpt-4o", max_tokens=1000, temperature=0)

        for image_path in image_files:
            print(f"Analyzing image with GPT-4o Vision: {image_path}")
            try:
                with open(image_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode("utf-8")

                message = HumanMessage(
                    content=[
                        {
                            "type": "text",
                            "text": (
                                "Analyze this technical diagram from a medical device "
                                "security standard.\n\n"
                                "Provide a detailed description including:\n"
                                "1. Process flow and sequence\n"
                                "2. All components and their relationships\n"
                                "3. Mathematical formulas or calculations shown\n"
                                "4. Technical terminology and definitions\n"
                                "5. How this relates to security risk management\n\n"
                                "Format your response as a comprehensive technical "
                                "description suitable for regulatory documentation."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            },
                        },
                    ]
                )

                response = vision_llm.invoke([message])

                image_name = os.path.basename(image_path)
                image_doc = Document(
                    page_content=f"Figure Analysis: {image_name}\n\n{response.content}",
                    metadata={
                        "source_file": image_name,
                        "source_path": image_path,
                        "content_type": "ai_image_analysis",
                        "has_visual_content": True,
                        "analysis_model": "gpt-4o-vision",
                    },
                )

                all_documents.append(image_doc)
                print(f"✅ AI analysis completed for {image_name}")

            except Exception as e:
                print(f"⚠️ Could not analyze image {image_path}: {str(e)}")

        # 3) Split combined text + image-derived docs
        text_splitter = MarkdownTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_documents(all_documents)
        print(
            f"Created {len(chunks)} chunks from "
            f"{len(markdown_files)} documents and {len(image_files)} images"
        )

        # 4) Create unified vector store
        print("Creating embeddings...")
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)

        os.makedirs("embeddings_cache/vectorstore", exist_ok=True)
        self.vectorstore.save_local("embeddings_cache/vectorstore")

        # 5) Build retrieval chain
        self._create_retrieval_chain()

        print("✅ Documents and AI-analyzed images processed successfully")

    def load_existing_vectorstore(self):
        """Load previously created vector store if available."""
        try:
            self.vectorstore = FAISS.load_local(
                "embeddings_cache/vectorstore",
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            print("✅ Existing vector store loaded")
            self._create_retrieval_chain()
            return True
        except Exception as e:
            print(f"❌ No existing vector store found or failed to load: {e}")
            return False

    def _create_retrieval_chain(self):
        """Create the retrieval chain for Q&A using runnables (LangChain 1.x style)."""
        if self.vectorstore is None:
            raise ValueError("Vectorstore is not initialized.")

        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5},
        )

        # Step 1: get docs and keep the original question
        retrieval = RunnableParallel(
            {
                "docs": retriever,           # retriever(question) -> list[Document]
                "input": RunnablePassthrough(),  # keep original question
            }
        )

        # Step 2: format docs into context string and pass input unchanged
        format_context = retrieval | {
            "context": lambda x: format_docs(x["docs"]),
            "input": lambda x: x["input"],
        }

        # Step 3: prompt -> LLM -> string
        document_chain = self.prompt | self.llm | StrOutputParser()

        # Full chain
        self.retrieval_chain = format_context | document_chain

    def query(self, question: str) -> str:
        """Query the regulatory documents."""
        if not self.retrieval_chain:
            return "❌ No documents loaded. Please load a document first."

        try:
            response = self.retrieval_chain.invoke(question)
            return response
        except Exception as e:
            return f"❌ Error processing query: {str(e)}"


def main():
    print("🏛️ LangChain Regulatory RAG System")
    print("=" * 50)

    os.makedirs("embeddings_cache/vectorstore", exist_ok=True)

    rag = RegulatoryRAG()

    # Try to load the embeddings cache first
    if not rag.load_existing_vectorstore():
        doc_folder = "documents/"
        img_folder = "images/"

        md_files = glob.glob(os.path.join(doc_folder, "*.md"))

        image_files = []
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            image_files.extend(glob.glob(os.path.join(img_folder, ext)))

        print(
            f"Found {len(md_files)} markdown documents "
            f"and {len(image_files)} images."
        )

        if md_files or image_files:
            rag.load_images_with_vision(md_files, image_files)
        else:
            print("⚠️ No documents or images found. Please add files and rerun.")
    else:
        print("Loaded existing embeddings. No need to recreate!")

    print("\n" + "=" * 70)
    print(
        "🚀 System ready! Ask questions about documents and images in the "
        "FDA Cybersecurity Guidance."
    )
    print("Type 'exit' to quit\n")

    while True:
        try:
            question = (
                input(
                    "❓ Your question about the FDA Cybersecurity Guidance: "
                )
                .strip()
                .replace("\n", " ")
                .replace("\r", " ")
            )
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

        if question.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break

        if not question:
            continue

        print("\n🤔 Thinking...")
        answer = rag.query(question)

        print("\n💡 Answer:\n")
        print_wrapped(answer)
        print("\n" + "-" * 70)


if __name__ == "__main__":
    main()
