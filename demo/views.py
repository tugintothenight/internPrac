import os
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from django.core.files.storage import FileSystemStorage
from PDFinput import process_pdf_and_update_faiss
from pathlib import Path
import google.generativeai as genai
from langchain.llms.base import LLM
from typing import Optional, List


GEMINI_API_KEY = ""
genai.configure(api_key=GEMINI_API_KEY)


class GeminiLLM(LLM):
    model_name: str = "gemini-2.5-flash"

    @property
    def _llm_type(self) -> str:
        return "gemini-custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            response = genai.GenerativeModel(self.model_name).generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Lỗi khi gọi Gemini API: {e}"


template = """You are a virtual assistant specialized in reading and summarizing documents.
Below is the context extracted from the document:
{context}
Question: {question}
Provide a concise and accurate answer based solely on the above context.
"""


def load_retriever():
    from django.conf import settings
    index_path = os.path.join(settings.BASE_DIR, "faiss_index")

    if not Path(index_path).exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục FAISS index tại: {index_path}")

    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(index_path, emb, allow_dangerous_deserialization=True)
    return db.as_retriever(search_kwargs={"k": 3})


def up(request):
    if request.method == "POST" and "pdffile" in request.FILES:
        print('nhận file')
        file = request.FILES["pdffile"]
        fs = FileSystemStorage(location="media/uploads/")
        filename = fs.save(file.name, file)
        file_path = fs.path(filename)
        process_pdf_and_update_faiss(file_path)
    return render(request, "home.html")


@csrf_exempt
def query_view(request):
    answer = ""
    context_chunks = []

    if request.method == "POST" and request.POST.get("query"):
        q = request.POST.get("query")
        retriever = load_retriever()
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=template,
        )
        llm = GeminiLLM()
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
        )

        res = qa_chain({"query": q})
        answer = res.get("result", "")
        docs = res.get("source_documents", [])

        for d in docs:
            context_chunks.append({"text": d.page_content, "meta": d.metadata})

    return render(request, "query.html", {"answer": answer, "contexts": context_chunks})
