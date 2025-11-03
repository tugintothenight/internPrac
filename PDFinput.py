from pathlib import Path
from pypdf import PdfReader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document


def read_pdf_text(file_path: str):
    pages = []
    with open(file_path, "rb") as f:
        pdf = PdfReader(f)
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages.append(text.strip())
    return pages


def chunk_text(text, chunk_size=300, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def process_pdf_and_update_faiss(file_path: str, index_path="faiss_index"):
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    file = Path(file_path)
    print(f"Đang xử lý file: {file.name}")
    docs = []

    pages = read_pdf_text(str(file))
    for page_num, page in enumerate(pages, start=1):
        if not page.strip():
            continue
        for chunk in chunk_text(page, chunk_size=300, overlap=50):
            metadata = {"source": file.name, "page": page_num}
            docs.append(Document(page_content=chunk, metadata=metadata))

    if not docs:
        print("File không có nội dung hợp lệ.")
        return


    index_dir = Path(index_path)
    index_dir.mkdir(parents=True, exist_ok=True)

    faiss_file = index_dir / "index.faiss"
    pkl_file = index_dir / "index.pkl"

    if not faiss_file.exists() or not pkl_file.exists():
        print("Chưa có FAISS index → Tạo mới...")
        db = FAISS.from_documents(docs, emb)
    else:
        try:
            print("Đã có FAISS index → Nạp và cập nhật thêm tài liệu...")
            db = FAISS.load_local(str(index_path), emb, allow_dangerous_deserialization=True)
            db.add_documents(docs)
        except Exception as e:
            print(f"Lỗi khi nạp index cũ ({e}), tạo mới lại...")
            db = FAISS.from_documents(docs, emb)

    db.save_local(str(index_path))
    print(f"Hoàn tất: {len(docs)} đoạn văn đã được lưu vào '{index_path}'.")
