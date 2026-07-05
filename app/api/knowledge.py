"""
知识库管理接口模块

提供:
1. POST /knowledge/upload - 上传文档到知识库
2. GET /knowledge/documents - 列出知识库文档
3. GET /knowledge/stats - 知识库统计信息
4. DELETE /knowledge/documents/{filename} - 删除文档
5. DELETE /knowledge/all - 清空知识库
"""
import os
import shutil
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.rag import rag_engine
from app.models.schemas import DocumentIngestResponse, KnowledgeDocument, KnowledgeStats
from app.utils.logger import log
from config import settings

router = APIRouter(prefix="/knowledge", tags=["知识库"])

# 支持的文件类型
SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx", ".md"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB


@router.post("/upload", response_model=DocumentIngestResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    上传文档到知识库

    流程: 保存文件 → 加载 → 切分 → 向量化 → 存入Chroma
    """
    # 校验文件类型
    filename = file.filename or "unknown.txt"
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式: {ext}, 支持: {list(SUPPORTED_EXTENSIONS)}",
        )

    # 读取文件内容并校验大小
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="文件过大, 最大支持 20MB")

    # 保存到知识库目录
    file_path = settings.knowledge_base_dir / filename
    with open(file_path, "wb") as f:
        f.write(content)
    log.info(f"文件已保存: {file_path}")

    # RAG 入库
    try:
        result = rag_engine.ingest_file(str(file_path))
        return DocumentIngestResponse(
            file=result["file"],
            chunks=result["chunks"],
            total=result["total"],
            time=result["time"],
        )
    except Exception as e:
        log.error(f"文档入库失败: {e}")
        # 入库失败时删除已保存的文件
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"文档入库失败: {e}")


@router.get("/documents", response_model=list[KnowledgeDocument])
async def list_documents():
    """列出知识库中的文档"""
    return rag_engine.list_documents()


@router.get("/stats", response_model=KnowledgeStats)
async def knowledge_stats():
    """获取知识库统计信息"""
    documents = rag_engine.list_documents()
    total_chunks = rag_engine._count_chunks()
    return KnowledgeStats(
        total_chunks=total_chunks,
        total_files=len(documents),
        documents=[KnowledgeDocument(**doc) for doc in documents],
    )


@router.delete("/documents/{filename}")
async def delete_document(filename: str):
    """
    删除知识库文档

    注意: 仅删除源文件, 向量库中的已索引内容需通过清空操作移除。
    """
    file_path = settings.knowledge_base_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"文档不存在: {filename}")

    file_path.unlink()
    log.info(f"已删除文档: {filename}")
    return {"message": f"已删除文档: {filename}", "note": "向量库内容需手动清空重建"}


@router.delete("/all")
async def clear_knowledge_base():
    """清空整个知识库 (文档 + 向量数据)"""
    # 清空向量数据库
    deleted_chunks = rag_engine.delete_all()

    # 清空文档目录
    deleted_files = 0
    for f in settings.knowledge_base_dir.iterdir():
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
            f.unlink()
            deleted_files += 1

    log.warning(f"知识库已清空: 删除 {deleted_chunks} 个向量块, {deleted_files} 个文件")
    return {
        "deleted_chunks": deleted_chunks,
        "deleted_files": deleted_files,
        "message": "知识库已清空",
    }
