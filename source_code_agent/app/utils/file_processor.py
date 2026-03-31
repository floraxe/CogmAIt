import os
import io
import logging
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
import csv
import json
import xml.etree.ElementTree as ET
import re
from datetime import datetime
import uuid
import pandas as pd
import uuid
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod
from magic_pdf.data.read_api import read_local_office, read_local_images

# 配置日志
logger = logging.getLogger(__name__)


async def analyze_image_with_model(image_path: str, model_id: str, db=None) -> Dict[str, Any]:
    """
    使用多模态大模型分析图片内容
    
    Args:
        image_path: 图片文件路径
        model_id: 多模态模型ID
        db: 数据库会话
    
    Returns:
        包含分析结果的字典
    """
    if not os.path.exists(image_path):
        logger.error(f"图片文件不存在: {image_path}")
        return {"error": "图片文件不存在"}
        
    try:
        # 导入需要的模块
        from sqlalchemy.orm import Session
        from app.models.model import Model
        from app.providers.manager import ProviderManager
        
        # 如果没有提供数据库会话，创建一个新的会话
        if db is None:
            from app.db.session import SessionLocal
            db = SessionLocal()
            need_close_db = True
        else:
            need_close_db = False
            
        # 查询模型信息
        model = db.query(Model).filter(Model.id == model_id).first()
        if not model:
            logger.error(f"找不到指定的模型: {model_id}")
            return {"error": "找不到指定的模型"}
            
        if not model.vision_support:
            logger.error(f"指定的模型不支持图片识别: {model.name}")
            return {"error": "指定的模型不支持图片识别"}
            
        # 获取模型提供商
        provider_manager = ProviderManager()
        provider = provider_manager.get_provider(model.provider)
        
        if provider is None:
            logger.error(f"找不到提供商: {model.provider}")
            return {"error": f"找不到提供商: {model.provider}"}
            
        # 调用提供商的图片分析API
        result = await provider.image_analysis(
            api_key=model.api_key,
            image_path=image_path,
            prompt="请详细描述这张图片的内容，包括图片中可能包含的文本、表格、图表和其他信息。",
            model=model.name,  # 所有提供商都使用model.name
            base_url=model.base_url
        )
        
        logger.info(f"图片分析完成: {image_path}")
        
        # 如果需要关闭数据库会话
        if need_close_db:
            db.close()
            
        return {
            "status": "success",
            "model": model.name,
            "provider": model.provider,
            "analysis": result.get("analysis", ""),
            "response_time_ms": result.get("response_time_ms")
        }
        
    except Exception as e:
        logger.error(f"分析图片时出错: {str(e)}")
        return {"error": f"分析图片时出错: {str(e)}"}


def process_file(file_path: str, output_dir: str = None,filename_uuid:str=None,knowledge_id:str=None, vision_model_id: str = None) -> Dict:
    """
    使用MinerU API处理文件，提取内容和元数据。
    
    支持PDF、Office文档(doc, docx, ppt, pptx, xls, xlsx)和图片文件(jpg, jpeg, png等)。
    
    Args:
        file_path: 要处理的文件路径
        output_dir: 输出目录，如果未提供则在当前目录创建
        filename_uuid: 文件UUID
        knowledge_id: 知识库ID
        vision_model_id: 多模态模型ID，用于图片分析
        
    Returns:
        包含处理结果的字典，包括文件路径、提取的文本、元数据等
    """
    try:
        logger.info(f"开始处理文件: {file_path}")
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            return {"error": "文件不存在"}
        
        # 获取文件名和扩展名
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_name)[1].lower()
        name_without_ext = filename_uuid
        # 创建输出目录，加入唯一标识符避免覆盖
        uid = str(uuid.uuid4().hex)
        print("output_dir::",output_dir)

        output_dir = os.path.join("uploads","processd",knowledge_id,filename_uuid )
        output_dir_images = os.path.join("uploads","processd",knowledge_id,filename_uuid,"images")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir_images, exist_ok=True)
        
        image_writer, md_writer = FileBasedDataWriter(output_dir_images), FileBasedDataWriter(
            output_dir
        )
        
        logger.info(f"输出目录: {output_dir}")
        
        # 初始化结果字典
        result = {
            "file_path": file_path,
            "file_name": file_name,
            "file_ext": file_ext,
            "output_dir": output_dir,
            "process_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        # 根据文件类型进行处理
        if file_ext in ['.pdf']:
            try:
                logger.info("处理PDF文件")
                # 创建数据读取器和写入器
                reader = FileBasedDataReader()
                writer = FileBasedDataWriter()
                
                # 读取PDF内容
                pdf_bytes = reader.read(file_path)
                
                # proc
                ## Create Dataset Instance
                ds = PymuDocDataset(pdf_bytes)
                # 使用MinerU API进行文档分析
                # doc = doc_analyze(
                #     doc,
                #     parse_method=SupportedPdfParseMethod.READABLE_PDF_FIRST,
                #     doc_type_classification=True,
                #     pdf_element_extraction=True,
                #     doc_section_extraction=True,
                #     doc_topic_extraction=True,
                #     doc_summary=True,
                #     doc_qa=False,
                #     page_processing_threads=1
                # )
                
                ## inference
                if ds.classify() == SupportedPdfParseMethod.OCR:
                    infer_result = ds.apply(doc_analyze, ocr=True)

                    ## pipeline
                    pipe_result = infer_result.pipe_ocr_mode(image_writer)

                else:
                    infer_result = ds.apply(doc_analyze, ocr=False)

                    ## pipeline
                    pipe_result = infer_result.pipe_txt_mode(image_writer)
                # 生成文档摘要和内容
                # markdown_content = doc.to_markdown()
                # json_content = doc.to_json()
                
                ### draw model result on each page
                infer_result.draw_model(os.path.join(output_dir, f"{name_without_ext}_visual.pdf"))

                ### get model inference result
                model_inference_result = infer_result.get_infer_res()
                # 保存处理结果
                md_output_path = os.path.join(output_dir, f"{name_without_ext}.md")
                json_output_path = os.path.join(output_dir, f"{name_without_ext}.json")
                

                ### dump markdown
                pipe_result.dump_md(md_writer, f"{name_without_ext}.md", output_dir)
                ### get markdown content
                md_content = pipe_result.get_markdown(output_dir)
                print(md_content)
                
                # 保存可视化结果
                # writer.write_pdf_visual_result(doc, os.path.join(output_dir, f"{name_without_ext}_visual.pdf"))
                
                # 更新结果
                result.update({
                    "markdown_path": md_output_path,
                    "visual_path": os.path.join(output_dir, f"{name_without_ext}_visual.pdf"),
                })
                
                # 如果启用了多模态模型，处理PDF中提取的图片
                if vision_model_id and os.path.exists(output_dir_images):
                    image_analyses = []
                    image_updates = {}  # 存储图片名和对应的分析结果
                    
                    logger.info("正在分析PDF文档中的图片...")
                    for img_file in os.listdir(output_dir_images):
                        if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')):
                            img_path = os.path.join(output_dir_images, img_file)
                            try:
                                # 异步调用图片分析函数
                                import asyncio
                                analysis_result = asyncio.run(analyze_image_with_model(img_path, vision_model_id))
                                if "error" not in analysis_result:
                                    analysis_data = {
                                        "image_name": img_file,
                                        "analysis": analysis_result.get("analysis", ""),
                                        "model": analysis_result.get("model", ""),
                                        "provider": analysis_result.get("provider", "")
                                    }
                                    image_analyses.append(analysis_data)
                                    
                                    # 保存图片分析结果用于更新Markdown
                                    image_updates[img_file] = analysis_data["analysis"]
                                    
                                    logger.info(f"完成图片分析: {img_file}")
                            except Exception as e:
                                logger.error(f"处理PDF中的图片时出错: {str(e)}")
                    
                    if image_analyses:
                        result["image_analyses"] = image_analyses
                        result["vision_model_id"] = vision_model_id
                        
                        # 获取模型信息
                        try:
                            from sqlalchemy.orm import Session
                            from app.models.model import Model
                            from app.db.session import SessionLocal
                            
                            db = SessionLocal()
                            model = db.query(Model).filter(Model.id == vision_model_id).first()
                            if model:
                                result["vision_model_name"] = model.name
                                result["vision_model_provider"] = model.provider
                            db.close()
                        except Exception as e:
                            logger.error(f"获取模型信息时出错: {str(e)}")
                        
                        # 更新Markdown文件，添加图片分析结果作为标题
                        if image_updates:
                            # 读取原始Markdown内容
                            with open(md_output_path, "r", encoding="utf-8") as md_file:
                                md_content = md_file.read()
                            
                            # 更新Markdown中的图片标记，添加分析结果
                            for img_name, analysis in image_updates.items():
                                # 查找图片引用并添加分析结果
                                # 尝试各种常见的图片引用格式
                                patterns = [
                                    f"!\\[\\]\\(images/{img_name}\\)",
                                    f"!\\[\\]\\([^)]*{img_name}\\)",
                                    f"!\\[[^\\]]*\\]\\([^)]*{img_name}\\)"
                                ]
                                
                                # 检查是否已经添加了分析结果
                                analysis_marker = f"> **图片分析**:"
                                if f"{analysis_marker} {analysis[:20]}" in md_content:
                                    logger.info(f"图片 {img_name} 的分析结果已存在，跳过")
                                    continue
                                
                                for pattern in patterns:
                                    import re
                                    # 检查这个图片是否已经添加过分析
                                    if re.search(pattern + r"\s*\n\n>\s*\*\*图片分析\*\*:", md_content):
                                        logger.info(f"图片 {img_name} 已有分析结果标记，跳过替换")
                                        break
                                        
                                    md_content = re.sub(
                                        pattern,
                                        f"![](images/{img_name})\n\n> **图片分析**: {analysis}\n",
                                        md_content,
                                        count=1  # 只替换第一次出现
                                    )
                            
                            # 保存更新后的Markdown内容
                            logger.info(f"更新Markdown文件，添加图片分析结果: {md_output_path}")
                            with open(md_output_path, "w", encoding="utf-8") as md_file:
                                md_file.write(md_content)
                
            except Exception as e:
                logger.error(f"处理PDF时出错: {str(e)}")
                result["error"] = f"处理PDF时出错: {str(e)}"
        
        # 处理Office文档
        elif file_ext in ['.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx']:
            try:
                logger.info(f"处理Office文档: {file_ext}")
                
                # 使用MinerU API读取Office文件
                doc = read_local_office(file_path)
                
                # 生成文档摘要和内容
                markdown_content = doc.to_markdown()
                json_content = doc.to_json()
                
                # 保存处理结果
                md_output_path = os.path.join(output_dir, f"{name_without_ext}.md")
                json_output_path = os.path.join(output_dir, f"{name_without_ext}.json")
                
                with open(md_output_path, "w", encoding="utf-8") as md_file:
                    md_file.write(markdown_content)
                
                with open(json_output_path, "w", encoding="utf-8") as json_file:
                    json_file.write(json_content)
                
                # 更新结果
                result.update({
                    "markdown_path": md_output_path,
                    "json_path": json_output_path,
                    "content_type": "office",
                    "office_type": file_ext[1:],
                })
                
                # 处理Office文档中的图片
                # 检查是否有图片目录
                office_images_dir = os.path.join(output_dir_images)
                if vision_model_id and os.path.exists(office_images_dir):
                    image_analyses = []
                    image_updates = {}
                    
                    logger.info("正在分析Office文档中的图片...")
                    for img_file in os.listdir(office_images_dir):
                        if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')):
                            img_path = os.path.join(office_images_dir, img_file)
                            try:
                                # 异步调用图片分析函数
                                import asyncio
                                analysis_result = asyncio.run(analyze_image_with_model(img_path, vision_model_id))
                                if "error" not in analysis_result:
                                    analysis_data = {
                                        "image_name": img_file,
                                        "analysis": analysis_result.get("analysis", ""),
                                        "model": analysis_result.get("model", ""),
                                        "provider": analysis_result.get("provider", "")
                                    }
                                    image_analyses.append(analysis_data)
                                    
                                    # 保存图片分析结果用于更新Markdown
                                    image_updates[img_file] = analysis_data["analysis"]
                                    
                                    logger.info(f"完成图片分析: {img_file}")
                            except Exception as e:
                                logger.error(f"处理Office文档中的图片时出错: {str(e)}")
                    
                    if image_analyses:
                        result["image_analyses"] = image_analyses
                        result["vision_model_id"] = vision_model_id
                        
                        # 获取模型信息
                        try:
                            from sqlalchemy.orm import Session
                            from app.models.model import Model
                            from app.db.session import SessionLocal
                            
                            db = SessionLocal()
                            model = db.query(Model).filter(Model.id == vision_model_id).first()
                            if model:
                                result["vision_model_name"] = model.name
                                result["vision_model_provider"] = model.provider
                            db.close()
                        except Exception as e:
                            logger.error(f"获取模型信息时出错: {str(e)}")
                        
                        # 更新Markdown文件，添加图片分析结果作为标题
                        if image_updates:
                            # 读取原始Markdown内容
                            with open(md_output_path, "r", encoding="utf-8") as md_file:
                                md_content = md_file.read()
                            
                            # 更新Markdown中的图片标记，添加分析结果
                            for img_name, analysis in image_updates.items():
                                # 查找图片引用并添加分析结果
                                # 尝试各种常见的图片引用格式
                                patterns = [
                                    f"!\\[\\]\\(images/{img_name}\\)",
                                    f"!\\[\\]\\([^)]*{img_name}\\)",
                                    f"!\\[[^\\]]*\\]\\([^)]*{img_name}\\)"
                                ]
                                
                                # 检查是否已经添加了分析结果
                                analysis_marker = f"> **图片分析**:"
                                if f"{analysis_marker} {analysis[:20]}" in md_content:
                                    logger.info(f"图片 {img_name} 的分析结果已存在，跳过")
                                    continue
                                
                                for pattern in patterns:
                                    import re
                                    # 检查这个图片是否已经添加过分析
                                    if re.search(pattern + r"\s*\n\n>\s*\*\*图片分析\*\*:", md_content):
                                        logger.info(f"图片 {img_name} 已有分析结果标记，跳过替换")
                                        break
                                        
                                    md_content = re.sub(
                                        pattern,
                                        f"![](images/{img_name})\n\n> **图片分析**: {analysis}\n",
                                        md_content,
                                        count=1  # 只替换第一次出现
                                    )
                            
                            # 保存更新后的Markdown内容
                            logger.info(f"更新Markdown文件，添加图片分析结果: {md_output_path}")
                            with open(md_output_path, "w", encoding="utf-8") as md_file:
                                md_file.write(md_content)
                
            except Exception as e:
                logger.error(f"处理Office文档时出错: {str(e)}")
                result["error"] = f"处理Office文档时出错: {str(e)}"
        
        # 处理图片文件
        elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp']:
            try:
                logger.info(f"处理图片文件: {file_ext}")
                
                # 使用MinerU API读取图片文件
                doc = read_local_images([file_path])
                
                # 生成文档摘要和内容
                markdown_content = doc.to_markdown()
                json_content = doc.to_json()
                
                # 保存处理结果
                md_output_path = os.path.join(output_dir, f"{name_without_ext}.md")
                json_output_path = os.path.join(output_dir, f"{name_without_ext}.json")
                
                with open(md_output_path, "w", encoding="utf-8") as md_file:
                    md_file.write(markdown_content)
                
                with open(json_output_path, "w", encoding="utf-8") as json_file:
                    json_file.write(json_content)
                
                # 更新结果
                result.update({
                    "markdown_path": md_output_path,
                    "json_path": json_output_path,
                    "content_type": "image",
                    "image_type": file_ext[1:],
                })
                
                # 如果启用了多模态模型，直接分析该图片
                if vision_model_id:
                    try:
                        # 异步调用图片分析函数
                        import asyncio
                        analysis_result = asyncio.run(analyze_image_with_model(file_path, vision_model_id))
                        if "error" not in analysis_result:
                            analysis_data = {
                                "image_name": os.path.basename(file_path),
                                "analysis": analysis_result.get("analysis", ""),
                                "model": analysis_result.get("model", ""),
                                "provider": analysis_result.get("provider", "")
                            }
                            result["image_analysis"] = {
                                "analysis": analysis_result.get("analysis", ""),
                                "model": analysis_result.get("model", ""),
                                "provider": analysis_result.get("provider", "")
                            }
                            
                            # 为了保持格式一致，也添加image_analyses字段
                            result["image_analyses"] = [analysis_data]
                            result["vision_model_id"] = vision_model_id
                            
                            # 获取模型信息
                            try:
                                from sqlalchemy.orm import Session
                                from app.models.model import Model
                                from app.db.session import SessionLocal
                                
                                db = SessionLocal()
                                model = db.query(Model).filter(Model.id == vision_model_id).first()
                                if model:
                                    result["vision_model_name"] = model.name
                                    result["vision_model_provider"] = model.provider
                                db.close()
                            except Exception as e:
                                logger.error(f"获取模型信息时出错: {str(e)}")
                            
                            # 获取分析内容并更新Markdown文件
                            analysis = analysis_data["analysis"]
                            short_analysis = analysis[:100] + ("..." if len(analysis) > 100 else "")
                            
                            # 读取原始Markdown内容
                            with open(md_output_path, "r", encoding="utf-8") as md_file:
                                md_content = md_file.read()
                            
                            # 修改Markdown内容，添加图片分析结果
                            enhanced_md_content = f"""# 图片内容

![](images/{os.path.basename(file_path)})

## 图片分析结果

**分析模型**: {analysis_data["model"]} ({analysis_data["provider"]})

{analysis}

---

{md_content}
"""
                            
                            # 保存更新后的Markdown内容
                            logger.info(f"更新单图片Markdown文件，添加图片分析结果: {md_output_path}")
                            with open(md_output_path, "w", encoding="utf-8") as md_file:
                                md_file.write(enhanced_md_content)
                                
                    except Exception as e:
                        logger.error(f"分析图片时出错: {str(e)}")
                
            except Exception as e:
                logger.error(f"处理图片文件时出错: {str(e)}")
                result["error"] = f"处理图片文件时出错: {str(e)}"
        
        # 对于不支持的文件类型，尝试使用传统处理器提取文本
        else:
            try:
                logger.info(f"不支持的文件类型: {file_ext}，尝试传统提取")
                
                # 尝试提取文本
                text_content = "无法提取文本内容"
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text_content = f.read()
                except:
                    logger.warning(f"无法以文本模式读取文件: {file_path}")
                
                # 创建简单的Markdown内容
                markdown_content = f"""# {file_name}

                                ## 文件信息
                                - 文件名: {file_name}
                                - 文件类型: {file_ext}
                                - 处理时间: {result['process_time']}

                                ## 文件内容
                                ```
                                {text_content[:1000]}
                                {'...' if len(text_content) > 1000 else ''}
                                ```
                                """
                
                # 保存处理结果
                md_output_path = os.path.join(output_dir, f"{name_without_ext}.md")
                
                with open(md_output_path, "w", encoding="utf-8") as md_file:
                    md_file.write(markdown_content)
                
                # 更新结果
                result.update({
                    "markdown_path": md_output_path,
                    "content_type": "unknown",
                    "file_type": file_ext[1:] if file_ext.startswith('.') else file_ext,
                })
                
            except Exception as e:
                logger.error(f"处理未知文件类型时出错: {str(e)}")
                result["error"] = f"处理未知文件类型时出错: {str(e)}"
        
        logger.info(f"文件处理完成: {file_path}")
        return result
        
    except Exception as e:
        logger.error(f"处理文件时发生错误: {str(e)}")
        return {"error": f"处理文件时发生错误: {str(e)}", "file_path": file_path}


async def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    将文本分成重叠的块
    
    参数:
        text (str): 要分块的文本
        chunk_size (int): 每个块的最大大小
        overlap (int): 块之间的重叠大小
        
    返回:
        List[str]: 文本块列表
    """
    if not text:
        return []
    
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        
        # 如果不是最后一块，尝试在一个完整的句子或段落结束时分割
        if end < text_length:
            # 尝试在段落处分割
            paragraph_end = text.rfind('\n\n', start, end)
            if paragraph_end != -1 and paragraph_end > start + chunk_size // 2:
                end = paragraph_end + 2  # 包含换行符
            else:
                # 尝试在句子处分割
                sentence_end = max(
                    text.rfind('. ', start, end),
                    text.rfind('? ', start, end),
                    text.rfind('! ', start, end),
                    text.rfind('.\n', start, end)
                )
                if sentence_end != -1 and sentence_end > start + chunk_size // 2:
                    end = sentence_end + 2  # 包含句号和空格
        
        chunks.append(text[start:end])
        
        # 更新起始位置，考虑重叠
        start = max(start + chunk_size - overlap, end - overlap)
    
    return chunks 

async def extract_text_from_file(file_path: str, file_type: str) -> str:
    """
    从文件中提取文本内容，用于向量化处理
    
    Args:
        file_path: 文件路径
        file_type: 文件类型
        
    Returns:
        提取的文本内容
    """
    try:
        logger.info(f"从文件中提取文本: {file_path}, 类型: {file_type}")
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            return "文件不存在"
        
        # 获取文件扩展名
        file_ext = f".{file_type}" if not file_type.startswith(".") else file_type
        file_ext = file_ext.lower()
        
        # 根据文件类型处理
        if file_ext in ['.pdf']:
            try:
                # 创建数据读取器
                reader = FileBasedDataReader()
                
                # 读取PDF内容
                pdf_bytes = reader.read(file_path)
                
                # 创建数据集实例
                ds = PymuDocDataset(pdf_bytes)
                
                # 根据PDF类型选择处理方法
                if ds.classify() == SupportedPdfParseMethod.OCR:
                    infer_result = ds.apply(doc_analyze, ocr=True)
                    
                    # 创建临时目录用于处理
                    temp_dir = os.path.join(os.getcwd(), "uploads", "temp", str(uuid.uuid4().hex))
                    os.makedirs(temp_dir, exist_ok=True)
                    
                    # 创建图片写入器
                    image_writer = FileBasedDataWriter(temp_dir)
                    
                    # 处理OCR模式
                    pipe_result = infer_result.pipe_ocr_mode(image_writer)
                    
                    # 获取Markdown内容
                    md_content = pipe_result.get_markdown(temp_dir)
                    
                    # 清理临时目录
                    try:
                        import shutil
                        shutil.rmtree(temp_dir)
                    except:
                        logger.warning(f"清理临时目录失败: {temp_dir}")
                    
                    return md_content
                else:
                    infer_result = ds.apply(doc_analyze, ocr=False)
                    
                    # 创建临时目录用于处理
                    temp_dir = os.path.join(os.getcwd(), "uploads", "temp", str(uuid.uuid4().hex))
                    os.makedirs(temp_dir, exist_ok=True)
                    
                    # 创建图片写入器
                    image_writer = FileBasedDataWriter(temp_dir)
                    
                    # 处理文本模式
                    pipe_result = infer_result.pipe_txt_mode(image_writer)
                    
                    # 获取Markdown内容
                    md_content = pipe_result.get_markdown(temp_dir)
                    
                    # 清理临时目录
                    try:
                        import shutil
                        shutil.rmtree(temp_dir)
                    except:
                        logger.warning(f"清理临时目录失败: {temp_dir}")
                    
                    return md_content
            except Exception as e:
                logger.error(f"提取PDF文本时出错: {str(e)}")
                return f"提取PDF文本时出错: {str(e)}"
        
        # 处理Office文档
        elif file_ext in ['.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx']:
            try:
                # 使用MinerU API读取Office文件
                doc = read_local_office(file_path)
                
                # 生成文档内容
                markdown_content = doc.to_markdown()
                return markdown_content
            except Exception as e:
                logger.error(f"提取Office文档文本时出错: {str(e)}")
                return f"提取Office文档文本时出错: {str(e)}"
        
        # 处理图片文件
        elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp']:
            try:
                # 使用MinerU API读取图片文件
                doc = read_local_images([file_path])
                
                # 生成文档内容
                markdown_content = doc.to_markdown()
                return markdown_content
            except Exception as e:
                logger.error(f"提取图片文本时出错: {str(e)}")
                return f"提取图片文本时出错: {str(e)}"
        
        # 处理文本文件
        elif file_ext in ['.txt', '.md', '.csv', '.json', '.xml', '.html', '.htm']:
            try:
                # 直接读取文本文件
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 根据文件类型进行特殊处理
                if file_ext == '.csv':
                    # 将CSV转换为可读文本
                    text_content = "CSV内容:\n\n"
                    with open(file_path, 'r', encoding='utf-8') as csv_file:
                        csv_reader = csv.reader(csv_file)
                        for row in csv_reader:
                            text_content += ", ".join(row) + "\n"
                    return text_content
                elif file_ext == '.json':
                    # 格式化JSON
                    try:
                        json_obj = json.loads(content)
                        return json.dumps(json_obj, ensure_ascii=False, indent=2)
                    except:
                        return content
                elif file_ext in ['.xml', '.html', '.htm']:
                    # 提取文本内容，去除标签
                    try:
                        text = re.sub(r'<[^>]+>', ' ', content)
                        text = re.sub(r'\s+', ' ', text).strip()
                        return text
                    except:
                        return content
                else:
                    # 其他文本文件直接返回内容
                    return content
            except UnicodeDecodeError:
                # 尝试其他编码
                try:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        return f.read()
                except Exception as e:
                    logger.error(f"读取文本文件时出错: {str(e)}")
                    return f"读取文本文件时出错: {str(e)}"
            except Exception as e:
                logger.error(f"读取文本文件时出错: {str(e)}")
                return f"读取文本文件时出错: {str(e)}"
        
        # 对于不支持的文件类型，尝试作为文本文件读取
        else:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except UnicodeDecodeError:
                try:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        return f.read()
                except Exception as e:
                    logger.error(f"读取未知类型文件时出错: {str(e)}")
                    return f"不支持的文件类型: {file_ext}，无法提取文本"
            except Exception as e:
                logger.error(f"读取未知类型文件时出错: {str(e)}")
                return f"不支持的文件类型: {file_ext}，无法提取文本"
    
    except Exception as e:
        logger.error(f"提取文件文本内容时出错: {str(e)}")
        return f"提取文件文本内容时出错: {str(e)}" 

async def extract_text_from_file_path(file_path: str) -> str:
    """
    从文件路径提取文本内容，自动推断文件类型
    
    Args:
        file_path: 文件路径
        
    Returns:
        提取的文本内容
    """
    try:
        logger.info(f"从文件路径提取文本: {file_path}")
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            return "文件不存在"
        
        # 获取文件扩展名
        file_ext = os.path.splitext(file_path)[1].lower()
        if not file_ext:
            # 如果没有扩展名，默认作为文本文件处理
            file_type = 'txt'
        else:
            # 去掉扩展名前面的点
            file_type = file_ext[1:]
        
        # 尝试直接读取文件内容
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # 尝试其他编码
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    return f.read()
            except Exception as e:
                logger.error(f"读取文件时出错: {str(e)}")
                return f"读取文件时出错: {str(e)}"
        except Exception as e:
            logger.error(f"读取文件时出错: {str(e)}")
            return f"读取文件时出错: {str(e)}"
    
    except Exception as e:
        logger.error(f"提取文件文本内容时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return f"提取文件文本内容时出错: {str(e)}" 