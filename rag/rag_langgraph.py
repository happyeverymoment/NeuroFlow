"""
基于 LangGraph 的 RAG 系统（参考 Milvus 官方文档）
使用 LangGraph 构建基于图的 RAG 流程，集成 Milvus 向量数据库
"""

import os
from typing import TypedDict, Annotated, Sequence

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    from langchain_community.chat_models import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

# 文件处理与 Milvus 导入逻辑在 faq_ingest 中
try:
    from .faq_ingest import (
        create_retriever_from_documents,
        get_retriever_from_existing_milvus,
        FAQ_FILE_PATH,
        FAQ_DATA_DIR,
    )
except ImportError:
    from faq_ingest import (
        create_retriever_from_documents,
        get_retriever_from_existing_milvus,
        FAQ_FILE_PATH,
        FAQ_DATA_DIR,
    )

# 尝试导入新的 create_agent，如果不存在则使用旧的 create_react_agent
try:
    from langchain.agents import create_agent
    CREATE_AGENT_FUNC = create_agent
except ImportError:
    try:
        from langgraph.prebuilt import create_react_agent
        CREATE_AGENT_FUNC = create_react_agent
    except ImportError:
        CREATE_AGENT_FUNC = None

# LLM 与 RAG 配置从统一配置加载（config.json）
try:
    from .config_loader import get_llm, get_rag
except ImportError:
    from config_loader import get_llm, get_rag

_llm_cfg = get_llm()
_rag_cfg = get_rag()
LLM_API_KEY = _llm_cfg["api_key"]
LLM_MODEL = _llm_cfg["model"]
LLM_BASE_URL = _llm_cfg["base_url"]
LLM_TEMPERATURE = _llm_cfg["temperature"]


class GraphState(TypedDict):
    """图状态定义"""
    messages: Annotated[Sequence[BaseMessage], add_messages]


def create_agent_node(llm, tools):
    """创建 Agent 节点"""
    def agent_node(state: GraphState):
        """Agent 节点处理函数"""
        messages = state["messages"]
        
        # 使用提示词方式引导 LLM 使用工具
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个 FAQ 问答助手，基于知识库中的答案进行总结和回答。

工作流程：
1. **必须**使用 retrieve_documents 工具检索与用户问题最相似的问题及其答案
2. 工具会返回"问题：xxx\n答案：xxx"格式的内容（可能带相似度说明）
3. **始终**基于检索到的内容进行润色后回复：即使相似度不是特别高，也请对相似度最高的那条进行归纳、润色后回答用户，并可在回答开头说明"以下为知识库中与您问题最相关的内容，供参考"
4. 如果答案中有步骤或列表，保持原有格式
5. 仅当工具明确返回"知识库中没有找到相关信息"（即完全没有检索结果）时，才回复用户"知识库中没有找到相关信息"

回答要求：
- 基于检索到的答案内容进行总结与润色，不要直接说"未找到"
- 保持答案的准确性和完整性，可适当优化表达
- 如果答案是操作步骤，保持步骤的清晰性

可用的工具：
- retrieve_documents: 检索与查询最相似的问题和答案。输入应该是用户的查询问题。
"""),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        # 尝试绑定工具
        llm_with_tools = None
        try:
            # 先尝试使用 bind_tools（新版本）
            llm_with_tools = llm.bind_tools(tools)
            print(f"✅ 使用 bind_tools 绑定工具 (工具数量: {len(tools)})")
            # 验证工具是否真的被绑定
            if hasattr(llm_with_tools, 'bound_tools'):
                print(f"   已绑定工具: {[t.name for t in llm_with_tools.bound_tools]}")
        except (NotImplementedError, AttributeError) as e:
            print(f"⚠️ bind_tools 失败: {e}，尝试其他方式...")
            # 如果不支持，尝试使用 bind（旧版本）
            try:
                # 将工具转换为函数格式
                from langchain_core.tools import tool
                # 尝试获取工具的 schema
                tool_schemas = []
                for tool_obj in tools:
                    if hasattr(tool_obj, 'args_schema') and tool_obj.args_schema:
                        tool_schemas.append(tool_obj.args_schema.schema())
                    elif hasattr(tool_obj, 'schema'):
                        tool_schemas.append(tool_obj.schema())
                    else:
                        # 手动构建简单的 schema
                        tool_schemas.append({
                            "type": "function",
                            "function": {
                                "name": tool_obj.name if hasattr(tool_obj, 'name') else "unknown",
                                "description": tool_obj.description if hasattr(tool_obj, 'description') else "",
                                "parameters": {"type": "object", "properties": {}}
                            }
                        })
                llm_with_tools = llm.bind(functions=tool_schemas)
                print(f"✅ 使用 bind(functions) 绑定工具 (工具数量: {len(tool_schemas)})")
            except Exception as e2:
                print(f"⚠️ bind(functions) 也失败: {e2}")
                print(f"   错误详情: {type(e2).__name__}: {str(e2)}")
                # 如果都不支持，直接使用 LLM，在提示词中说明工具
                llm_with_tools = llm
                print("⚠️ 无法绑定工具，LLM 可能不支持工具调用")
        
        if llm_with_tools is None:
            llm_with_tools = llm
        
        # 格式化消息
        formatted_messages = prompt.format_messages(messages=messages)
        
        # 调用 LLM
        print(f"📤 调用 LLM (消息数: {len(formatted_messages)})...")
        response = llm_with_tools.invoke(formatted_messages)
        
        # 检查是否有工具调用
        print(f"📥 LLM 响应类型: {type(response).__name__}")
        print(f"📥 LLM 响应属性: {dir(response)}")
        
        # 检查多种可能的工具调用属性
        tool_calls = None
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_calls = response.tool_calls
        elif hasattr(response, 'tool_calls') and getattr(response, 'tool_calls', None):
            tool_calls = getattr(response, 'tool_calls', None)
        elif hasattr(response, 'additional_kwargs') and 'tool_calls' in response.additional_kwargs:
            tool_calls = response.additional_kwargs.get('tool_calls')
        
        if tool_calls:
            print(f"🔧 Agent 调用了 {len(tool_calls)} 个工具")
            for i, tool_call in enumerate(tool_calls):
                if isinstance(tool_call, dict):
                    print(f"  工具 {i+1}: {tool_call.get('name', 'unknown')}, 参数: {tool_call.get('args', {})}")
                else:
                    print(f"  工具 {i+1}: {tool_call}")
        else:
            print("⚠️ Agent 没有调用工具，直接生成回答")
            print(f"   响应内容预览: {response.content[:200] if hasattr(response, 'content') else str(response)[:200]}...")
        
        return {"messages": [response]}
    
    return agent_node


def should_continue(state: GraphState):
    """判断是否继续"""
    messages = state["messages"]
    last_message = messages[-1]
    
    # 如果最后一条消息是 AIMessage 且有工具调用，继续执行工具
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        print(f"➡️ 检测到工具调用，继续执行工具节点...")
        return "tools"
    
    # 否则结束
    print("✅ 没有工具调用，流程结束")
    return END


def create_langgraph_rag(file_path=None, data_dir=None, *, skip_ingest: bool = True):
    """创建 LangGraph RAG 系统。
    file_path: 单个 FAQ Excel 路径（仅当 skip_ingest=False 时使用）。
    data_dir: rag_faqdata 目录路径（仅当 skip_ingest=False 时使用）；与 file_path 二选一。
    skip_ingest: 为 True 时只连已有 Milvus 集合，不读文件、不导入；为 False 时从文件/目录入库后再建检索器。
    默认 skip_ingest=True，适用于服务启动；需要重建库时传 skip_ingest=False 或在命令行执行 python -m rag.faq_ingest。
    """
    print("🚀 初始化 LangGraph RAG 系统...")
    
    # 1. 创建检索器（先建 retriever，若需入库则先入库），再取 vectorstore 供带分数检索
    if skip_ingest:
        retriever = get_retriever_from_existing_milvus()
    else:
        retriever = create_retriever_from_documents(data_dir=data_dir) if (data_dir is not None and os.path.isdir(data_dir)) else None
        if retriever is None:
            retriever = create_retriever_from_documents(file_path=file_path) if (file_path and os.path.isfile(file_path)) else None
        if retriever is None:
            retriever = create_retriever_from_documents(data_dir=FAQ_DATA_DIR) if os.path.isdir(FAQ_DATA_DIR) else create_retriever_from_documents(file_path=file_path or FAQ_FILE_PATH)
    try:
        from .faq_ingest import get_milvus_vectorstore
    except ImportError:
        from faq_ingest import get_milvus_vectorstore
    vectorstore = get_milvus_vectorstore()
    
    # 相似度阈值与候选数（统一配置文件 rag.similarity_threshold / candidates_k_default）
    similarity_threshold = float(_rag_cfg.get("similarity_threshold", 0.3))
    candidates_k = int(_rag_cfg.get("candidates_k_default", 5))
    
    # 2. 创建自定义检索工具：仅当存在相似度超过阈值的答案时，取相似度最高的一条返回给大模型润色
    from langchain_core.tools import tool
    
    def _format_doc(doc, similarity=None) -> str:
        """将单条文档格式化为 问题/答案 文本。"""
        question = doc.metadata.get("question", doc.page_content)
        answer = doc.metadata.get("answer", "") or doc.page_content or "（无正文）"
        head = ""
        if similarity is not None:
            head = f"（相似度：{similarity:.2%}，请基于此条润色后回答）\n"
        return f"{head}问题：{question}\n答案：{answer}"
    
    def retrieve_documents_impl(query: str) -> str:
        # 使用 vectorstore 带分数检索，与 rag_server 一致：相似度 = 1 - score（score 为距离）
        try:
            scored_list = vectorstore.similarity_search_with_score(query, k=candidates_k)
        except Exception as e:
            print(f"⚠️ similarity_search_with_score 失败: {e}")
            return "知识库中没有找到相关信息"
        if not scored_list:
            return "知识库中没有找到相关信息"
        # 过滤出相似度 >= 阈值的文档，取相似度最高的一条
        best_doc, best_sim = None, -1.0
        for doc, raw_score in scored_list:
            sim = 1.0 - float(raw_score)
            if sim >= similarity_threshold and sim > best_sim:
                best_sim = sim
                best_doc = doc
        if best_doc is None:
            print(f"🔍 无结果超过相似度阈值 {similarity_threshold:.0%}，返回未找到")
            return "知识库中没有找到相关信息"
        print(f"🔍 检索到最相似问题（相似度 {best_sim:.2%}）: {(best_doc.metadata.get('question') or best_doc.page_content)[:100]}...")
        return _format_doc(best_doc, similarity=best_sim)
    
    @tool
    def retrieve_documents(query: str) -> str:
        """
        从知识库中检索与查询最相似的问题，仅当相似度超过配置阈值时返回最高的一条供大模型润色，否则返回未找到。
        
        Args:
            query: 用户的查询问题
            
        Returns:
            最相似问题对应的答案（格式：问题：xxx\n答案：xxx），或「知识库中没有找到相关信息」
        """
        return retrieve_documents_impl(query)
    
    retriever_tool = retrieve_documents
    
    tools = [retriever_tool]
    
    # 3. 初始化 LLM（使用 DashScope 千问模型）
    print("🤖 初始化 LLM...")
    # 注意：确保 LLM 支持工具调用
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        model_kwargs={
            "tools": None  # 先不在这里设置，在 bind_tools 时设置
        }
    )
    print(f"   LLM 模型: {LLM_MODEL}")
    print(f"   支持工具调用: 检查中...")
    
    # 4. 强制使用手动构建方式（确保工具被正确调用）
    # 注意：create_agent 可能不会强制 LLM 使用工具，所以我们使用手动构建方式
    print("📊 使用手动构建方式（确保工具调用）...")
    
    # 5. 创建工具节点（带调试信息）
    def tool_node_with_debug(state: GraphState):
        """工具节点（带调试信息）"""
        messages = state["messages"]
        last_message = messages[-1]
        
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            print(f"🔧 执行工具调用: {len(last_message.tool_calls)} 个工具")
            for i, tool_call in enumerate(last_message.tool_calls):
                tool_name = tool_call.get('name', 'unknown')
                tool_args = tool_call.get('args', {})
                print(f"  工具 {i+1}: {tool_name}, 参数: {tool_args}")
        
        # 使用标准的 ToolNode
        tool_node = ToolNode(tools)
        result = tool_node.invoke(state)
        
        # 显示工具执行结果
        if result.get("messages"):
            for msg in result["messages"]:
                if hasattr(msg, 'content'):
                    content_preview = str(msg.content)[:200] if msg.content else ""
                    print(f"📄 工具返回内容预览: {content_preview}...")
        
        return result
    
    tool_node = tool_node_with_debug
    
    # 6. 创建 Agent 节点
    agent_node = create_agent_node(llm, tools)
    
    # 7. 构建图
    print("📊 手动构建 LangGraph...")
    workflow = StateGraph(GraphState)
    
    # 添加节点
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    
    # 设置入口点
    workflow.set_entry_point("agent")
    
    # 添加条件边
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END
        }
    )
    
    # 工具执行后返回 Agent
    workflow.add_edge("tools", "agent")
    
    # 编译图
    app = workflow.compile()
    
    print("✅ LangGraph RAG 系统创建成功")
    return app


def main():
    """主函数：测试 LangGraph RAG 系统"""
    # 创建 RAG 系统
    app = create_langgraph_rag()
    
    # 测试查询
    print("\n" + "="*60)
    print("测试 LangGraph RAG 系统")
    print("="*60)
    
    test_queries = [
        "如何查询商机？",
        "签约流程是什么？",
        "系统如何使用？",
        "你好是，晚上吃什么"
    ]
    
    for query in test_queries:
        print(f"\n❓ 问题: {query}")
        print("-" * 60)
        
        # 运行图
        config = {"recursion_limit": 50}
        result = app.invoke(
            {"messages": [HumanMessage(content=query)]},
            config=config
        )
        
        # 显示所有消息（用于调试）
        print(f"\n📨 消息流分析:")
        print(f"   总消息数: {len(result['messages'])}")
        
        tool_was_called = False
        retrieved_content = None
        
        for i, msg in enumerate(result["messages"]):
            msg_type = type(msg).__name__
            if isinstance(msg, AIMessage):
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    tool_was_called = True
                    print(f"  [{i}] 🤖 AI消息 (有工具调用): {len(msg.tool_calls)} 个工具调用")
                    for j, tc in enumerate(msg.tool_calls):
                        print(f"      工具 {j+1}: {tc.get('name', 'unknown')}")
                else:
                    print(f"  [{i}] 🤖 AI消息: {msg.content[:100] if msg.content else '(空)'}...")
            elif msg_type == "ToolMessage":
                tool_was_called = True
                content = str(msg.content) if hasattr(msg, 'content') else str(msg)
                retrieved_content = content
                print(f"  [{i}] 🔧 工具消息: {content[:200]}...")
            else:
                print(f"  [{i}] {msg_type}")
        
        # 提取最后一条 AI 消息
        last_message = result["messages"][-1]
        if isinstance(last_message, AIMessage):
            print(f"\n🤖 最终回答:")
            print(f"   {last_message.content}")
            
            # 判断回答是否基于检索结果
            print(f"\n📊 分析结果:")
            print(f"   ✅ 工具是否被调用: {'是' if tool_was_called else '否'}")
            if retrieved_content:
                # 检查回答中是否包含检索到的内容关键词
                answer_lower = last_message.content.lower()
                retrieved_lower = retrieved_content.lower()[:100]
                # 简单的关键词匹配
                has_retrieved_info = any(word in answer_lower for word in retrieved_lower.split()[:10] if len(word) > 3)
                print(f"   ✅ 回答是否基于检索结果: {'是' if has_retrieved_info else '否（可能自由发挥）'}")
                if not has_retrieved_info:
                    print(f"   ⚠️ 警告: 回答可能包含自由拓展的内容！")
            else:
                print(f"   ⚠️ 回答是否基于检索结果: 无法判断（未检测到工具返回内容）")
        else:
            print(f"\n📝 响应: {last_message}")


if __name__ == "__main__":
    main()
