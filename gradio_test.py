import gradio as gr
import time
import asyncio
from threading import Lock
from datetime import datetime

# 全局变量
logged_clients = {}  # (ip, port) -> 最后活动时间戳
current_model = "Qwen2-VL-7B-Instruct-GPTQ-Int4"
model_choices = ["Qwen2-VL-7B-Instruct-GPTQ-Int4", "Qwen2-VL-7B-Instruct-GPTQ-Int8"]
lock = Lock()
TIMEOUT = 30  # 客户端超时时间（秒）


def cleanup_clients():
    """定期清理超时客户端"""
    global logged_clients, lock
    current_time = time.time()
    with lock:
        expired = [k for k, v in logged_clients.items() if current_time - v > TIMEOUT]
        for k in expired:
            del logged_clients[k]


async def model_predict(model_name, message, image_path):
    """模拟异步模型预测函数"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    # 模拟处理延迟（不阻塞主线程）
    await asyncio.sleep(0.1)  # 最小延迟保证异步处理
    return f"[{timestamp}] {model_name}回复：已收到您的消息和图片（{image_path}），内容为：{message}"


async def login(username, password, request: gr.Request):
    """异步登录处理"""
    global logged_clients, lock
    client_key = (request.client.host, request.client.port)

    cleanup_clients()

    if username != "admin" or password != "admin123":
        return [
            gr.Column(visible=True),  # login_col保持可见
            gr.Column(visible=False),  # main_col隐藏
            gr.Dropdown(interactive=False),
            gr.Markdown(value="当前模型：未登录"),
            gr.Markdown(value="登录失败：用户名或密码错误", visible=True)
        ]

    with lock:
        logged_clients[client_key] = time.time()
        num_clients = len(logged_clients)

    return [
        gr.Column(visible=False),  # 隐藏登录界面
        gr.Column(visible=True),  # 显示主界面
        gr.Dropdown(value=current_model, interactive=num_clients == 1),
        gr.Markdown(value=f"当前模型：{current_model}"),
        gr.Markdown(visible=False)
    ]


async def switch_model(new_model, request: gr.Request):
    """异步模型切换"""
    global current_model, logged_clients, lock
    client_key = (request.client.host, request.client.port)

    with lock:
        cleanup_clients()
        if client_key not in logged_clients:
            return [
                gr.Dropdown(value=current_model),
                gr.Markdown(value="会话已过期，请重新登录"),
                gr.Column(visible=False),
                gr.Column(visible=True)
            ]

        num_clients = len(logged_clients)
        if num_clients != 1:
            return [
                gr.Dropdown(value=current_model),
                gr.Markdown(value=f"切换失败：当前有{num_clients}个活动客户端"),
                gr.Column(),
                gr.Column()
            ]

        current_model = new_model
        logged_clients[client_key] = time.time()

    return [
        gr.Dropdown(value=current_model),
        gr.Markdown(value=f"当前模型：{current_model}"),
        gr.Column(),
        gr.Column()
    ]


async def chat_message(message, image, chat_history, request: gr.Request):
    """异步聊天处理"""
    global logged_clients, current_model, lock
    client_key = (request.client.host, request.client.port)

    with lock:
        cleanup_clients()
        if client_key not in logged_clients:
            return [
                gr.Column(visible=True),
                gr.Column(visible=False),
                chat_history,
                "",
                gr.Dropdown(interactive=False)
            ]

        logged_clients[client_key] = time.time()
        num_clients = len(logged_clients)

    # 立即更新输入框（不等待模型响应）
    immediate_response = "请求已接收，正在处理..."
    new_history = chat_history + [(message, immediate_response)]

    # 异步获取真实响应
    if image:
        real_response = await model_predict(current_model, message, image)
    else:
        real_response = "请先上传图片再进行对话"

    # 更新最终响应
    final_history = new_history[:-1] + [(message, real_response)]

    return [
        gr.Column(visible=False),
        gr.Column(visible=True),
        final_history,
        "",
        gr.Dropdown(interactive=num_clients == 1)
    ]


def update_components():
    """定期更新组件状态"""
    cleanup_clients()
    num_clients = len(logged_clients)
    return [
        gr.Dropdown(interactive=num_clients == 1),
        gr.Markdown(value=f"当前模型：{current_model}"),
        num_clients
    ]


# 构建界面
with gr.Blocks(title="多客户端模型管理系统", theme=gr.themes.Soft()) as app:
    # 状态组件
    client_count = gr.Number(label="当前在线客户端", visible=False)

    # 登录界面
    login_col = gr.Column(visible=True)
    with login_col:
        gr.Markdown("## 多客户端模型管理系统登录")
        username = gr.Textbox(label="用户名", placeholder="输入admin")
        password = gr.Textbox(label="密码", type="password", placeholder="输入admin123")
        login_btn = gr.Button("登录", variant="primary")
        login_status = gr.Markdown(visible=False)

    # 主界面
    main_col = gr.Column(visible=False)
    with main_col:
        model_status = gr.Markdown()
        with gr.Row():
            with gr.Column(min_width=200):
                model_dropdown = gr.Dropdown(
                    choices=model_choices,
                    label="模型选择",
                    value=current_model,
                    interactive=False
                )
            with gr.Column():
                image_input = gr.Image(
                    type="filepath",
                    label="上传分析图片",
                    height=200
                )

        chatbot = gr.Chatbot(
            label="对话记录",
            avatar_images=(
                "assets/user.png",
                "assets/bot.png"
            ),
            height=400,  type='messages'
        )
        msg = gr.Textbox(label="输入消息", placeholder="输入您的分析请求...")
        submit_btn = gr.Button("发送", variant="primary")
        logout_btn = gr.Button("退出登录")

    # 事件绑定
    login_btn.click(
        login,
        [username, password],
        [login_col, main_col, model_dropdown, model_status, login_status]
    )

    model_dropdown.change(
        switch_model,
        [model_dropdown],
        [model_dropdown, model_status, main_col, login_col]
    )

    submit_btn.click(
        chat_message,
        [msg, image_input, chatbot],
        [login_col, main_col, chatbot, msg, model_dropdown],
        concurrency_limit=10  # 提高并发处理能力
    )

    msg.submit(
        chat_message,
        [msg, image_input, chatbot],
        [login_col, main_col, chatbot, msg, model_dropdown],
        concurrency_limit=10
    )

    logout_btn.click(
        lambda: [gr.Column(visible=True), gr.Column(visible=False)],
        outputs=[login_col, main_col]
    )

    # 定期状态更新
    app.load(
        update_components,
        outputs=[model_dropdown, model_status, client_count]
    )

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        share=False,
        auth=("admin", "admin123")
    )