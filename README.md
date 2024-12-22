# DLQuant v2

本项目旨在提供自动化、可扩展的行情监控与交易执行示例，基于 APScheduler 实现定时任务管理，并支持对接券商 API（或其他方式）进行自动交易策略。下述结构和示例文件仅作为参考，可自行调整以满足实际业务需求。

## 项目结构

```
dlquant_v2/
├─ .env                # 环境变量文件(敏感信息不要提交到版本库)
├─ .gitignore          # 忽略规则, 确保 .env 等敏感文件不被提交
├─ logger_config.py    # 全局日志配置文件 (示例中会给出基本配置)
├─ monitor.py          # 主入口, 负责初始化/启动任务、进程等
├─ monitor.bat         # Windows批处理文件 (如有需要)
├─ managers/
│  ├─ process_manager.py
│  └─ task_manager.py
├─ strategies/
│  └─ buying_strategy.py
└─ tasks/
   ├─ data_preprocessing.py
   └─ kafka_consumer.py
```

各模块功能说明：
1. **monitor.py**  
   - 主程序入口，负责初始化日志配置与调度器，注册定时任务并启动主循环。  
   - 可在此处调用后台任务管理（ProcessManager）和任务调度（TaskManager）。  
   - 启动程序后进入阻塞循环，等待定时任务按需触发；如需停止程序，可以通过 Ctrl+C 或其他方式结束。

2. **managers/**  
   - **process_manager.py**：后台进程管理类，可以启动并跟踪进程、避免重复实例，并提供进程终止与资源清理功能。  
   - **task_manager.py**：定时任务管理，基于 APScheduler 实现定时执行（支持 interval 和 cron 触发方式）。通过类 TaskConfig 提供任务配置。

3. **strategies/**  
   - **buying_strategy.py**：与交易相关的示例策略函数，包括行情预测 (predict_market)、异步买入 (buy_stock_async) 以及止损逻辑 (stop_loss_main)。示例中只是大致流程，也可自行扩展或拆分到更多文件。

4. **tasks/**  
   - **data_preprocessing.py**：数据下载与预处理任务示例，比如获取历史行情并进行数据清洗、格式化等。  
   - **kafka_consumer.py**：Kafka 消费示例脚本，长期运行来订阅和处理消息队列中的日志或数据。

5. **logger_config.py**  
   - 全局日志配置文件，设定日志级别、输出格式和输出位置。此外，可以在这里扩展日志到文件或使用更高级的日志管理。

6. **.env / .gitignore**  
   - `.env` 用于存储项目本地配置或敏感信息（如 API key、数据库密码），应将其添加到 `.gitignore` 中，避免在公共仓库泄露机密。  
   - `.gitignore` 根据自身需求添加忽略规则，确保敏感信息或不必要的文件不被提交。

7. **monitor.bat** (可选)  
   - Windows 平台的启动脚本，用于快速执行 `python monitor.py` 或其他命令。Linux/MacOS 可使用 .sh 脚本。

## 快速开始

1. **克隆项目 / 下载代码**  
   克隆本仓库或将代码下载到本地后，参照下述步骤安装依赖、配置环境等。

2. **安装依赖**  
   在终端进入项目目录后，建议使用虚拟环境（如 `venv`）或 Conda 环境：
   ```
   pip install -r requirements.txt
   ```
   若未提供 requirements.txt，可根据项目需要手动安装：  
   - `apscheduler` (定时任务)  
   - `psutil` (进程管理)  
   - …其他依赖
  
3. **环境变量 (.env)**  
   在 `.env` 文件中配置项目相关变量。例如：  
   ```
   LOG_LEVEL=INFO
   BROKER_HOST=localhost
   BROKER_PORT=9092
   ```
   或者更多数据库、第三方接口 Key、用户名密码等。

4. **运行项目**  
   启动项目主程序：  
   ```
   python monitor.py
   ```
   程序将按照 `monitor.py` 中配置的任务计划自动执行定期或一次性任务。也可以根据需要在 `monitor.py` 里增加或调整调度任务。

5. **日志查看**  
   使用默认配置时日志会在控制台输出。若需输出到文件，可在 `logger_config.py` 中添加 FileHandler 或 RotatingFileHandler，并重启服务查看日志文件。

## 扩展与自定义

- 将更多交易策略拆分到 `strategies/` 下不同文件中，如 `stop_loss_strategy.py`、`predict_strategy.py` 等。  
- 在 `tasks/` 下添加更多脚本以完成各种数据处理或集成第三方服务（如数据库、消息队列、ETL 流程）。  
- 在 `monitor.py` 中灵活使用 `TaskManager` 向调度器注册更多定时任务，或根据业务场景自定义 Cron 表达式。  
- 若需要调度更复杂的一次性任务或编排工作流，可与其他任务调度框架集成，或在 TaskManager 中扩展更多功能。

## 注意事项

1. **真实交易风险**  
   - 本示例中的交易策略与数据处理逻辑均为演示用，并不保证盈利或适合实盘使用。  
   - 投资需谨慎，使用前请充分了解策略及其风险。

2. **敏感信息管理**  
   - 确保 API key、数据库密码、服务器地址等机密信息不会在公共仓库泄露。  
   - .env 文件或其他凭据文件应加入到 `.gitignore` 进行保护。

3. **日志量与性能**  
   - 若大量或频繁输出日志，需考虑磁盘空间以及日志归档策略。  
   - APScheduler 任务如涉及大规模数据处理时，可考虑单独拆分到独立服务以减轻主程序压力。

欢迎根据自身需求自由扩展项目功能。如有问题或需求，可在此基础上进行修改或发起讨论。