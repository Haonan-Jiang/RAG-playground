1. pip install -r requirements.txt
2. 安装WSL Ubuntu
    - wsl --install
3. 安装docker并启用wsl后端：https://learn.microsoft.com/zh-cn/windows/wsl/tutorials/wsl-containers
4. docker启动Milvus：https://milvus.io/docs/install_standalone-docker.md
    - curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
    - bash standalone_embed.sh start
5. 启动ollama服务器
    - ollama serve
6. 依次运行 createCollection, insertPoint, searchTest, search
