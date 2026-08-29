# AnythingLLM

- Docker Deployment
  - References
    - https://blog.csdn.net/weixin_44585288/article/details/139344486
    - https://ithelp.ithome.com.tw/articles/10364159
    - https://docs.docker.com/compose/how-tos/environment-variables/set-environment-variables/
  - Usage
    - `docker compose up -d`: Run `docker-compose.yaml` file
    - `docker compose down`: Stop containers in `docker-compose.yaml`
    - `docker compose up --build`: Rebuild changed `docker-compose.yaml` file

## Notes

- How to remove local installed ollama
  - https://github.com/ollama/ollama/blob/main/docs/linux.md

## Issues

- AnythingLLM Directory Permission Must be set to 777
  - https://github.com/Mintplex-Labs/anything-llm/issues/2564
- Container Error: Docker-compose is replaced by docker compose
  - https://docs.fylr.io/for-system-administrators/symptom-and-solution/containerconfig-error

# LM Studio

- AppImage
  - Resources
    - https://lmstudio.ai/

# Docker Container with Nvidia GPU

- Resource:
  - https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/sample-workload.html
