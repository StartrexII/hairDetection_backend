# Deploying a hair color prediction model

### 1. The development and saving of the model is presented on GitHub(including a description of the idea and the scripts used)

[GitHub](https://github.com/StartrexII/hairColorDetection)

### 2. Create Docker container

```bash
docker build -t app-name .

docker run -p 29:80 app-name
```