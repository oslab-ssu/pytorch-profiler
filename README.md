# SNN-DNN Pytorch-Profiler Profiling 
코드 삽입을 통해 pytorch의 여러 연산들을 분석할 수 있습니다.

## 환경 세팅 (Prerequisites)
- **OS:** Ubuntu (WSL2 `5.15.167.4-microsoft-standard` 테스트 완료)
- **Python:** `3.12.3`

### 1. 파이썬 가상환경 및 패키지 설치
```bash
python3 -m venv venv
source venv/bin/activate

# 요구 패키지 설치
pip install -r requirements.txt
```

## 실행 가이드 (Quick Start)
### 단계 1: 쉘 스크립트 실행

```bash
chmod +x src/run.sh
./src/run.sh
```

### 단계 2: 결과확인
json 파일을 gui를 통해 상세 분석이 가능합니다.

#### 실행 요약 결과 예시
<img width="1784" height="420" alt="image" src="https://github.com/user-attachments/assets/b336218a-a6ca-4118-8257-61be8cfd5899" />
<img width="1593" height="697" alt="image" src="https://github.com/user-attachments/assets/f9fdbac5-ded7-4e1a-99b0-bdaa70b55f7a" />
